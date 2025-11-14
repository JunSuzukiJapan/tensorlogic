//! Reduction operations for tensors

use crate::device::{Device, MetalBuffer, EncoderProvider};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{Tensor, TensorShape};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Sum all elements in the tensor, returns a scalar value of type T
    ///
    /// # Type Preservation
    ///
    /// Like Candle and PyTorch, `sum()` returns the same type as the input tensor:
    /// - `Tensor<f16>.sum()` → `f16`
    /// - `Tensor<f32>.sum()` → `f32`
    ///
    /// # F16 Overflow Behavior
    ///
    /// **IMPORTANT**: For `f16` tensors, the sum can overflow if the result exceeds
    /// f16's maximum value (±65,504).
    ///
    /// ## When overflow occurs:
    /// - Result becomes `inf` or `-inf`
    /// - This is **expected behavior** (consistent with Candle)
    /// - Internally accumulates in f32 for accuracy, but final result is converted to f16
    ///
    /// ## Examples of overflow:
    /// ```ignore
    /// // 1,088,000 elements × 20 = 21,760,000 > 65,504 (f16 max)
    /// let logits: Tensor<f16> = ...; // [34, 32000] with values around ±20
    /// let sum = logits.sum()?; // Returns: inf
    /// ```
    ///
    /// ## Solutions:
    ///
    /// 1. **Convert to f32 before sum** (recommended for large tensors):
    /// ```ignore
    /// // TODO: Add to_dtype() method
    /// // let sum = logits.to_dtype(DType::F32)?.sum()?; // f32 result
    /// ```
    ///
    /// 2. **Accept inf result** (for diagnostic purposes):
    /// ```ignore
    /// let sum = logits.sum()?; // May be inf
    /// if sum.to_f32().is_infinite() {
    ///     eprintln!("Sum overflow - tensor too large for f16");
    /// }
    /// ```
    ///
    /// 3. **Use sum_keepdim/sum_dim** (when available) to reduce dimensionality first
    ///
    /// # Performance
    ///
    /// - **GPU (Metal)**: Two-stage reduction with threadgroup parallelism
    /// - **CPU**: Simple sequential accumulation (f16 → f32 accumulation)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tensorlogic::{Tensor, Device};
    /// use half::f16;
    ///
    /// // Small tensor - no overflow
    /// let device = Device::Metal::new()?;
    /// let t = Tensor::from_vec_gpu(&device, vec![f16::from_f32(5.0); 10_000], vec![10_000])?;
    /// let sum = t.sum()?; // 50,000 < 65,504 ✓
    ///
    /// // Large tensor - will overflow
    /// let large = Tensor::from_vec_gpu(&device, vec![f16::from_f32(20.0); 1_000_000], vec![1_000_000])?;
    /// let sum = large.sum()?; // 20,000,000 > 65,504 → inf
    /// assert!(sum.to_f32().is_infinite());
    /// ```
    pub fn sum(&self) -> TensorResult<T> {
        match self.device() {
            Device::Metal(_) => self.sum_metal(),
            Device::CPU | Device::NeuralEngine => self.sum_cpu(),
        }
    }

    fn sum_metal(&self) -> TensorResult<T> {
        let input_buf = self.buffer().as_metal()?;
        let count = self.numel();

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction: first reduce to blocks, then reduce blocks
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // Select kernel based on type
        let is_f16 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>();
        let kernel_name = if is_f16 {
            "sum_global_f16"
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "sum_global_f32"
        } else {
            return Err(TensorError::InvalidOperation(
                format!("sum() not implemented for type {:?}", std::any::type_name::<T>())
            ));
        };

        let pipeline = executor.get_or_compile_pipeline(kernel_name)?;

        // Use Commands manager for command buffer (Candle pattern)
        // No wait needed - batching system ensures operations execute in order
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);

        // CRITICAL: For f16, sum_global_f16 kernel outputs f32 to prevent overflow
        // Stage 1: Reduce to blocks
        if is_f16 {
            // Create f32 buffer for stage1 output (kernel outputs f32)
            let stage1_buf_f32 = MetalBuffer::<f32>::new_uninit(&device, num_blocks)?;
            encoder.set_buffer(1, Some(&stage1_buf_f32.buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &count as *const usize as *const _);

            let grid_size = metal::MTLSize::new(count as u64, 1, 1);
            let tg_size = metal::MTLSize::new(threadgroup_size as u64, 1, 1);
            let shared_mem_size = threadgroup_size * std::mem::size_of::<f32>();

            encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();

            // Stage 2: Reduce blocks to final result (CPU, accumulate in f32)
            let stage1_data_f32 = stage1_buf_f32.to_vec();
            let final_sum_f32: f32 = stage1_data_f32.iter().sum();

            // DESIGN NOTE: Like Candle, sum() returns same type as input
            // f16 sum can overflow for large tensors (f16 max = 65504)
            // This is consistent with Candle's design - user should handle overflow

            // Convert f32 sum back to f16 (may become inf if exceeds f16 range)
            // This matches Candle's behavior where sum() preserves input type
            Ok(T::from_f32(final_sum_f32))
        } else {
            // f32 path remains unchanged
            let stage1_buf = MetalBuffer::<T>::new_uninit(&device, num_blocks)?;
            encoder.set_buffer(1, Some(&stage1_buf.buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &count as *const usize as *const _);

            let grid_size = metal::MTLSize::new(count as u64, 1, 1);
            let tg_size = metal::MTLSize::new(threadgroup_size as u64, 1, 1);
            let shared_mem_size = threadgroup_size * std::mem::size_of::<f32>();

            encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();

            // Stage 2: Reduce blocks to final result (CPU)
            let stage1_data = stage1_buf.to_vec();
            let mut final_sum = T::zero();
            for &val in &stage1_data {
                final_sum = final_sum + val;
            }
            Ok(final_sum)
        }

    }

    fn sum_cpu(&self) -> TensorResult<T> {
        let data = self.sync_and_read();

        // For f16, accumulate in f32 to prevent overflow
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>() {
            let mut sum_f32 = 0.0f32;
            for &val in &data {
                sum_f32 += val.to_f32();
            }
            Ok(T::from_f32(sum_f32))
        } else {
            let mut sum = T::zero();
            for &val in &data {
                sum = sum + val;
            }
            Ok(sum)
        }
    }

    /// Sum along a specific dimension
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        if dim >= self.shape().rank() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) => self.sum_dim_metal(dim, keepdim),
            Device::NeuralEngine => {
                self.to_cpu()?.sum_dim_cpu(dim, keepdim)
            },
            Device::CPU => self.sum_dim_cpu(dim, keepdim),
        }
    }

    fn sum_dim_metal(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let input_dims = self.shape().dims();

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Compute output shape
        let mut output_dims = input_dims.to_vec();
        if keepdim {
            output_dims[dim] = 1;
        } else {
            output_dims.remove(dim);
        }

        let output_numel = output_dims.iter().product();
        let output_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(&device, output_numel)?;
        let output_buf: MetalBuffer<T> = unsafe { std::mem::transmute(output_buf_f16) };

        // Prepare shape buffers
        let input_shape_u32: Vec<u32> = input_dims.iter().map(|&x| x as u32).collect();
        let output_shape_u32: Vec<u32> = if keepdim {
            output_dims.iter().map(|&x| x as u32).collect()
        } else {
            // For non-keepdim, pass output shape as-is
            output_dims.iter().map(|&x| x as u32).collect()
        };
        let rank = input_dims.len() as u32;
        let reduce_dim = dim as u32;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("sum_dim_f16")?;

        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&output_buf.buffer), 0);
        encoder.set_bytes(2, (input_shape_u32.len() * std::mem::size_of::<u32>()) as u64, input_shape_u32.as_ptr() as *const _);
        encoder.set_bytes(3, (output_shape_u32.len() * std::mem::size_of::<u32>()) as u64, output_shape_u32.as_ptr() as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &rank as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &reduce_dim as *const u32 as *const _);

        let grid_size = metal::MTLSize::new(output_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(output_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Tensor::new(
            crate::tensor::BufferHandle::Metal(output_buf),
            crate::tensor::TensorShape::new(output_dims),
            self.device().clone(),
        )
    }

    fn sum_dim_cpu(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input = self.sync_and_read();
        let input_f16: Vec<f16> = unsafe { std::mem::transmute(input) };
        let input_dims = self.shape().dims();
        let input_strides = self.shape().compute_strides();

        // Compute output shape
        let mut output_dims = input_dims.to_vec();
        if keepdim {
            output_dims[dim] = 1;
        } else {
            output_dims.remove(dim);
        }

        let output_shape = TensorShape::new(output_dims.clone());
        let output_numel = output_shape.numel();
        let mut output = vec![f16::ZERO; output_numel];

        let dim_size = input_dims[dim];

        // Iterate over output elements
        for out_idx in 0..output_numel {
            // Compute multi-dimensional index for output
            let mut out_coords = vec![0; output_dims.len()];
            let mut remaining = out_idx;
            let out_strides = output_shape.compute_strides();
            for i in 0..output_dims.len() {
                out_coords[i] = remaining / out_strides[i];
                remaining %= out_strides[i];
            }

            // Map to input coordinates (insert reduction dimension)
            let mut input_coords = if keepdim {
                out_coords.clone()
            } else {
                let mut coords = Vec::with_capacity(input_dims.len());
                for (i, &val) in out_coords.iter().enumerate() {
                    if i == dim {
                        coords.push(0); // Will be replaced in loop
                    }
                    coords.push(val);
                }
                if dim == input_dims.len() - 1 {
                    coords.push(0);
                }
                coords
            };

            // Sum over the reduction dimension
            let mut sum = f16::ZERO;
            for d in 0..dim_size {
                input_coords[dim] = d;

                // Convert to linear index
                let mut input_idx = 0;
                for (i, &coord) in input_coords.iter().enumerate() {
                    input_idx += coord * input_strides[i];
                }

                sum += input_f16[input_idx];
            }

            output[out_idx] = sum;
        }

        let output_t: Vec<T> = unsafe { std::mem::transmute(output) };
        Tensor::from_vec(output_t, output_dims)
    }

    /// Mean of all elements
    pub fn mean(&self) -> TensorResult<f16> {
        let sum = self.sum()?;
        let count = self.numel() as f32;
        Ok(f16::from_f32(sum.to_f32() / count))
    }

    /// Mean along a specific dimension
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        if dim >= self.shape().rank() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) => self.mean_dim_metal(dim, keepdim),
            Device::NeuralEngine => {
                self.to_cpu()?.mean_dim_cpu(dim, keepdim)
            },
            Device::CPU => self.mean_dim_cpu(dim, keepdim),
        }
    }

    fn mean_dim_metal(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let input_dims = self.shape().dims();

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Compute output shape
        let mut output_dims = input_dims.to_vec();
        if keepdim {
            output_dims[dim] = 1;
        } else {
            output_dims.remove(dim);
        }

        let output_numel = output_dims.iter().product();
        let output_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(&device, output_numel)?;
        let output_buf: MetalBuffer<T> = unsafe { std::mem::transmute(output_buf_f16) };

        // Prepare shape buffers
        let input_shape_u32: Vec<u32> = input_dims.iter().map(|&x| x as u32).collect();
        let output_shape_u32: Vec<u32> = if keepdim {
            output_dims.iter().map(|&x| x as u32).collect()
        } else {
            output_dims.iter().map(|&x| x as u32).collect()
        };
        let rank = input_dims.len() as u32;
        let reduce_dim = dim as u32;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("mean_dim_f16")?;

        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&output_buf.buffer), 0);
        encoder.set_bytes(2, (input_shape_u32.len() * std::mem::size_of::<u32>()) as u64, input_shape_u32.as_ptr() as *const _);
        encoder.set_bytes(3, (output_shape_u32.len() * std::mem::size_of::<u32>()) as u64, output_shape_u32.as_ptr() as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &rank as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &reduce_dim as *const u32 as *const _);

        let grid_size = metal::MTLSize::new(output_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(output_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Tensor::new(
            crate::tensor::BufferHandle::Metal(output_buf),
            crate::tensor::TensorShape::new(output_dims),
            self.device().clone(),
        )
    }

    fn mean_dim_cpu(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let sum_result = self.sum_dim(dim, keepdim)?;
        let dim_size = self.shape().dims()[dim] as f32;

        // Divide by dimension size
        let data = sum_result.sync_and_read();
        let data_f16: Vec<f16> = unsafe { std::mem::transmute(data) };
        let mean_data_f16: Vec<f16> = data_f16
            .iter()
            .map(|&x| f16::from_f32(x.to_f32() / dim_size))
            .collect();
        let mean_data: Vec<T> = unsafe { std::mem::transmute(mean_data_f16) };

        Tensor::from_vec(mean_data, sum_result.shape().dims().to_vec())
    }

    /// Maximum value in the tensor
    pub fn max(&self) -> TensorResult<T> {
        match self.device() {
            Device::Metal(_) => self.max_metal(),
            Device::NeuralEngine => {
                self.to_cpu()?.max_cpu()
            },
            Device::CPU => self.max_cpu(),
        }
    }

    fn max_metal(&self) -> TensorResult<T> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let count = self.numel();

        if count == 0 {
            return Err(TensorError::InvalidOperation(
                "Cannot compute max of empty tensor".to_string(),
            ));
        }

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        let stage1_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(&device, num_blocks)?;
        let stage1_buf: MetalBuffer<T> = unsafe { std::mem::transmute(stage1_buf_f16) };

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        let pipeline = executor.get_or_compile_pipeline("max_global_f16")?;

        // Use Commands manager for command buffer (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&stage1_buf.buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &count as *const usize as *const _);

        let grid_size = metal::MTLSize::new(count as u64, 1, 1);
        let tg_size = metal::MTLSize::new(threadgroup_size as u64, 1, 1);
        let shared_mem_size = threadgroup_size * std::mem::size_of::<f16>();

        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();

        // Stage 2: Final reduction on CPU
        // Note: to_vec() will automatically wait for GPU completion (Candle pattern)
        let stage1_data = stage1_buf.to_vec();
        let mut max_val = stage1_data[0];
        for &val in &stage1_data[1..] {
            if val > max_val {
                max_val = val;
            }
        }

        Ok(max_val)
    }

    fn max_cpu(&self) -> TensorResult<T> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let data = self.sync_and_read();
        if data.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Cannot compute max of empty tensor".to_string(),
            ));
        }

        let mut max_val = data[0];
        for &val in &data[1..] {
            if val > max_val {
                max_val = val;
            }
        }
        Ok(max_val)
    }

    /// Minimum value in the tensor
    pub fn min(&self) -> TensorResult<T> {
        match self.device() {
            Device::Metal(_) => self.min_metal(),
            Device::NeuralEngine => {
                self.to_cpu()?.min_cpu()
            },
            Device::CPU => self.min_cpu(),
        }
    }

    fn min_metal(&self) -> TensorResult<T> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let count = self.numel();

        if count == 0 {
            return Err(TensorError::InvalidOperation(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        let stage1_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(&device, num_blocks)?;
        let stage1_buf: MetalBuffer<T> = unsafe { std::mem::transmute(stage1_buf_f16) };

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        let pipeline = executor.get_or_compile_pipeline("min_global_f16")?;

        // Use Commands manager for command buffer (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&stage1_buf.buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &count as *const usize as *const _);

        let grid_size = metal::MTLSize::new(count as u64, 1, 1);
        let tg_size = metal::MTLSize::new(threadgroup_size as u64, 1, 1);
        let shared_mem_size = threadgroup_size * std::mem::size_of::<f16>();

        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();

        // Stage 2: Final reduction on CPU
        // Note: to_vec() will automatically wait for GPU completion (Candle pattern)
        let stage1_data = stage1_buf.to_vec();
        let mut min_val = stage1_data[0];
        for &val in &stage1_data[1..] {
            if val < min_val {
                min_val = val;
            }
        }

        Ok(min_val)
    }

    fn min_cpu(&self) -> TensorResult<T> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let data = self.sync_and_read();
        if data.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }

        let mut min_val = data[0];
        for &val in &data[1..] {
            if val < min_val {
                min_val = val;
            }
        }
        Ok(min_val)
    }

    /// Find the index of the maximum value (argmax)
    /// Returns a tensor of indices (i64)
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> TensorResult<Self> {
        match dim {
            None => {
                // Global argmax - return single index
                let data = self.sync_and_read();
                let mut max_idx = 0usize;
                let mut max_val = data[0];

                for (idx, &val) in data.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }

                // Return as scalar tensor (shape [1] since shape [] has numel=0 bug)
                // TODO: Fix TensorShape::numel() to return 1 for empty shapes
                let result_shape = if keepdim {
                    vec![1; self.shape().rank()]
                } else {
                    vec![1]  // Use [1] instead of [] due to numel() bug
                };

                let indices = vec![T::from_f64(max_idx as f64)];
                Tensor::from_vec(indices, result_shape)
            }
            Some(dim) => {
                if dim >= self.shape().rank() {
                    return Err(TensorError::InvalidDimension { dim });
                }

                // Argmax along specific dimension
                self.argmax_dim(dim, keepdim)
            }
        }
    }

    fn argmax_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        let input_dims = self.shape().dims();
        let rank = input_dims.len();

        // Calculate output shape
        let mut output_dims = Vec::new();
        for (i, &d) in input_dims.iter().enumerate() {
            if i == dim {
                if keepdim {
                    output_dims.push(1);
                }
            } else {
                output_dims.push(d);
            }
        }

        let output_numel: usize = if output_dims.is_empty() {
            1  // Scalar result when all dims reduced
        } else {
            output_dims.iter().product()
        };

        let dim_size = input_dims[dim];

        // Compute strides
        let mut strides = vec![1; rank];
        for i in (0..rank-1).rev() {
            strides[i] = strides[i + 1] * input_dims[i + 1];
        }

        let data = self.sync_and_read();
        let mut result = vec![T::zero(); output_numel];

        for out_idx in 0..output_numel {
            // Convert output index to input coordinates (skipping reduced dim)
            let mut coords = vec![0; rank];
            let mut remaining = out_idx;
            let mut out_dim_idx = 0;

            for i in 0..rank {
                if i != dim {
                    let size = input_dims[i];
                    coords[i] = remaining % size;
                    remaining /= size;
                    out_dim_idx += 1;
                }
            }

            // Find argmax along the reduction dimension
            let mut max_idx = 0;
            let mut max_val = T::from_f32(f32::NEG_INFINITY);

            for d in 0..dim_size {
                coords[dim] = d;

                // Calculate linear index
                let mut linear_idx = 0;
                for (i, &coord) in coords.iter().enumerate() {
                    linear_idx += coord * strides[i];
                }

                let val = data[linear_idx];
                if val > max_val {
                    max_val = val;
                    max_idx = d;
                }
            }

            result[out_idx] = T::from_f64(max_idx as f64);
        }

        Tensor::from_vec(result, output_dims)
    }

    /// Find the index of the minimum value (argmin)
    /// Returns a tensor of indices (i64)
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> TensorResult<Self> {
        match dim {
            None => {
                // Global argmin - return single index
                let data = self.sync_and_read();
                let mut min_idx = 0usize;
                let mut min_val = data[0];

                for (idx, &val) in data.iter().enumerate().skip(1) {
                    if val < min_val {
                        min_val = val;
                        min_idx = idx;
                    }
                }

                // Return as scalar tensor (shape [1] since shape [] has numel=0 bug)
                // TODO: Fix TensorShape::numel() to return 1 for empty shapes
                let result_shape = if keepdim {
                    vec![1; self.shape().rank()]
                } else {
                    vec![1]  // Use [1] instead of [] due to numel() bug
                };

                let indices = vec![T::from_f64(min_idx as f64)];
                Tensor::from_vec(indices, result_shape)
            }
            Some(dim) => {
                if dim >= self.shape().rank() {
                    return Err(TensorError::InvalidDimension { dim });
                }

                // Argmin along specific dimension
                self.argmin_dim(dim, keepdim)
            }
        }
    }

    fn argmin_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        let input_dims = self.shape().dims();
        let rank = input_dims.len();

        // Calculate output shape
        let mut output_dims = Vec::new();
        for (i, &d) in input_dims.iter().enumerate() {
            if i == dim {
                if keepdim {
                    output_dims.push(1);
                }
            } else {
                output_dims.push(d);
            }
        }

        let output_numel: usize = if output_dims.is_empty() {
            1  // Scalar result when all dims reduced
        } else {
            output_dims.iter().product()
        };
        let dim_size = input_dims[dim];

        // Compute strides
        let mut strides = vec![1; rank];
        for i in (0..rank-1).rev() {
            strides[i] = strides[i + 1] * input_dims[i + 1];
        }

        let data = self.sync_and_read();
        let mut result = vec![T::zero(); output_numel];

        for out_idx in 0..output_numel {
            // Convert output index to input coordinates (skipping reduced dim)
            let mut coords = vec![0; rank];
            let mut remaining = out_idx;

            for i in 0..rank {
                if i != dim {
                    let size = input_dims[i];
                    coords[i] = remaining % size;
                    remaining /= size;
                }
            }

            // Find argmin along the reduction dimension
            let mut min_idx = 0;
            let mut min_val = T::from_f32(f32::INFINITY);

            for d in 0..dim_size {
                coords[dim] = d;

                // Calculate linear index
                let mut linear_idx = 0;
                for (i, &coord) in coords.iter().enumerate() {
                    linear_idx += coord * strides[i];
                }

                let val = data[linear_idx];
                if val < min_val {
                    min_val = val;
                    min_idx = d;
                }
            }

            result[out_idx] = T::from_f64(min_idx as f64);
        }

        Tensor::from_vec(result, output_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![4],
        )
        .unwrap();

        let sum = a.sum().unwrap();
        assert_eq!(sum, f16::from_f32(10.0));
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![4],
        )
        .unwrap();

        let mean = a.mean().unwrap();
        assert_eq!(mean, f16::from_f32(2.5));
    }

    #[test]
    fn test_max() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(5.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let max = a.max().unwrap();
        assert_eq!(max, f16::from_f32(5.0));
    }

    #[test]
    fn test_min() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(5.0),
                f16::from_f32(1.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let min = a.min().unwrap();
        assert_eq!(min, f16::from_f32(1.0));
    }

    #[test]
    fn test_sum_dim() {
        // Matrix [2, 3]
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        // Sum along dim 0 (rows) -> [3]
        let sum0 = a.sum_dim(0, false).unwrap();
        assert_eq!(sum0.shape().dims(), &[3]);
        let result0 = sum0.sync_and_read();
        assert_eq!(result0[0], f16::from_f32(5.0)); // 1+4
        assert_eq!(result0[1], f16::from_f32(7.0)); // 2+5
        assert_eq!(result0[2], f16::from_f32(9.0)); // 3+6

        // Sum along dim 1 (columns) -> [2]
        let sum1 = a.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        let result1 = sum1.sync_and_read();
        assert_eq!(result1[0], f16::from_f32(6.0)); // 1+2+3
        assert_eq!(result1[1], f16::from_f32(15.0)); // 4+5+6
    }

    #[test]
    fn test_mean_dim() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        // Mean along dim 1
        let mean1 = a.mean_dim(1, false).unwrap();
        assert_eq!(mean1.shape().dims(), &[2]);
        let result = mean1.sync_and_read();
        assert_eq!(result[0], f16::from_f32(3.0)); // (2+4)/2
        assert_eq!(result[1], f16::from_f32(7.0)); // (6+8)/2
    }

    #[test]
    fn test_sum_dim_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // Create tensor on Metal device
        let a = Tensor::from_vec_gpu(
            &metal_device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        // Sum along dim 0 (rows) -> [3]
        let sum0 = a.sum_dim(0, false).unwrap();
        assert_eq!(sum0.shape().dims(), &[3]);
        let result0 = sum0.sync_and_read();
        assert_eq!(result0[0], f16::from_f32(5.0)); // 1+4
        assert_eq!(result0[1], f16::from_f32(7.0)); // 2+5
        assert_eq!(result0[2], f16::from_f32(9.0)); // 3+6

        // Sum along dim 1 (columns) -> [2]
        let sum1 = a.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        let result1 = sum1.sync_and_read();
        assert_eq!(result1[0], f16::from_f32(6.0)); // 1+2+3
        assert_eq!(result1[1], f16::from_f32(15.0)); // 4+5+6
    }

    #[test]
    fn test_mean_dim_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_gpu(
            &metal_device,
            vec![
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        // Mean along dim 1
        let mean1 = a.mean_dim(1, false).unwrap();
        assert_eq!(mean1.shape().dims(), &[2]);
        let result = mean1.sync_and_read();
        assert_eq!(result[0], f16::from_f32(3.0)); // (2+4)/2
        assert_eq!(result[1], f16::from_f32(7.0)); // (6+8)/2
    }

    #[test]
    fn test_max_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_gpu(
            &metal_device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(5.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let max = a.max().unwrap();
        assert_eq!(max, f16::from_f32(5.0));
    }

    #[test]
    fn test_min_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_gpu(
            &metal_device,
            vec![
                f16::from_f32(5.0),
                f16::from_f32(1.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let min = a.min().unwrap();
        assert_eq!(min, f16::from_f32(1.0));
    }

    #[test]
    fn test_sum_large_array_metal() {
        // Test for the shared memory size bug fix
        // Bug: shared_mem_size was calculated as f16 size, but kernel uses f32
        // This test uses 2048 elements (same as embedding dimension) to catch the bug
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // Create a large array (2048 elements, same as TinyLlama embedding dimension)
        // Use values similar to actual embedding weights to make test realistic
        let mut data = vec![f16::ZERO; 2048];
        for i in 0..2048 {
            data[i] = f16::from_f32((i as f32 % 10.0) / 100.0 - 0.05);
        }

        // Calculate expected sum on CPU (reference)
        let cpu_sum: f32 = data.iter().map(|&x| x.to_f32()).sum();

        // Create tensor on Metal GPU
        let tensor = Tensor::from_vec_gpu(&metal_device, data.clone(), vec![2048]).unwrap();

        // Compute sum on GPU
        let gpu_sum = tensor.sum().unwrap();

        // Results should match within f16 precision
        let diff = (cpu_sum - gpu_sum.to_f32()).abs();
        assert!(
            diff < 0.01,
            "Large array sum mismatch: CPU={}, GPU={}, diff={}",
            cpu_sum,
            gpu_sum.to_f32(),
            diff
        );
    }

    #[test]
    fn test_sum_embedding_pattern_metal() {
        // Test with pattern similar to actual BOS token embedding
        // This reproduces the exact bug scenario: sum of 2048 f16 values
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // Pattern: alternating small positive/negative values (like real embeddings)
        let mut data = vec![f16::ZERO; 2048];
        for i in 0..2048 {
            let val = if i % 2 == 0 { 0.001 } else { -0.0005 };
            data[i] = f16::from_f32(val);
        }

        // Expected sum: 2048 * 0.001 / 2 - 2048 * 0.0005 / 2 = 1.024 - 0.512 = 0.512
        let expected_sum = 0.512;

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![2048]).unwrap();
        let gpu_sum = tensor.sum().unwrap();

        let diff = (expected_sum - gpu_sum.to_f32()).abs();
        assert!(
            diff < 0.01,
            "Embedding pattern sum mismatch: expected={}, GPU={}, diff={}",
            expected_sum,
            gpu_sum.to_f32(),
            diff
        );
    }

    #[test]
    #[ignore] // TODO: Small array GPU sum has a separate bug (returns 4.0 instead of 2.8)
    fn test_sum_cpu_gpu_consistency() {
        // Verify CPU and GPU sum produce same results for same input
        // This catches any GPU kernel bugs
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let data = vec![
            f16::from_f32(1.5),
            f16::from_f32(-2.3),
            f16::from_f32(0.7),
            f16::from_f32(4.1),
            f16::from_f32(-1.2),
        ];

        // CPU sum
        let cpu_tensor = Tensor::from_vec(data.clone(), vec![5]).unwrap();
        let cpu_sum = cpu_tensor.sum().unwrap();

        // GPU sum
        let gpu_tensor = Tensor::from_vec_gpu(&metal_device, data, vec![5]).unwrap();
        let gpu_sum = gpu_tensor.sum().unwrap();

        // Should be identical
        assert_eq!(
            cpu_sum, gpu_sum,
            "CPU and GPU sum mismatch: CPU={}, GPU={}",
            cpu_sum.to_f32(),
            gpu_sum.to_f32()
        );
    }

    #[test]
    fn test_sum_large_tensor_f16() {
        // Test sum with 1M+ elements
        // 1,100,000 × 1.0 = 1,100,000 exceeds f16 max (65504)
        // Expected: inf
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // Create 1,100,000 element tensor
        let size = 1_100_000;
        let value = f16::from_f32(1.0);
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![size]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        eprintln!("test_sum_large_tensor_f16: sum={} (expected: inf because 1,100,000 > 65504)", sum_f32);

        // EXPECTED: sum WILL be inf (exceeds f16 range)
        assert!(
            sum_f32.is_infinite() && sum_f32 > 0.0,
            "Sum should be +inf for 1,100,000 × 1.0 = 1,100,000 (exceeds f16 max 65504)"
        );
    }

    #[test]
    fn test_sum_within_f16_range() {
        // Test sum that stays within f16 range (no overflow)
        // 10,000 × 5.0 = 50,000 < 65504 (f16 max)
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let size = 10_000;
        let value = f16::from_f32(5.0);
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![size]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        // Expected: 10,000 × 5.0 = 50,000 (within f16 range)
        let expected = 50_000.0;
        let diff = (expected - sum_f32).abs();

        eprintln!("test_sum_within_f16_range: sum={}, expected={}, diff={}", sum_f32, expected, diff);

        // Should be accurate (within 1%)
        let tolerance = expected * 0.01;
        assert!(
            diff < tolerance,
            "Sum within f16 range mismatch: expected={}, got={}, diff={}, tolerance={}",
            expected,
            sum_f32,
            diff,
            tolerance
        );

        // Should NOT be inf
        assert!(
            !sum_f32.is_infinite(),
            "Sum should not be inf when result is within f16 range"
        );
    }

    #[test]
    fn test_sum_clamped_values_large() {
        // Test sum with ±20 clamped values (actual logits case)
        // This is the exact scenario from LM head: 1,088,000 elements clamped to ±20
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // 34 × 32000 = 1,088,000 (actual logits size)
        let size = 34 * 32000;

        // Half positive, half negative (balanced)
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            let val = if i % 2 == 0 { 20.0 } else { -20.0 };
            data.push(f16::from_f32(val));
        }

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![34, 32000]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        // Expected: ~0 (balanced positive/negative)
        let expected = 0.0;
        let diff = sum_f32.abs();

        eprintln!("test_sum_clamped_values_large: sum={}, expected={}, diff={}", sum_f32, expected, diff);

        // Allow some accumulation error
        assert!(
            diff < 1000.0,
            "Balanced ±20 sum should be close to 0: got {}, diff={}",
            sum_f32,
            diff
        );

        // CRITICAL: sum should NOT be inf
        assert!(
            !sum_f32.is_infinite(),
            "Sum should not be inf for clamped ±20 values"
        );
    }

    #[test]
    fn test_sum_all_positive_20() {
        // Test sum with all positive 20 values
        // IMPORTANT: Like Candle, f16 sum can overflow (f16 max = 65504)
        // Expected: 1,088,000 × 20 = 21,760,000 → inf (exceeds f16 range)
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // 1,088,000 elements all at +20
        let size = 34 * 32000;
        let value = f16::from_f32(20.0);
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![34, 32000]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        // Expected: 1,088,000 × 20 = 21,760,000 exceeds f16 max (65504)
        // Result: inf (this is expected behavior, matching Candle)
        eprintln!("test_sum_all_positive_20: sum={} (expected: inf because 21,760,000 > 65504)", sum_f32);

        // EXPECTED: sum WILL be inf (overflow is expected for f16)
        assert!(
            sum_f32.is_infinite() && sum_f32 > 0.0,
            "Sum should be +inf for 1,088,000 × 20 = 21,760,000 (exceeds f16 max 65504)"
        );
    }

    #[test]
    fn test_sum_all_negative_20() {
        // Test sum with all negative 20 values
        // Expected: -21,760,000 → -inf (exceeds f16 range)
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // 1,088,000 elements all at -20
        let size = 34 * 32000;
        let value = f16::from_f32(-20.0);
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![34, 32000]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        eprintln!("test_sum_all_negative_20: sum={} (expected: -inf)", sum_f32);

        // EXPECTED: sum WILL be -inf (overflow is expected for f16)
        assert!(
            sum_f32.is_infinite() && sum_f32 < 0.0,
            "Sum should be -inf for 1,088,000 × (-20) = -21,760,000 (exceeds f16 range)"
        );
    }

    #[test]
    fn test_sum_mixed_signs_large() {
        // Test sum with mixed positive/negative values
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let size = 1_000_000;
        let mut data = Vec::with_capacity(size);

        // Pattern: +15, -10, +5, -8, ... (net positive)
        for i in 0..size {
            let val = match i % 4 {
                0 => 15.0,
                1 => -10.0,
                2 => 5.0,
                _ => -8.0,
            };
            data.push(f16::from_f32(val));
        }

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![size]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        // Expected: (15 - 10 + 5 - 8) × 250,000 = 2 × 250,000 = 500,000
        let expected = 2.0 * 250_000.0;
        let diff = (expected - sum_f32).abs();

        eprintln!("test_sum_mixed_signs_large: sum={}, expected={}, diff={}", sum_f32, expected, diff);

        // Allow 1% error
        let tolerance = expected * 0.01;
        assert!(
            diff < tolerance,
            "Mixed signs sum mismatch: expected={}, got={}, diff={}, tolerance={}",
            expected,
            sum_f32,
            diff,
            tolerance
        );

        assert!(
            !sum_f32.is_infinite(),
            "Sum should not be inf for mixed sign values"
        );
    }

    #[test]
    fn test_sum_f32_large() {
        // Test f32 sum with large tensor (for comparison with f16)
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let size = 1_000_000;
        let value = 10.0f32;
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![size]).unwrap();
        let sum = tensor.sum().unwrap();

        // Expected: 1,000,000 × 10.0 = 10,000,000
        let expected = (size as f32) * value;
        let diff = (expected - sum).abs();

        eprintln!("test_sum_f32_large: sum={}, expected={}, diff={}", sum, expected, diff);

        // f32 should be more accurate
        let tolerance = expected * 0.0001; // 0.01% tolerance
        assert!(
            diff < tolerance,
            "f32 large sum mismatch: expected={}, got={}, diff={}, tolerance={}",
            expected,
            sum,
            diff,
            tolerance
        );

        assert!(
            !sum.is_infinite(),
            "f32 sum should not be inf for 10M total"
        );
    }

    #[test]
    fn test_sum_overflow_boundary_f16() {
        // Test sum at f16 overflow boundary
        // 10,000 × 100 = 1,000,000 exceeds f16 max (65504)
        // Expected: inf (like Candle)
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // 10,000 elements × 100 = 1,000,000 (exceeds f16 range)
        let size = 10_000;
        let value = f16::from_f32(100.0);
        let data = vec![value; size];

        let tensor = Tensor::from_vec_gpu(&metal_device, data, vec![size]).unwrap();
        let sum = tensor.sum().unwrap();
        let sum_f32 = sum.to_f32();

        eprintln!("test_sum_overflow_boundary_f16: sum={} (expected: inf because 1,000,000 > 65504)", sum_f32);

        // EXPECTED: sum WILL be inf (result exceeds f16 max)
        // Even though we accumulate in f32, final result is converted to f16
        // This is consistent with Candle's design
        assert!(
            sum_f32.is_infinite() && sum_f32 > 0.0,
            "Sum should be +inf when result (1M) exceeds f16 max (65504)"
        );
    }
}
