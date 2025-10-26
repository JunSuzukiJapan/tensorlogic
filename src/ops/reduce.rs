//! Reduction operations for tensors

use crate::device::{Device, MetalBuffer};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{Tensor, TensorShape};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Sum all elements in the tensor
    pub fn sum(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) => self.sum_metal(),
            Device::CPU | Device::NeuralEngine => self.sum_cpu(),
        }
    }

    fn sum_metal(&self) -> TensorResult<f16> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let count = self.numel();

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load reduction shaders
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/reductions.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction: first reduce to blocks, then reduce blocks
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        // Stage 1: Reduce to blocks
        let stage1_buf = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), num_blocks)?;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("sum_global_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Stage 2: Reduce blocks to final result (CPU for simplicity)
        // Note: For small num_blocks (<256), CPU reduction is faster than launching
        // another GPU kernel due to ~0.15-0.20ms kernel launch overhead
        let stage1_data = stage1_buf.to_vec();
        let mut final_sum = f16::ZERO;
        for &val in &stage1_data {
            final_sum += val;
        }

        Ok(final_sum)
    }

    fn sum_cpu(&self) -> TensorResult<f16> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let data = self.to_vec();
        let mut sum = f16::ZERO;
        for &val in &data {
            sum += val;
        }
        Ok(sum)
    }

    /// Sum along a specific dimension
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        if dim >= self.shape().rank() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) => self.sum_dim_metal(dim, keepdim),
            Device::NeuralEngine => self.to_cpu()?.sum_dim_cpu(dim, keepdim),
            Device::CPU => self.sum_dim_cpu(dim, keepdim),
        }
    }

    fn sum_dim_metal(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
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
            let shader_source = include_str!("../../shaders/reductions.metal");
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
        let output_buf = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), output_numel)?;

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

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            crate::tensor::BufferHandle::Metal(output_buf),
            crate::tensor::TensorShape::new(output_dims),
            self.device().clone(),
        )
    }

    fn sum_dim_cpu(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input = self.to_vec();
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

                sum += input[input_idx];
            }

            output[out_idx] = sum;
        }

        Tensor::from_vec(output, output_dims)
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
            Device::NeuralEngine => self.to_cpu()?.mean_dim_cpu(dim, keepdim),
            Device::CPU => self.mean_dim_cpu(dim, keepdim),
        }
    }

    fn mean_dim_metal(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
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
            let shader_source = include_str!("../../shaders/reductions.metal");
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
        let output_buf = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), output_numel)?;

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

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            crate::tensor::BufferHandle::Metal(output_buf),
            crate::tensor::TensorShape::new(output_dims),
            self.device().clone(),
        )
    }

    fn mean_dim_cpu(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let sum_result = self.sum_dim(dim, keepdim)?;
        let dim_size = self.shape().dims()[dim] as f32;

        // Divide by dimension size
        let data = sum_result.to_vec();
        let mean_data: Vec<f16> = data
            .iter()
            .map(|&x| f16::from_f32(x.to_f32() / dim_size))
            .collect();

        Tensor::from_vec(mean_data, sum_result.shape().dims().to_vec())
    }

    /// Maximum value in the tensor
    pub fn max(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) => self.max_metal(),
            Device::NeuralEngine => self.to_cpu()?.max_cpu(),
            Device::CPU => self.max_cpu(),
        }
    }

    fn max_metal(&self) -> TensorResult<f16> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
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
            let shader_source = include_str!("../../shaders/reductions.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        let stage1_buf = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), num_blocks)?;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("max_global_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Stage 2: Final reduction on CPU
        let stage1_data = stage1_buf.to_vec();
        let mut max_val = stage1_data[0];
        for &val in &stage1_data[1..] {
            if val > max_val {
                max_val = val;
            }
        }

        Ok(max_val)
    }

    fn max_cpu(&self) -> TensorResult<f16> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let data = self.to_vec();
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
    pub fn min(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) => self.min_metal(),
            Device::NeuralEngine => self.to_cpu()?.min_cpu(),
            Device::CPU => self.min_cpu(),
        }
    }

    fn min_metal(&self) -> TensorResult<f16> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
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
            let shader_source = include_str!("../../shaders/reductions.metal");
            device.load_library(shader_source)?;
        }

        // Two-stage reduction
        let threadgroup_size = 256;
        let num_blocks = (count + threadgroup_size - 1) / threadgroup_size;

        let stage1_buf = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), num_blocks)?;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("min_global_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Stage 2: Final reduction on CPU
        let stage1_data = stage1_buf.to_vec();
        let mut min_val = stage1_data[0];
        for &val in &stage1_data[1..] {
            if val < min_val {
                min_val = val;
            }
        }

        Ok(min_val)
    }

    fn min_cpu(&self) -> TensorResult<f16> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let data = self.to_vec();
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
                let data = self.to_vec();
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

                let indices = vec![half::f16::from_f64(max_idx as f64)];
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

        let data = self.to_vec();
        let mut result = vec![half::f16::ZERO; output_numel];

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
            let mut max_val = half::f16::NEG_INFINITY;

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

            result[out_idx] = half::f16::from_f64(max_idx as f64);
        }

        Tensor::from_vec(result, output_dims)
    }

    /// Find the index of the minimum value (argmin)
    /// Returns a tensor of indices (i64)
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> TensorResult<Self> {
        match dim {
            None => {
                // Global argmin - return single index
                let data = self.to_vec();
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

                let indices = vec![half::f16::from_f64(min_idx as f64)];
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

        let data = self.to_vec();
        let mut result = vec![half::f16::ZERO; output_numel];

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
            let mut min_val = half::f16::INFINITY;

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

            result[out_idx] = half::f16::from_f64(min_idx as f64);
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
        let result0 = sum0.to_vec();
        assert_eq!(result0[0], f16::from_f32(5.0)); // 1+4
        assert_eq!(result0[1], f16::from_f32(7.0)); // 2+5
        assert_eq!(result0[2], f16::from_f32(9.0)); // 3+6

        // Sum along dim 1 (columns) -> [2]
        let sum1 = a.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        let result1 = sum1.to_vec();
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
        let result = mean1.to_vec();
        assert_eq!(result[0], f16::from_f32(3.0)); // (2+4)/2
        assert_eq!(result[1], f16::from_f32(7.0)); // (6+8)/2
    }

    #[test]
    fn test_sum_dim_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        // Create tensor on Metal device
        let a = Tensor::from_vec_metal(
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
        let result0 = sum0.to_vec();
        assert_eq!(result0[0], f16::from_f32(5.0)); // 1+4
        assert_eq!(result0[1], f16::from_f32(7.0)); // 2+5
        assert_eq!(result0[2], f16::from_f32(9.0)); // 3+6

        // Sum along dim 1 (columns) -> [2]
        let sum1 = a.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        let result1 = sum1.to_vec();
        assert_eq!(result1[0], f16::from_f32(6.0)); // 1+2+3
        assert_eq!(result1[1], f16::from_f32(15.0)); // 4+5+6
    }

    #[test]
    fn test_mean_dim_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_metal(
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
        let result = mean1.to_vec();
        assert_eq!(result[0], f16::from_f32(3.0)); // (2+4)/2
        assert_eq!(result[1], f16::from_f32(7.0)); // (6+8)/2
    }

    #[test]
    fn test_max_metal() {
        use crate::device::MetalDevice;

        let metal_device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_metal(
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

        let a = Tensor::from_vec_metal(
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
}
