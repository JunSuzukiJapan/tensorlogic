//! Indexing operations: gather and scatter

use crate::device::{Device, MetalBuffer};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor, TokenIdArray};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Token embedding lookup for language models
    ///
    /// Performs embedding lookup from a weight matrix given token IDs.
    /// After GGUF dimension reversal, expects PyTorch format: weight[vocab_size, d_model]
    ///
    /// # Arguments
    /// * `token_ids` - Tensor of token indices with shape [batch, seq_len] or [seq_len]
    ///
    /// # Returns
    /// Embedding tensor with shape [batch, seq_len, d_model] or [seq_len, d_model]
    ///
    /// # Example
    /// ```ignore
    /// let weight = Tensor::from_vec_metal(&device, data, vec![vocab_size, d_model])?;
    /// let token_ids = Tensor::from_vec_metal(&device, ids, vec![batch, seq_len])?;
    /// let embeddings = weight.embedding(&token_ids)?;
    /// // embeddings.shape() == [batch, seq_len, d_model]
    /// ```
    pub fn embedding(&self, token_ids: &Tensor<T>) -> TensorResult<Self> {
        // self is weight with shape [vocab_size, d_model] (PyTorch format after GGUF reverse)
        // token_ids has shape [batch, seq_len] or [seq_len]
        let weight_dims = self.dims();
        let token_dims = token_ids.dims();

        if weight_dims.len() != 2 {
            return Err(TensorError::InvalidOperation(
                format!("Embedding weight must be 2D, got shape {:?}", weight_dims)
            ));
        }

        let vocab_size = weight_dims[0];
        let d_model = weight_dims[1];

        // Validate token IDs are in range [0, vocab_size)
        let token_data = token_ids.to_vec();
        for &token_id in &token_data {
            let id = token_id.to_f32() as usize;
            if id >= vocab_size {
                return Err(TensorError::InvalidOperation(
                    format!("Token ID {} out of range for vocab size {}", id, vocab_size)
                ));
            }
        }

        // Calculate output shape
        let mut output_shape = token_dims.to_vec();
        output_shape.push(d_model);

        // Perform embedding lookup
        let num_tokens: usize = token_dims.iter().product();
        let weight_data = self.to_vec();
        let mut output = Vec::with_capacity(num_tokens * d_model);

        for &token_id in &token_data {
            let id = token_id.to_f32() as usize;

            // Extract row `id` from weight matrix [vocab_size, d_model]
            // Row-major layout: row `id` contains d_model consecutive elements
            for col in 0..d_model {
                let idx = id * d_model + col;
                output.push(weight_data[idx]);
            }
        }

        // Create output tensor on same device as weight
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, output, output_shape),
            _ => Tensor::from_vec(output, output_shape),
        }
    }

    /// Token embedding lookup for language models (from TokenIdArray)
    ///
    /// Performs embedding lookup from a weight matrix given token IDs without f16 precision loss.
    /// After GGUF dimension reversal, expects PyTorch format: weight[vocab_size, d_model]
    ///
    /// # Arguments
    /// * `token_ids` - TokenIdArray of token indices with shape [batch, seq_len] or [seq_len]
    ///
    /// # Returns
    /// Embedding tensor with shape [batch, seq_len, d_model] or [seq_len, d_model]
    ///
    /// # Example
    /// ```ignore
    /// let weight = Tensor::from_vec_metal(&device, data, vec![vocab_size, d_model])?;
    /// let token_ids = TokenIdArray::new(vec![1, 20358, 20359]);
    /// let embeddings = weight.embedding_from_token_ids(&token_ids)?;
    /// // embeddings.shape() == [3, d_model]
    /// ```
    pub fn embedding_from_token_ids(&self, token_ids: &TokenIdArray) -> TensorResult<Self> {
        use crate::tensor::TensorLike;

        // self is weight with shape [vocab_size, d_model] (PyTorch format after GGUF reverse)
        // token_ids has shape [batch, seq_len] or [seq_len]
        let weight_dims = self.dims();
        let token_dims = token_ids.shape();

        if weight_dims.len() != 2 {
            return Err(TensorError::InvalidOperation(
                format!("Embedding weight must be 2D, got shape {:?}", weight_dims)
            ));
        }

        let vocab_size = weight_dims[0];
        let d_model = weight_dims[1];

        // Get token IDs directly as i64 (no precision loss)
        let token_data = token_ids.data();

        // Validate token IDs are in range [0, vocab_size)
        for &token_id in token_data {
            let id = token_id as usize;
            if id >= vocab_size {
                return Err(TensorError::InvalidOperation(
                    format!("Token ID {} out of range for vocab size {}", id, vocab_size)
                ));
            }
        }

        // Calculate output shape
        let mut output_shape = token_dims.clone();
        output_shape.push(d_model);

        // Perform embedding lookup
        let num_tokens: usize = token_dims.iter().product();
        let weight_data = self.to_vec();
        let mut output = Vec::with_capacity(num_tokens * d_model);

        for &token_id in token_data {
            let id = token_id as usize;

            // Extract row `id` from weight matrix [vocab_size, d_model]
            // Row-major layout: row `id` contains d_model consecutive elements
            for col in 0..d_model {
                let idx = id * d_model + col;
                output.push(weight_data[idx]);
            }
        }

        // Create output tensor on same device as weight
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, output, output_shape),
            _ => Tensor::from_vec(output, output_shape),
        }
    }

    /// Gather values along an axis according to indices
    ///
    /// Gathers values from `self` along dimension `dim` using `indices`.
    /// The output shape is the same as indices shape, with dim replaced by self.dims()[dim].
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to gather
    /// * `indices` - Tensor of indices (must be integer-like, stored as f16)
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Tensor::from_vec(
    ///     vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
    ///     vec![3]
    /// )?;
    /// let indices = Tensor::from_vec(
    ///     vec![f16::from_f32(0.0), f16::from_f32(2.0), f16::from_f32(1.0)],
    ///     vec![3]
    /// )?;
    /// let result = x.gather(0, &indices)?;
    /// // result = [1.0, 3.0, 2.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn gather<I: FloatType>(&self, dim: usize, indices: &Tensor<I>) -> TensorResult<Self> {
        if dim >= self.dims().len() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) if self.buffer().is_metal() && indices.buffer().is_metal() => {
                self.gather_metal(dim, indices)
            }
            _ => self.gather_cpu(dim, indices),
        }
    }

    /// Metal GPU implementation of gather
    fn gather_metal<I: FloatType>(&self, dim: usize, indices: &Tensor<I>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let indices_buf = indices.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        let self_dims = self.dims();
        let indices_dims = indices.dims();
        let ndim = self_dims.len();
        let indices_ndim = indices_dims.len();

        // Calculate strides for input
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            input_strides[i] = input_strides[i + 1] * self_dims[i + 1];
        }

        // Calculate strides for output (based on indices dimensions)
        let mut output_strides = vec![1usize; indices_ndim];
        for i in (0..indices_ndim - 1).rev() {
            output_strides[i] = output_strides[i + 1] * indices_dims[i + 1];
        }

        // Create output buffer
        let output_numel: usize = indices_dims.iter().product();
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), output_numel)?;

        // Create parameter buffers
        let input_strides_f16: Vec<f16> = input_strides.iter().map(|&s| f16::from_f32(s as f32)).collect();
        let output_strides_f16: Vec<f16> = output_strides.iter().map(|&s| f16::from_f32(s as f32)).collect();
        let input_dims_f16: Vec<f16> = self_dims.iter().map(|&d| f16::from_f32(d as f32)).collect();
        let output_dims_f16: Vec<f16> = indices_dims.iter().map(|&d| f16::from_f32(d as f32)).collect();

        let input_strides_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &input_strides_f16)?;
        let output_strides_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &output_strides_f16)?;
        let input_dims_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &input_dims_f16)?;
        let output_dims_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &output_dims_f16)?;
        let dim_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &[f16::from_f32(dim as f32)])?;
        let ndim_buf = MetalBuffer::<f16>::from_slice(device.metal_device(), &[f16::from_f32(ndim as f32)])?;

        // Get pipeline
        let library_ref = device.library();
        let library = library_ref.as_ref().ok_or_else(|| {
            TensorError::MetalError("Library not loaded".to_string())
        })?;
        let pipeline = library
            .get_function("gather_f16", None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel gather_f16: {:?}", e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to create pipeline: {:?}", e))
            })?;

        // Execute kernel
        let command_queue = device.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(indices_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(input_strides_buf.metal_buffer()), 0);
        encoder.set_buffer(4, Some(output_strides_buf.metal_buffer()), 0);
        encoder.set_buffer(5, Some(input_dims_buf.metal_buffer()), 0);
        encoder.set_buffer(6, Some(output_dims_buf.metal_buffer()), 0);
        encoder.set_buffer(7, Some(dim_buf.metal_buffer()), 0);
        encoder.set_buffer(8, Some(ndim_buf.metal_buffer()), 0);

        let grid_size = metal::MTLSize::new(output_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(output_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            indices.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU implementation of gather
    fn gather_cpu<I: FloatType>(&self, dim: usize, indices: &Tensor<I>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let self_data_t = self.to_vec();
        let self_data: Vec<f16> = unsafe { std::mem::transmute(self_data_t) };
        let indices_data_t = indices.to_vec();
        let indices_data: Vec<f16> = unsafe { std::mem::transmute(indices_data_t) };
        let self_dims = self.dims();
        let indices_dims = indices.dims();

        // Validate indices shape matches self shape except at dim
        if indices_dims.len() != self_dims.len() {
            return Err(TensorError::InvalidOperation(
                format!(
                    "Indices ndim {} must match input ndim {}",
                    indices_dims.len(),
                    self_dims.len()
                ),
            ));
        }

        for (i, (&s_dim, &i_dim)) in self_dims.iter().zip(indices_dims.iter()).enumerate() {
            if i != dim && s_dim != i_dim {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "Indices shape {:?} must match input shape {:?} except at dim {}",
                        indices_dims, self_dims, dim
                    ),
                ));
            }
        }

        let output_numel: usize = indices_dims.iter().product();
        let mut output = vec![f16::ZERO; output_numel];

        // Calculate strides for self
        let mut self_strides = vec![1usize; self_dims.len()];
        for i in (0..self_dims.len() - 1).rev() {
            self_strides[i] = self_strides[i + 1] * self_dims[i + 1];
        }

        // Calculate strides for indices/output
        let mut out_strides = vec![1usize; indices_dims.len()];
        for i in (0..indices_dims.len() - 1).rev() {
            out_strides[i] = out_strides[i + 1] * indices_dims[i + 1];
        }

        // Gather operation
        for out_idx in 0..output_numel {
            // Convert flat index to multi-dimensional index
            let mut coords = vec![0usize; indices_dims.len()];
            let mut remainder = out_idx;
            for i in 0..indices_dims.len() {
                coords[i] = remainder / out_strides[i];
                remainder %= out_strides[i];
            }

            // Get the index value at this coordinate
            let index_value = indices_data[out_idx].to_f32() as usize;

            // Validate index
            if index_value >= self_dims[dim] {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        index_value, dim, self_dims[dim]
                    ),
                ));
            }

            // Replace coordinate at dim with index value
            coords[dim] = index_value;

            // Calculate flat index in self
            let self_idx: usize = coords
                .iter()
                .zip(self_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();

            output[out_idx] = self_data[self_idx];
        }

        let output_t: Vec<T> = unsafe { std::mem::transmute(output) };
        Tensor::from_vec(output_t, indices_dims.to_vec())
    }

    /// Scatter values along an axis according to indices
    ///
    /// Scatters values from `src` into `self` along dimension `dim` using `indices`.
    /// Returns a new tensor (does not modify self in-place).
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to scatter
    /// * `indices` - Tensor of indices
    /// * `src` - Source values to scatter
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = MetalDevice::new()?;
    /// let x = Tensor::zeros(&device, vec![5])?;
    /// let indices = Tensor::from_vec(
    ///     vec![f16::from_f32(0.0), f16::from_f32(2.0), f16::from_f32(4.0)],
    ///     vec![3]
    /// )?;
    /// let src = Tensor::from_vec(
    ///     vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
    ///     vec![3]
    /// )?;
    /// let result = x.scatter(0, &indices, &src)?;
    /// // result = [1.0, 0.0, 2.0, 0.0, 3.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn scatter<I: FloatType>(&self, dim: usize, indices: &Tensor<I>, src: &Tensor<T>) -> TensorResult<Self> {
        if dim >= self.dims().len() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) if self.buffer().is_metal() && indices.buffer().is_metal() && src.buffer().is_metal() => {
                self.scatter_metal(dim, indices, src)
            }
            _ => self.scatter_cpu(dim, indices, src),
        }
    }

    /// Metal GPU implementation of scatter
    fn scatter_metal<I: FloatType>(&self, dim: usize, indices: &Tensor<I>, src: &Tensor<T>) -> TensorResult<Self> {
        let input_buf = self.buffer().as_metal()?;
        let indices_buf = indices.buffer().as_metal()?;
        let src_buf = src.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        let self_dims = self.dims();
        let src_dims = src.dims();
        let ndim = self_dims.len();

        // Calculate strides
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            input_strides[i] = input_strides[i + 1] * self_dims[i + 1];
        }

        let mut src_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            src_strides[i] = src_strides[i + 1] * src_dims[i + 1];
        }

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit(device.metal_device(), self.numel())?;

        // Create parameter buffers and select kernel based on type
        let kernel_name = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>() {
            "scatter_f16"
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "scatter_f32"
        } else {
            return Err(TensorError::InvalidOperation(
                format!("scatter() not implemented for type {:?}", std::any::type_name::<T>())
            ));
        };

        // Create parameter buffers with appropriate type
        let (input_strides_buf, src_strides_buf, input_dims_buf, src_dims_buf, dim_buf, ndim_buf) =
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>() {
                let input_strides_data: Vec<f16> = input_strides.iter().map(|&s| f16::from_f32(s as f32)).collect();
                let src_strides_data: Vec<f16> = src_strides.iter().map(|&s| f16::from_f32(s as f32)).collect();
                let input_dims_data: Vec<f16> = self_dims.iter().map(|&d| f16::from_f32(d as f32)).collect();
                let src_dims_data: Vec<f16> = src_dims.iter().map(|&d| f16::from_f32(d as f32)).collect();

                let buf1 = MetalBuffer::<f16>::from_slice(device.metal_device(), &input_strides_data)?;
                let buf2 = MetalBuffer::<f16>::from_slice(device.metal_device(), &src_strides_data)?;
                let buf3 = MetalBuffer::<f16>::from_slice(device.metal_device(), &input_dims_data)?;
                let buf4 = MetalBuffer::<f16>::from_slice(device.metal_device(), &src_dims_data)?;
                let buf5 = MetalBuffer::<f16>::from_slice(device.metal_device(), &[f16::from_f32(dim as f32)])?;
                let buf6 = MetalBuffer::<f16>::from_slice(device.metal_device(), &[f16::from_f32(ndim as f32)])?;

                // Transmute to generic type
                unsafe { (
                    std::mem::transmute(buf1),
                    std::mem::transmute(buf2),
                    std::mem::transmute(buf3),
                    std::mem::transmute(buf4),
                    std::mem::transmute(buf5),
                    std::mem::transmute(buf6),
                )}
            } else {
                // f32 case
                let input_strides_data: Vec<f32> = input_strides.iter().map(|&s| s as f32).collect();
                let src_strides_data: Vec<f32> = src_strides.iter().map(|&s| s as f32).collect();
                let input_dims_data: Vec<f32> = self_dims.iter().map(|&d| d as f32).collect();
                let src_dims_data: Vec<f32> = src_dims.iter().map(|&d| d as f32).collect();

                let buf1 = MetalBuffer::from_slice(device.metal_device(), &input_strides_data)?;
                let buf2 = MetalBuffer::from_slice(device.metal_device(), &src_strides_data)?;
                let buf3 = MetalBuffer::from_slice(device.metal_device(), &input_dims_data)?;
                let buf4 = MetalBuffer::from_slice(device.metal_device(), &src_dims_data)?;
                let buf5 = MetalBuffer::from_slice(device.metal_device(), &[dim as f32])?;
                let buf6 = MetalBuffer::from_slice(device.metal_device(), &[ndim as f32])?;

                (buf1, buf2, buf3, buf4, buf5, buf6)
            };

        // Get pipeline
        let library_ref = device.library();
        let library = library_ref.as_ref().ok_or_else(|| {
            TensorError::MetalError("Library not loaded".to_string())
        })?;
        let pipeline = library
            .get_function(kernel_name, None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to create pipeline: {:?}", e))
            })?;

        // Execute kernel
        let command_queue = device.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(indices_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(src_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(4, Some(input_strides_buf.metal_buffer()), 0);
        encoder.set_buffer(5, Some(src_strides_buf.metal_buffer()), 0);
        encoder.set_buffer(6, Some(input_dims_buf.metal_buffer()), 0);
        encoder.set_buffer(7, Some(src_dims_buf.metal_buffer()), 0);
        encoder.set_buffer(8, Some(dim_buf.metal_buffer()), 0);
        encoder.set_buffer(9, Some(ndim_buf.metal_buffer()), 0);

        let src_numel: usize = src_dims.iter().product();
        let grid_size = metal::MTLSize::new(src_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(src_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU implementation of scatter
    fn scatter_cpu<I: FloatType>(&self, dim: usize, indices: &Tensor<I>, src: &Tensor<T>) -> TensorResult<Self> {
        let self_data = self.to_vec();
        let indices_data = indices.to_vec();
        let src_data = src.to_vec();
        let self_dims = self.dims();
        let indices_dims = indices.dims();
        let src_dims = src.dims();

        // Validate shapes
        if indices_dims != src_dims {
            return Err(TensorError::ShapeMismatch {
                expected: indices_dims.to_vec(),
                actual: src_dims.to_vec(),
            });
        }

        if indices_dims.len() != self_dims.len() {
            return Err(TensorError::InvalidOperation(
                format!(
                    "Indices ndim {} must match input ndim {}",
                    indices_dims.len(),
                    self_dims.len()
                ),
            ));
        }

        // Check that all dimensions except dim match
        for (i, (&s_dim, &i_dim)) in self_dims.iter().zip(indices_dims.iter()).enumerate() {
            if i != dim && s_dim != i_dim {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "Indices shape {:?} must match input shape {:?} except at dim {}",
                        indices_dims, self_dims, dim
                    ),
                ));
            }
        }

        // Start with a copy of self
        let mut output = self_data.clone();

        // Calculate strides
        let mut self_strides = vec![1usize; self_dims.len()];
        for i in (0..self_dims.len() - 1).rev() {
            self_strides[i] = self_strides[i + 1] * self_dims[i + 1];
        }

        let mut src_strides = vec![1usize; src_dims.len()];
        for i in (0..src_dims.len() - 1).rev() {
            src_strides[i] = src_strides[i + 1] * src_dims[i + 1];
        }

        let src_numel: usize = src_dims.iter().product();

        // Scatter operation
        for src_idx in 0..src_numel {
            // Convert flat index to multi-dimensional index
            let mut coords = vec![0usize; src_dims.len()];
            let mut remainder = src_idx;
            for i in 0..src_dims.len() {
                coords[i] = remainder / src_strides[i];
                remainder %= src_strides[i];
            }

            // Get the index value at this coordinate
            let index_value = indices_data[src_idx].to_f32() as usize;

            // Validate index
            if index_value >= self_dims[dim] {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        index_value, dim, self_dims[dim]
                    ),
                ));
            }

            // Replace coordinate at dim with index value
            coords[dim] = index_value;

            // Calculate flat index in output
            let out_idx: usize = coords
                .iter()
                .zip(self_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();

            output[out_idx] = src_data[src_idx];
        }

        Tensor::from_vec(output, self_dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn test_gather_1d() {
        let x = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
            ],
            vec![5],
        )
        .unwrap();

        let indices = Tensor::from_vec(
            vec![
                f16::from_f32(0.0),
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(1.0),
            ],
            vec![4],
        )
        .unwrap();

        let result = x.gather(0, &indices).unwrap();
        let values = result.to_vec();

        assert_eq!(values.len(), 4);
        assert_eq!(values[0], f16::from_f32(1.0));
        assert_eq!(values[1], f16::from_f32(3.0));
        assert_eq!(values[2], f16::from_f32(5.0));
        assert_eq!(values[3], f16::from_f32(2.0));
    }

    #[test]
    fn test_gather_2d() {
        let x = Tensor::from_vec(
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

        // Gather along dimension 1 (columns)
        let indices = Tensor::from_vec(
            vec![
                f16::from_f32(0.0),
                f16::from_f32(2.0),
                f16::from_f32(2.0),
                f16::from_f32(1.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let result = x.gather(1, &indices).unwrap();
        let values = result.to_vec();

        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(values[0], f16::from_f32(1.0)); // x[0, 0]
        assert_eq!(values[1], f16::from_f32(3.0)); // x[0, 2]
        assert_eq!(values[2], f16::from_f32(6.0)); // x[1, 2]
        assert_eq!(values[3], f16::from_f32(5.0)); // x[1, 1]
    }

    #[test]
    fn test_scatter_1d() {
        let x = Tensor::from_vec(vec![f16::ZERO; 5], vec![5]).unwrap();

        let indices = Tensor::from_vec(
            vec![
                f16::from_f32(0.0),
                f16::from_f32(2.0),
                f16::from_f32(4.0),
            ],
            vec![3],
        )
        .unwrap();

        let src = Tensor::from_vec(
            vec![
                f16::from_f32(10.0),
                f16::from_f32(20.0),
                f16::from_f32(30.0),
            ],
            vec![3],
        )
        .unwrap();

        let result = x.scatter(0, &indices, &src).unwrap();
        let values = result.to_vec();

        assert_eq!(values.len(), 5);
        assert_eq!(values[0], f16::from_f32(10.0));
        assert_eq!(values[1], f16::ZERO);
        assert_eq!(values[2], f16::from_f32(20.0));
        assert_eq!(values[3], f16::ZERO);
        assert_eq!(values[4], f16::from_f32(30.0));
    }

    #[test]
    fn test_scatter_2d() {
        let x = Tensor::from_vec(vec![f16::ZERO; 12], vec![3, 4]).unwrap();

        // Scatter along dimension 1, all dimensions must match except dim 1
        let indices = Tensor::from_vec(
            vec![
                f16::from_f32(0.0),
                f16::from_f32(2.0),
                f16::from_f32(1.0),
                f16::from_f32(3.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            vec![3, 4],
        )
        .unwrap();

        let src = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
                f16::from_f32(9.0),
                f16::from_f32(10.0),
                f16::from_f32(11.0),
                f16::from_f32(12.0),
            ],
            vec![3, 4],
        )
        .unwrap();

        let result = x.scatter(1, &indices, &src).unwrap();
        let values = result.to_vec();

        assert_eq!(result.dims(), &[3, 4]);
        // First row should have values scattered to indices [0, 2, 1, 3]
        assert_eq!(values[0], f16::from_f32(1.0)); // src[0,0] -> out[0, indices[0,0]=0]
        assert_eq!(values[1], f16::from_f32(3.0)); // src[0,2] -> out[0, indices[0,2]=1]
        assert_eq!(values[2], f16::from_f32(2.0)); // src[0,1] -> out[0, indices[0,1]=2]
        assert_eq!(values[3], f16::from_f32(4.0)); // src[0,3] -> out[0, indices[0,3]=3]
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let x = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let indices = Tensor::from_vec(
            vec![f16::from_f32(0.0), f16::from_f32(5.0)], // 5 is out of bounds
            vec![2],
        )
        .unwrap();

        let result = x.gather(0, &indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_overwrite() {
        let x = Tensor::from_vec(vec![f16::ONE; 3], vec![3]).unwrap();

        let indices = Tensor::from_vec(
            vec![f16::from_f32(0.0), f16::from_f32(0.0)], // Same index twice
            vec![2],
        )
        .unwrap();

        let src = Tensor::from_vec(
            vec![f16::from_f32(10.0), f16::from_f32(20.0)],
            vec![2],
        )
        .unwrap();

        let result = x.scatter(0, &indices, &src).unwrap();
        let values = result.to_vec();

        // Last write wins
        assert_eq!(values[0], f16::from_f32(20.0));
        assert_eq!(values[1], f16::from_f32(1.0));
        assert_eq!(values[2], f16::from_f32(1.0));
    }

    #[test]
    fn test_gather_gpu() {
        let device = Device::default_metal().unwrap_or(Device::CPU);

        let (x, indices) = match &device {
            Device::Metal(dev) => (
                Tensor::from_vec_metal(
                    dev,
                    vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                        f16::from_f32(5.0),
                    ],
                    vec![5],
                )
                .unwrap(),
                Tensor::from_vec_metal(
                    dev,
                    vec![
                        f16::from_f32(0.0),
                        f16::from_f32(2.0),
                        f16::from_f32(4.0),
                    ],
                    vec![3],
                )
                .unwrap(),
            ),
            _ => return, // Skip GPU test if Metal not available
        };

        let result = x.gather(0, &indices).unwrap();
        let values = result.to_vec();

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], f16::from_f32(1.0));
        assert_eq!(values[1], f16::from_f32(3.0));
        assert_eq!(values[2], f16::from_f32(5.0));
    }

    #[test]
    fn test_scatter_gpu() {
        let device = Device::default_metal().unwrap_or(Device::CPU);

        let (x, indices, src) = match &device {
            Device::Metal(dev) => (
                Tensor::from_vec_metal(dev, vec![f16::ZERO; 5], vec![5]).unwrap(),
                Tensor::from_vec_metal(
                    dev,
                    vec![
                        f16::from_f32(0.0),
                        f16::from_f32(2.0),
                        f16::from_f32(4.0),
                    ],
                    vec![3],
                )
                .unwrap(),
                Tensor::from_vec_metal(
                    dev,
                    vec![
                        f16::from_f32(10.0),
                        f16::from_f32(20.0),
                        f16::from_f32(30.0),
                    ],
                    vec![3],
                )
                .unwrap(),
            ),
            _ => return, // Skip GPU test if Metal not available
        };

        let result = x.scatter(0, &indices, &src).unwrap();
        let values = result.to_vec();

        assert_eq!(values.len(), 5);
        assert_eq!(values[0], f16::from_f32(10.0));
        assert_eq!(values[1], f16::ZERO);
        assert_eq!(values[2], f16::from_f32(20.0));
        assert_eq!(values[3], f16::ZERO);
        assert_eq!(values[4], f16::from_f32(30.0));
    }
}
