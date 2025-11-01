//! KV Cache operations optimized for transformer inference
//!
//! This module provides efficient operations for managing Key-Value caches
//! in transformer models, avoiding quadratic memory allocation overhead.

use crate::device::{Device, MetalBuffer, MetalDevice, EncoderProvider};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{FloatType, Tensor, TensorAccessors, TensorIO};

impl<T: FloatType> Tensor<T> {
    /// Create a pre-allocated cache tensor with maximum capacity
    ///
    /// Creates a buffer with capacity for `max_length` elements along dimension 0,
    /// but with initial shape [0, ...other_dims]. Data can be appended efficiently
    /// using `append_cache`.
    ///
    /// # Arguments
    /// * `max_length` - Maximum sequence length capacity
    /// * `shape_rest` - Remaining dimensions [num_heads, head_dim]
    /// * `device` - Target device (Metal or CPU)
    ///
    /// # Example
    /// ```ignore
    /// // Create cache for 512 tokens, 4 KV heads, 128 head dimension
    /// let kv_cache = Tensor::<f32>::create_cache_tensor(512, &[4, 128], &device)?;
    /// // Initial shape: [0, 4, 128]
    /// ```
    pub fn create_cache_tensor(
        max_length: usize,
        shape_rest: &[usize],
        device: &Device,
    ) -> TensorResult<Self> {
        // Calculate total capacity
        let elements_per_item: usize = shape_rest.iter().product();
        let total_capacity = max_length * elements_per_item;

        // Allocate buffer
        let buffer = match device {
            Device::Metal(metal_dev) => {
                let metal_buf = MetalBuffer::<T>::new_uninit(metal_dev.metal_device(), total_capacity)?;
                crate::tensor::BufferHandle::Metal(metal_buf)
            }
            Device::CPU => {
                let cpu_data = vec![T::zero(); total_capacity];
                crate::tensor::BufferHandle::CPU(cpu_data)
            }
            Device::NeuralEngine => {
                return Err(TensorError::InvalidOperation(
                    "NeuralEngine not supported for cache operations".to_string()
                ));
            }
        };

        // Initial shape: [0, ...shape_rest]
        let mut initial_shape = vec![0];
        initial_shape.extend_from_slice(shape_rest);

        // Create tensor with metadata for tracking capacity
        let tensor = Tensor {
            shape: crate::tensor::TensorShape::new(initial_shape.clone()),
            strides: Self::compute_strides(&initial_shape),
            buffer,
            device: device.clone(),
            grad: None,
            requires_grad: false,
            grad_node: None,
            version: 0,
            buffer_pool: None,
            _phantom: std::marker::PhantomData,
        };

        // Store max capacity in a way we can retrieve later
        // We'll use the actual buffer size vs shape[0] to track this

        Ok(tensor)
    }

    /// Append data to a pre-allocated cache tensor (zero-copy, in-place)
    ///
    /// Writes `new_data` to the cache at the current position without copying
    /// existing data. Returns a new view with updated shape.
    ///
    /// # Arguments
    /// * `cache` - Pre-allocated cache tensor from `create_cache_tensor`
    /// * `new_data` - Data to append, shape [new_len, ...same_other_dims]
    ///
    /// # Returns
    /// New tensor view with shape [current_len + new_len, ...], sharing the same buffer
    ///
    /// # Example
    /// ```ignore
    /// let mut kv_cache = Tensor::<f32>::create_cache_tensor(512, &[4, 128], &device)?;
    /// let new_kv = Tensor::<f32>::zeros(&device, vec![1, 4, 128])?;
    /// kv_cache = Tensor::append_cache(&kv_cache, &new_kv)?;
    /// // Now kv_cache.shape() == [1, 4, 128]
    /// ```
    pub fn append_cache(cache: &Tensor<T>, new_data: &Tensor<T>) -> TensorResult<Tensor<T>> {
        // Validate dimensions match (except dim 0)
        let cache_dims = cache.dims();
        let new_dims = new_data.dims();

        if cache_dims.len() != new_dims.len() {
            return Err(TensorError::ShapeMismatch {
                expected: cache_dims.to_vec(),
                actual: new_dims.to_vec(),
            });
        }

        for i in 1..cache_dims.len() {
            if cache_dims[i] != new_dims[i] {
                return Err(TensorError::ShapeMismatch {
                    expected: cache_dims.to_vec(),
                    actual: new_dims.to_vec(),
                });
            }
        }

        let current_length = cache_dims[0];
        let new_length = new_dims[0];
        let elements_per_item: usize = new_dims[1..].iter().product();

        // Calculate buffer capacity
        let buffer_size = match &cache.buffer {
            crate::tensor::BufferHandle::Metal(buf) => buf.len(),
            crate::tensor::BufferHandle::CPU(data) => data.len(),
            crate::tensor::BufferHandle::NeuralEngine(_) => {
                return Err(TensorError::InvalidOperation(
                    "NeuralEngine not supported for cache operations".to_string()
                ));
            }
        };
        let max_capacity = buffer_size / elements_per_item;

        // Check capacity
        if current_length + new_length > max_capacity {
            return Err(TensorError::InvalidOperation(
                format!(
                    "Cache overflow: current {} + new {} > capacity {}",
                    current_length, new_length, max_capacity
                ),
            ));
        }

        // Perform in-place write
        if cache.buffer.is_metal() && new_data.buffer.is_metal() {
            Self::append_cache_metal(cache, new_data, current_length, elements_per_item)?;
        } else {
            Self::append_cache_cpu(cache, new_data, current_length, elements_per_item)?;
        }

        // Create new view with updated shape
        let mut new_shape = cache_dims.to_vec();
        new_shape[0] = current_length + new_length;

        Ok(Tensor {
            shape: crate::tensor::TensorShape::new(new_shape.clone()),
            strides: Self::compute_strides(&new_shape),
            buffer: cache.buffer.clone(),
            device: cache.device.clone(),
            grad: None,
            requires_grad: false,
            grad_node: None,
            version: cache.version + 1,
            buffer_pool: cache.buffer_pool.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    fn append_cache_metal(
        cache: &Tensor<T>,
        new_data: &Tensor<T>,
        offset_items: usize,
        elements_per_item: usize,
    ) -> TensorResult<()> {
        let cache_buf = cache.buffer.as_metal()?;
        let new_buf = new_data.buffer.as_metal()?;

        let mut device = match &cache.device {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load shader library
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Calculate offset in elements
        let offset_elements = offset_items * elements_per_item;
        let num_elements = new_data.numel();

        // Use simple copy kernel
        let kernel_name = format!("cache_append{}", T::kernel_suffix());
        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline(&kernel_name)?;

        let offset_buf = device.metal_device().new_buffer_with_data(
            &(offset_elements as u32) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&new_buf.buffer), 0);
        encoder.set_buffer(1, Some(&cache_buf.buffer), 0);
        encoder.set_buffer(2, Some(&offset_buf), 0);

        let grid_size = metal::MTLSize::new(num_elements as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(num_elements as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Ok(())
    }

    fn append_cache_cpu(
        _cache: &Tensor<T>,
        _new_data: &Tensor<T>,
        _offset_items: usize,
        _elements_per_item: usize,
    ) -> TensorResult<()> {
        // CPU cache append not yet implemented - Metal is the primary target
        Err(TensorError::InvalidOperation(
            "CPU cache append not yet implemented - use Metal device".to_string()
        ))
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
