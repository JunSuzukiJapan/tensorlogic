//! Slice operations for tensor manipulation

use crate::device::{Device, MetalBuffer, MetalDevice, EncoderProvider};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorShape};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;
use metal::MTLSize;

impl<T: FloatType> Tensor<T> {
    /// Extract the last slice along axis 0 using GPU (Metal) implementation
    ///
    /// # Arguments
    /// * `axis` - Currently only axis=0 is supported
    ///
    /// # Returns
    /// Tensor with shape [dims[1], dims[2], ...] (one fewer dimension)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::ones(&device, vec![35, 22, 2048])?;
    /// let last = x.slice_last(0)?;  // [35, 22, 2048] -> [22, 2048]
    /// ```
    pub fn slice_last(&self, axis: usize) -> TensorResult<Self> {
        let dims = self.dims();

        // Validate axis
        if axis >= dims.len() {
            return Err(TensorError::InvalidOperation(
                format!("slice_last() axis {} out of bounds for {}D tensor", axis, dims.len())
            ));
        }

        // Currently only axis=0 is supported for GPU implementation
        if axis != 0 {
            return Err(TensorError::InvalidOperation(
                format!("slice_last() currently only supports axis=0, got axis={}", axis)
            ));
        }

        // For 1D tensors, use CPU fallback (small data, sync is acceptable)
        if dims.len() == 1 {
            let data = self.sync_and_read();
            let last_val = data[dims[0] - 1];
            return match self.device() {
                Device::Metal(dev) => Tensor::from_vec_gpu(dev, vec![last_val], vec![]),
                _ => Tensor::from_vec(vec![last_val], vec![]),
            };
        }

        // GPU implementation for 2D+ tensors on Metal device (avoids sync)
        match self.device() {
            Device::Metal(dev) => {
                self.slice_last_axis0_metal(dev)
            }
            _ => {
                // CPU fallback for non-Metal devices
                self.slice_last_cpu_fallback(axis)
            }
        }
    }

    /// GPU implementation of slice_last for axis=0 using Metal
    fn slice_last_axis0_metal(&self, device: &MetalDevice) -> TensorResult<Self> {
        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last_axis0_metal: START");
        }

        let dims = self.dims();

        // For axis=0, we remove the first dimension
        let i = dims[0] as u32;

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: dims={:?}, i={}", dims, i);
        }

        // Calculate H and D for 3D case, or adjust for other dimensions
        let output_shape: Vec<usize> = dims[1..].to_vec();
        let output_numel: usize = output_shape.iter().product();

        // For simplicity, treat as [I, H*D] where H*D is the product of remaining dims
        let h = if dims.len() > 2 { dims[1] as u32 } else { 1 };
        let d = if output_numel > 0 { (output_numel / h as usize) as u32 } else { 1 };

        // Get input buffer
        let input_buf = self.buffer().as_metal()?;

        // Load Metal shader library
        let mut device_mut = device.clone();
        if device_mut.library().is_none() {
            let source = include_str!("../../shaders/unified.metal");
            device_mut.load_library(source)?;
        }

        let library = device_mut.library().ok_or_else(|| {
            TensorError::InvalidOperation("Failed to load Metal library".to_string())
        })?;

        // Select kernel based on type
        let kernel_name = if std::mem::size_of::<T>() == 4 {
            "slice_last_axis0_f32"
        } else {
            "slice_last_axis0_f16"
        };

        let kernel = library.get_function(kernel_name, None)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to get kernel {}: {}", kernel_name, e)))?;

        let pipeline = device_mut
            .metal_device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to create pipeline: {}", e)))?;

        // Create output buffer
        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: creating output buffer, size={}", output_numel);
        }
        let output_buf = MetalBuffer::zeros(&device_mut, output_numel)?;

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: getting command buffer");
        }
        // Get command buffer from batch (candle-style)
        let (_flushed, command_buffer) = device_mut.command_buffer()?;
        let encoder = command_buffer.encoder();

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: setting up encoder");
        }
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(output_buf.metal_buffer()), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &i as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &h as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &d as *const u32 as *const _);

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: dispatching kernel, grid={}x{}, threadgroup=8x8", h, d);
        }
        // Dispatch threads: one thread per output element
        let grid_size = MTLSize::new(h as u64, d as u64, 1);
        let threadgroup_size = MTLSize::new(
            8.min(h as u64).max(1),
            8.min(d as u64).max(1),
            1
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: ending encoding");
        }
        encoder.end_encoding();
        // NO commit here - Commands manager will batch and commit automatically

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last: creating output tensor");
        }
        // Create output tensor
        let output_tensor_shape = TensorShape::new(output_shape);
        let result = self.new_from_pool(
            BufferHandle::Metal(output_buf),
            output_tensor_shape,
        )?;

        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] slice_last_axis0_metal: DONE");
        }
        Ok(result)
    }

    /// CPU fallback implementation for slice_last
    fn slice_last_cpu_fallback(&self, axis: usize) -> TensorResult<Self> {
        let dims = self.dims();

        // Get last index along specified axis
        let last_idx = dims[axis] - 1;

        // Get tensor data with proper GPU sync
        let data = self.sync_and_read();

        // Calculate slice parameters
        let (offset, length, new_shape) = if axis == 0 {
            // Extract last row/slice along axis 0
            let row_size: usize = dims[1..].iter().product();
            (last_idx * row_size, row_size, dims[1..].to_vec())
        } else {
            return Err(TensorError::InvalidOperation(
                format!("slice_last() CPU fallback currently only supports axis=0, got axis={}", axis)
            ));
        };

        // Extract slice
        let slice_data: Vec<T> = data[offset..offset + length].to_vec();

        // Create new tensor on same device
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, slice_data, new_shape),
            _ => Tensor::from_vec(slice_data, new_shape),
        }
    }
}
