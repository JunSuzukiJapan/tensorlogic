//! Tensor transformation methods (reshape, flatten, etc.)

use crate::error::TensorResult;
use crate::tensor::{BufferHandle, FloatType, Tensor};
use crate::tensor::TensorAccessors;
use crate::tensor::TensorCreation;
use crate::tensor::TensorIO;
use crate::device::EncoderProvider;
use std::marker::PhantomData;
use std::io::Write;

/// Trait for transforming tensor shapes
pub trait TensorTransform: Sized {
    /// Reshape tensor (must preserve number of elements)
    fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self>;

    /// Get tensor as a 1D view
    fn flatten(&self) -> TensorResult<Self>;

    /// Check if tensor has contiguous memory layout (row-major)
    fn is_contiguous(&self) -> bool;

    /// Return a contiguous copy of the tensor if needed
    fn contiguous(&self) -> TensorResult<Self>;
}

impl<T: FloatType> TensorTransform for Tensor<T> {
    fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self> {
        // eprintln!("[RESHAPE] === Entry === old_shape={:?} -> new_shape={:?}", self.dims(), new_shape);

        // eprintln!("[RESHAPE] Step 1: Calling shape.reshape()...");
        let new_tensor_shape = self.shape.reshape(new_shape)?;
        // eprintln!("[RESHAPE] Step 1: DONE");

        // eprintln!("[RESHAPE] Step 2: Cloning buffer...");
        let buffer_clone = self.buffer.clone();
        // eprintln!("[RESHAPE] Step 2: DONE");

        // eprintln!("[RESHAPE] Step 3: Computing strides...");
        let new_strides = new_tensor_shape.compute_strides();
        // eprintln!("[RESHAPE] Step 3: DONE, strides={:?}", new_strides);

        // eprintln!("[RESHAPE] Step 4: Cloning device...");
        let device_clone = self.device.clone();
        // eprintln!("[RESHAPE] Step 4: DONE");

        // eprintln!("[RESHAPE] Step 5: Cloning buffer_pool...");
        let pool_clone = self.buffer_pool.clone();
        // eprintln!("[RESHAPE] Step 5: DONE");

        // eprintln!("[RESHAPE] Step 6: Creating new tensor struct...");
        let result = Self {
            shape: new_tensor_shape.clone(),
            strides: new_strides,
            buffer: buffer_clone,
            device: device_clone,
            grad: None,
            requires_grad: self.requires_grad,
            grad_node: None,
            version: 0,
            buffer_pool: pool_clone,
            _phantom: PhantomData,
        };
        // eprintln!("[RESHAPE] Step 6: DONE");

        // eprintln!("[RESHAPE] === Exit ===");
        Ok(result)
    }

    fn flatten(&self) -> TensorResult<Self> {
        self.reshape(vec![self.numel()])
    }

    fn is_contiguous(&self) -> bool {
        // Check if strides match the expected contiguous (row-major) layout
        let expected_strides = self.shape.compute_strides();
        self.strides == expected_strides
    }

    fn contiguous(&self) -> TensorResult<Self> {
        // If already contiguous, return a clone
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS] Making tensor contiguous: shape={:?}, strides={:?}",
                     self.dims(), self.strides);
        }

        // Use GPU implementation for Metal tensors
        use crate::device::Device;
        match self.device() {
            Device::Metal(metal_device) => {
                if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
                    eprintln!("[CONTIGUOUS] Using GPU implementation");
                }
                self.contiguous_metal(metal_device)
            }
            _ => {
                // CPU fallback for non-Metal devices
                if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
                    eprintln!("[CONTIGUOUS] Using CPU fallback");
                }
                self.contiguous_cpu()
            }
        }
    }
}

impl<T: FloatType> Tensor<T> {
    /// GPU implementation of contiguous() for Metal tensors
    fn contiguous_metal(&self, metal_device: &crate::device::MetalDevice) -> TensorResult<Self> {
        use crate::device::MetalBuffer;

        let dims = self.dims().to_vec();
        let numel = self.numel();
        let ndim = dims.len();

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS_METAL] Starting GPU contiguous: numel={}, ndim={}", numel, ndim);
        }

        let mut device = metal_device.clone();

        // Load shader library if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Get input buffer
        let input_buf = self.buffer().as_metal()?;

        // Create output buffer (pooled allocation)
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] contiguous_metal: About to call new_uninit_pooled (numel={})...", numel);
            std::io::stderr().flush().ok();
        }
        let output_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), numel)?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] contiguous_metal: new_uninit_pooled returned successfully");
            std::io::stderr().flush().ok();
        }

        // Create shape buffer
        let shape_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let shape_buf = device.metal_device().new_buffer_with_data(
            shape_u32.as_ptr() as *const _,
            (shape_u32.len() * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create strides buffer
        let strides_u32: Vec<u32> = self.strides.iter().map(|&s| s as u32).collect();
        let strides_buf = device.metal_device().new_buffer_with_data(
            strides_u32.as_ptr() as *const _,
            (strides_u32.len() * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create ndim buffer
        let ndim_u32 = ndim as u32;
        let ndim_buf = device.metal_device().new_buffer_with_data(
            &ndim_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create numel buffer
        let numel_u32 = numel as u32;
        let numel_buf = device.metal_device().new_buffer_with_data(
            &numel_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get kernel
        let suffix = T::kernel_suffix();
        let kernel_name = format!("make_contiguous{}", suffix);

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS_METAL] Compiling kernel: {}", kernel_name);
        }

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline(&kernel_name)?;

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS_METAL] Kernel compiled, executing...");
        }

        // Execute kernel
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(output_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(&shape_buf), 0);
        encoder.set_buffer(3, Some(&strides_buf), 0);
        encoder.set_buffer(4, Some(&ndim_buf), 0);
        encoder.set_buffer(5, Some(&numel_buf), 0);

        // Dispatch threads
        let max_threads = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid_size = metal::MTLSize::new(numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(max_threads, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS_METAL] Kernel encoded, creating result tensor...");
        }

        // No wait here - let Commands manager handle batching

        // Create result tensor with contiguous strides
        let result = self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(output_buf) }),
            self.shape().clone(),
        )?;

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS_METAL] GPU contiguous complete");
        }

        Ok(result)
    }

    /// CPU fallback implementation of contiguous()
    fn contiguous_cpu(&self) -> TensorResult<Self> {
        let dims = self.dims().to_vec();
        let numel = self.numel();

        // Get data as CPU vector (handles GPU->CPU transfer if needed)
        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS] Transferring {} elements to CPU...", numel);
        }
        let src_data = self.to_vec();
        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS] Transfer complete, reordering data...");
        }

        // Create contiguous data vector with proper ordering
        let mut dst_data = vec![T::zero(); numel];

        for linear_idx in 0..numel {
            // Convert linear index to multi-dimensional indices
            let mut indices = vec![0; dims.len()];
            let mut remaining = linear_idx;
            for i in (0..dims.len()).rev() {
                let dim_size = dims[i];
                indices[i] = remaining % dim_size;
                remaining /= dim_size;
            }

            // Calculate strided offset in source
            let mut src_offset = 0;
            for (idx, &stride) in indices.iter().zip(self.strides.iter()) {
                src_offset += idx * stride;
            }

            // Copy element
            dst_data[linear_idx] = src_data[src_offset];
        }

        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS] Reordering complete, creating new tensor...");
        }

        // Create new tensor on CPU
        Self::from_vec(dst_data, dims)
    }
}
