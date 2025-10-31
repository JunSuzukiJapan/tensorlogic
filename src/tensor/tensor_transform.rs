//! Tensor transformation methods (reshape, flatten, etc.)

use crate::error::TensorResult;
use crate::tensor::{FloatType, Tensor};
use crate::tensor::TensorAccessors;
use crate::tensor::TensorCreation;
use crate::tensor::TensorIO;
use std::marker::PhantomData;

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
        if std::env::var("TL_DEBUG_RESHAPE").is_ok() {
            eprintln!("[RESHAPE] old_shape={:?} -> new_shape={:?}", self.dims(), new_shape);
        }

        let new_tensor_shape = self.shape.reshape(new_shape)?;

        if std::env::var("TL_DEBUG_RESHAPE").is_ok() {
            eprintln!("[RESHAPE] shape.reshape() OK, cloning buffer...");
        }

        let buffer_clone = self.buffer.clone();

        if std::env::var("TL_DEBUG_RESHAPE").is_ok() {
            eprintln!("[RESHAPE] buffer cloned, creating new tensor...");
        }

        Ok(Self {
            shape: new_tensor_shape.clone(),
            strides: new_tensor_shape.compute_strides(),
            buffer: buffer_clone,
            device: self.device.clone(),
            grad: None,
            requires_grad: self.requires_grad,
            grad_node: None,
            version: 0,
            buffer_pool: self.buffer_pool.clone(),
            _phantom: PhantomData,
        })
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

        // For non-contiguous tensors, copy data via CPU
        // This ensures proper element ordering regardless of strides
        let dims = self.dims().to_vec();
        let numel = self.numel();

        // Get data as CPU vector (handles GPU->CPU transfer if needed)
        if std::env::var("TL_DEBUG_CONTIGUOUS").is_ok() {
            eprintln!("[CONTIGUOUS] Transferring {} elements from GPU to CPU...", numel);
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
            eprintln!("[CONTIGUOUS] Reordering complete, creating new tensor on device...");
        }

        // Create new tensor from contiguous data on the same device
        use crate::device::Device;
        match self.device() {
            Device::Metal(metal_device) => {
                // Use pooled allocation for GPU tensor
                Self::from_vec_gpu_pooled(metal_device, dst_data, dims)
            }
            Device::CPU => {
                // CPU tensor
                Self::from_vec(dst_data, dims)
            }
            Device::NeuralEngine => {
                // For now, fallback to CPU
                Self::from_vec(dst_data, dims)
            }
        }
    }
}
