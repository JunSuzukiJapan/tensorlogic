//! Tensor transformation methods (reshape, flatten, etc.)

use crate::error::TensorResult;
use crate::tensor::{FloatType, Tensor};
use std::marker::PhantomData;

/// Trait for transforming tensor shapes
pub trait TensorTransform: Sized {
    /// Reshape tensor (must preserve number of elements)
    fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self>;

    /// Get tensor as a 1D view
    fn flatten(&self) -> TensorResult<Self>;
}

impl<T: FloatType> TensorTransform for Tensor<T> {
    fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self> {
        let new_tensor_shape = self.shape.reshape(new_shape)?;

        Ok(Self {
            shape: new_tensor_shape.clone(),
            strides: new_tensor_shape.compute_strides(),
            buffer: self.buffer.clone(),
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
}
