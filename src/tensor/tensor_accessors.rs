//! Tensor accessor methods for shape, device, and buffer information

use crate::autograd::NodeId;
use crate::device::Device;
use crate::tensor::{BufferHandle, FloatType, Tensor, TensorShape};

/// Trait for accessing tensor properties
pub trait TensorAccessors<T: FloatType> {
    /// Get the tensor shape
    fn shape(&self) -> &TensorShape;

    /// Get the tensor dimensions
    fn dims(&self) -> &[usize];

    /// Get the rank (number of dimensions)
    fn rank(&self) -> usize;

    /// Get the total number of elements
    fn numel(&self) -> usize;

    /// Get the strides
    fn strides(&self) -> &[usize];

    /// Get the device
    fn device(&self) -> &Device;

    /// Get reference to buffer
    fn buffer(&self) -> &BufferHandle<T>;

    /// Check if tensor requires gradient
    fn requires_grad(&self) -> bool;

    /// Get computation graph node ID
    fn grad_node(&self) -> Option<NodeId>;

    /// Get version (for gradient accumulation tracking)
    fn version(&self) -> u64;
}

impl<T: FloatType> TensorAccessors<T> for Tensor<T> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    fn rank(&self) -> usize {
        self.shape.rank()
    }

    fn numel(&self) -> usize {
        self.shape.numel()
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn buffer(&self) -> &BufferHandle<T> {
        &self.buffer
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn grad_node(&self) -> Option<NodeId> {
        self.grad_node
    }

    fn version(&self) -> u64 {
        self.version
    }
}
