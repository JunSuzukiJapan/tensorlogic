//! Element-wise tensor operations with Metal GPU acceleration

use crate::autograd::gradients::{AddBackward, DivBackward, MulBackward, SubBackward};
use crate::autograd::{AutogradContext, GradientFunction, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Helper function to record a binary operation in the computation graph
fn record_binary_op(
    op: Operation,
    grad_fn: Box<dyn GradientFunction>,
    self_tensor: &Tensor,
    other_tensor: &Tensor,
    result: &mut Tensor,
) {
    if !(self_tensor.requires_grad() || other_tensor.requires_grad()) || !AutogradContext::is_enabled() {
        return;
    }

    // Get or allocate node IDs for input tensors
    let self_node_id = self_tensor.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());
    let other_node_id = other_tensor.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());

    // Add node to computation graph (this allocates the result node ID internally)
    let result_node_id = AutogradContext::add_node(
        op,
        vec![self_node_id, other_node_id],
        Some(grad_fn),
    );

    // Register tensors
    AutogradContext::register_tensor(self_node_id, self_tensor.clone());
    AutogradContext::register_tensor(other_node_id, other_tensor.clone());

    // Set node ID and requires_grad on result
    result.set_grad_node(result_node_id);
    result.set_requires_grad(true);
}

impl Tensor {
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> TensorResult<Self> {
        // Check shape compatibility
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        // Perform computation
        let mut result = if self.buffer().is_metal() && other.buffer().is_metal() {
            self.add_metal(other)?
        } else {
            self.add_cpu(other)?
        };

        // Record in computation graph
        let grad_fn = Box::new(AddBackward::new(self.shape().clone(), other.shape().clone()));
        record_binary_op(Operation::Add, grad_fn, self, other, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of addition
    fn add_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        // Get device
        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load shaders if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        // Create local executor for this operation
        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_binary_op("add_f16", a_buf, b_buf, &result_buf)?;

        // Create result tensor
        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU fallback for addition
    fn add_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        // Keep result on same device as self
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result, self.dims().to_vec()),
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        let mut result = if self.buffer().is_metal() && other.buffer().is_metal() {
            self.sub_metal(other)?
        } else {
            self.sub_cpu(other)?
        };

        let grad_fn = Box::new(SubBackward::new(self.shape().clone(), other.shape().clone()));
        record_binary_op(Operation::Sub, grad_fn, self, other, &mut result);

        Ok(result)
    }

    fn sub_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_binary_op("sub_f16", a_buf, b_buf, &result_buf)?;

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    fn sub_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result, self.dims().to_vec()),
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        let mut result = if self.buffer().is_metal() && other.buffer().is_metal() {
            self.mul_metal(other)?
        } else {
            self.mul_cpu(other)?
        };

        let grad_fn = Box::new(MulBackward::new(self.clone(), other.clone()));
        record_binary_op(Operation::Mul, grad_fn, self, other, &mut result);

        Ok(result)
    }

    fn mul_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_binary_op("mul_f16", a_buf, b_buf, &result_buf)?;

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    fn mul_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result, self.dims().to_vec()),
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        let mut result = if self.buffer().is_metal() && other.buffer().is_metal() {
            self.div_metal(other)?
        } else {
            self.div_cpu(other)?
        };

        let grad_fn = Box::new(DivBackward::new(self.clone(), other.clone()));
        record_binary_op(Operation::Div, grad_fn, self, other, &mut result);

        Ok(result)
    }

    fn div_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_binary_op("div_f16", a_buf, b_buf, &result_buf)?;

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    fn div_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect();

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result, self.dims().to_vec()),
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_add_gpu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![3],
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        let expected = vec![f16::from_f32(5.0), f16::from_f32(7.0), f16::from_f32(9.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_sub_gpu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(5.0), f16::from_f32(7.0), f16::from_f32(9.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        let expected = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_mul_gpu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)],
            vec![3],
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        let expected = vec![f16::from_f32(10.0), f16::from_f32(18.0), f16::from_f32(28.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_div_gpu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(5.0)],
            vec![3],
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        let expected = vec![f16::from_f32(5.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_shape_mismatch() {
        let device = get_test_device();

        let a = Tensor::zeros(&device, vec![2, 3]).unwrap();
        let b = Tensor::zeros(&device, vec![3, 2]).unwrap();

        assert!(a.add(&b).is_err());
    }
}
