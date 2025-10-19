//! Core Tensor type

use crate::device::{Device, MetalBuffer, MetalDevice};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, TensorShape};
use half::f16;

/// Tensor data structure (f16 only)
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor shape
    shape: TensorShape,

    /// Strides for memory layout (row-major)
    strides: Vec<usize>,

    /// Data buffer (on GPU or CPU)
    buffer: BufferHandle,

    /// Device location
    device: Device,

    /// Gradient (for auto-differentiation)
    grad: Option<Box<Tensor>>,

    /// Whether gradient computation is required
    requires_grad: bool,
}

impl Tensor {
    /// Create a new tensor from buffer and shape
    pub fn new(buffer: BufferHandle, shape: TensorShape, device: Device) -> TensorResult<Self> {
        let expected_len = shape.numel();
        let actual_len = buffer.len();

        if expected_len != actual_len {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected_len],
                actual: vec![actual_len],
            });
        }

        let strides = shape.compute_strides();

        Ok(Self {
            shape,
            strides,
            buffer,
            device,
            grad: None,
            requires_grad: false,
        })
    }

    /// Create a tensor from f16 vector on CPU
    pub fn from_vec(data: Vec<f16>, shape: Vec<usize>) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);

        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }

        let buffer = BufferHandle::CPU(data);

        Self::new(buffer, shape, Device::CPU)
    }

    /// Create a tensor from f16 vector on Metal device
    pub fn from_vec_metal(
        device: &MetalDevice,
        data: Vec<f16>,
        shape: Vec<usize>,
    ) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);

        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }

        let metal_buffer = MetalBuffer::from_f16_slice(device.metal_device(), &data)?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, shape, Device::Metal(device.clone()))
    }

    /// Create a tensor filled with zeros on Metal device
    pub fn zeros(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);
        let metal_buffer = MetalBuffer::zeros(device.metal_device(), shape.numel())?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, shape, Device::Metal(device.clone()))
    }

    /// Create a tensor filled with ones on Metal device
    pub fn ones(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);
        let metal_buffer = MetalBuffer::ones(device.metal_device(), shape.numel())?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, shape, Device::Metal(device.clone()))
    }

    /// Create a scalar tensor
    pub fn scalar(device: &MetalDevice, value: f16) -> TensorResult<Self> {
        Self::from_vec_metal(device, vec![value], vec![1])
    }

    // === Accessors ===

    /// Get the tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get the tensor dimensions
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get reference to buffer
    pub fn buffer(&self) -> &BufferHandle {
        &self.buffer
    }

    /// Check if tensor requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether gradient is required
    pub fn set_requires_grad(&mut self, requires: bool) {
        self.requires_grad = requires;
    }

    /// Get the gradient
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Zero out the gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    // === Device transfers ===

    /// Transfer tensor to CPU
    pub fn to_cpu(&self) -> TensorResult<Self> {
        if self.buffer.is_cpu() {
            return Ok(self.clone());
        }

        let data = self.buffer.to_cpu_vec();
        let buffer = BufferHandle::CPU(data);

        Self::new(buffer, self.shape.clone(), Device::CPU)
    }

    /// Transfer tensor to Metal device
    pub fn to_metal(&self, device: &MetalDevice) -> TensorResult<Self> {
        if let BufferHandle::Metal(_) = &self.buffer {
            if let Device::Metal(dev) = &self.device {
                if dev == device {
                    return Ok(self.clone());
                }
            }
        }

        let data = self.buffer.to_cpu_vec();
        let metal_buffer = MetalBuffer::from_f16_slice(device.metal_device(), &data)?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, self.shape.clone(), Device::Metal(device.clone()))
    }

    /// Get data as Vec<f16> (copies from GPU if needed)
    pub fn to_vec(&self) -> Vec<f16> {
        self.buffer.to_cpu_vec()
    }

    /// Get data as Vec<f32> (copies from GPU if needed)
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.buffer.to_cpu_vec().iter().map(|x| x.to_f32()).collect()
    }

    // === Shape operations ===

    /// Reshape tensor (must preserve number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self> {
        let new_tensor_shape = self.shape.reshape(new_shape)?;

        Ok(Self {
            shape: new_tensor_shape.clone(),
            strides: new_tensor_shape.compute_strides(),
            buffer: self.buffer.clone(),
            device: self.device.clone(),
            grad: None,
            requires_grad: self.requires_grad,
        })
    }

    /// Get tensor as a 1D view
    pub fn flatten(&self) -> TensorResult<Self> {
        self.reshape(vec![self.numel()])
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        self_data == other_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_create_from_vec() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let tensor = Tensor::from_vec_metal(&device, data.clone(), vec![3]).unwrap();

        assert_eq!(tensor.dims(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_zeros() {
        let device = get_test_device();
        let tensor = Tensor::zeros(&device, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec();
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_ones() {
        let device = get_test_device();
        let tensor = Tensor::ones(&device, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec();
        assert!(data.iter().all(|&x| x == f16::ONE));
    }

    #[test]
    fn test_reshape() {
        let device = get_test_device();
        let tensor = Tensor::zeros(&device, vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(vec![6, 1]).unwrap();

        assert_eq!(reshaped.dims(), &[6, 1]);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_flatten() {
        let device = get_test_device();
        let tensor = Tensor::zeros(&device, vec![2, 3, 4]).unwrap();
        let flat = tensor.flatten().unwrap();

        assert_eq!(flat.dims(), &[24]);
    }

    #[test]
    fn test_to_cpu() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let tensor = Tensor::from_vec_metal(&device, data.clone(), vec![2]).unwrap();

        let cpu_tensor = tensor.to_cpu().unwrap();
        assert!(cpu_tensor.buffer().is_cpu());
        assert_eq!(cpu_tensor.to_vec(), data);
    }
}
