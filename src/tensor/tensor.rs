//! Core Tensor type

use crate::autograd::NodeId;
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

    /// Computation graph node ID (if part of a computation graph)
    grad_node: Option<NodeId>,

    /// Version counter for gradient accumulation tracking
    version: u64,
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
            grad_node: None,
            version: 0,
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
        // Use buffer pool for allocation
        let metal_buffer = MetalBuffer::zeros_pooled(device.buffer_pool(), shape.numel())?;
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

        // Allocate a node ID for this tensor if it doesn't have one and requires_grad is true
        if requires && self.grad_node.is_none() {
            use crate::autograd::AutogradContext;
            let node_id = AutogradContext::allocate_id();
            self.grad_node = Some(node_id);
            AutogradContext::register_tensor(node_id, self.clone());
        }
    }

    /// Get the gradient
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Zero out the gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
        self.version += 1;
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
            grad_node: None,
            version: 0,
        })
    }

    /// Get tensor as a 1D view
    pub fn flatten(&self) -> TensorResult<Self> {
        self.reshape(vec![self.numel()])
    }

    // === Save/Load operations ===

    /// Save tensor to a binary file
    ///
    /// Format: [num_dims: u32][dim0: u32][dim1: u32]...[data: f16...]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> TensorResult<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to create file: {}", e)
        ))?;

        // Write number of dimensions
        let num_dims = self.dims().len() as u32;
        file.write_all(&num_dims.to_le_bytes()).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to write dimensions count: {}", e)
        ))?;

        // Write each dimension
        for &dim in self.dims() {
            file.write_all(&(dim as u32).to_le_bytes()).map_err(|e| TensorError::InvalidOperation(
                format!("Failed to write dimension: {}", e)
            ))?;
        }

        // Write tensor data (f16)
        let data = self.to_vec();
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        file.write_all(&bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to write tensor data: {}", e)
        ))?;

        Ok(())
    }

    /// Load tensor from a binary file
    ///
    /// Format: [num_dims: u32][dim0: u32][dim1: u32]...[data: f16...]
    pub fn load<P: AsRef<std::path::Path>>(device: &Device, path: P) -> TensorResult<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to open file: {}", e)
        ))?;

        // Read number of dimensions
        let mut num_dims_bytes = [0u8; 4];
        file.read_exact(&mut num_dims_bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to read dimensions count: {}", e)
        ))?;
        let num_dims = u32::from_le_bytes(num_dims_bytes) as usize;

        // Read each dimension
        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let mut dim_bytes = [0u8; 4];
            file.read_exact(&mut dim_bytes).map_err(|e| TensorError::InvalidOperation(
                format!("Failed to read dimension: {}", e)
            ))?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        // Calculate expected data size
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * 2; // 2 bytes per f16

        // Read tensor data
        let mut bytes = vec![0u8; expected_bytes];
        file.read_exact(&mut bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to read tensor data: {}", e)
        ))?;

        // Convert bytes to f16 vector
        let data: Vec<f16> = bytes.chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        // Create tensor on specified device
        match device {
            Device::Metal(metal_device) => Self::from_vec_metal(metal_device, data, shape),
            Device::CPU => Self::from_vec(data, shape),
            Device::NeuralEngine => {
                // NeuralEngine uses Metal backend for tensor storage
                // We need a Metal device - for now, create a new one
                let metal = MetalDevice::new()?;
                Self::from_vec_metal(&metal, data, shape)
            }
        }
    }

    // === Autograd operations ===

    /// Set gradient (public for optimizer use)
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(Box::new(grad));
    }

    /// Get computation graph node ID
    pub fn grad_node(&self) -> Option<NodeId> {
        self.grad_node
    }

    /// Set computation graph node ID (internal use)
    pub(crate) fn set_grad_node(&mut self, node_id: NodeId) {
        self.grad_node = Some(node_id);
    }

    /// Get version (for gradient accumulation tracking)
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Perform backward pass (for scalar tensors)
    pub fn backward(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward() can only be called on scalar tensors. Use backward_with_grad() for non-scalar tensors."
                    .to_string(),
            ));
        }

        // 初期勾配は1.0
        let grad = Tensor::from_vec(vec![f16::ONE], vec![1])?;
        self.backward_with_grad(grad)
    }

    /// Perform backward pass with specified gradient
    pub fn backward_with_grad(&mut self, grad: Tensor) -> TensorResult<()> {
        self.backward_impl(grad, false)
    }

    /// Perform backward pass with computation graph creation (for higher-order derivatives)
    pub fn backward_create_graph(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward_create_graph() can only be called on scalar tensors."
                    .to_string(),
            ));
        }

        let grad = Tensor::from_vec(vec![f16::ONE], vec![1])?;
        self.backward_impl(grad, true)
    }

    /// Internal backward implementation
    fn backward_impl(&mut self, grad: Tensor, create_graph: bool) -> TensorResult<()> {
        use crate::autograd::AutogradContext;

        // Get node ID for this tensor
        let node_id = self.grad_node.ok_or_else(|| {
            TensorError::InvalidOperation(
                "Cannot call backward on tensor without computation graph node".to_string(),
            )
        })?;

        // Perform backward pass through computation graph
        let gradients = if create_graph {
            AutogradContext::backward_with_graph(node_id, grad)?
        } else {
            AutogradContext::backward(node_id, grad)?
        };

        // Distribute gradients to all tensors in the graph
        // If create_graph is true, enable gradient recording for gradient accumulation
        if create_graph {
            AutogradContext::set_enabled(true);
        }

        for (tensor_node_id, gradient) in gradients {
            if let Some(mut tensor) = AutogradContext::get_tensor(tensor_node_id) {
                // Accumulate gradient if it already exists
                if let Some(existing_grad) = tensor.grad() {
                    let accumulated_grad = if create_graph {
                        // With create_graph, gradient accumulation should be recorded
                        existing_grad.add(&gradient).unwrap()
                    } else {
                        // Without create_graph, use no_grad for accumulation
                        AutogradContext::no_grad(|| existing_grad.add(&gradient).unwrap())
                    };
                    tensor.set_grad(accumulated_grad);
                } else {
                    // If create_graph is true, make gradient tensor require grad
                    let mut grad_tensor = gradient;
                    if create_graph {
                        grad_tensor.set_requires_grad(true);
                    }
                    tensor.set_grad(grad_tensor);
                }

                // Re-register the updated tensor
                AutogradContext::register_tensor(tensor_node_id, tensor);
            }
        }

        // Disable gradient recording after distribution
        if create_graph {
            AutogradContext::set_enabled(false);
        }

        Ok(())
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

    #[test]
    fn test_save_load() {
        use std::fs;

        let metal_device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let tensor = Tensor::from_vec_metal(&metal_device, data.clone(), vec![2, 2]).unwrap();

        let path = "/tmp/test_tensor.bin";
        tensor.save(path).unwrap();

        let device = Device::Metal(metal_device);
        let loaded = Tensor::load(&device, path).unwrap();
        assert_eq!(loaded.dims(), tensor.dims());
        assert_eq!(loaded.to_vec(), data);

        // Cleanup
        fs::remove_file(path).ok();
    }
}
