//! Element-wise tensor operations with Metal GPU acceleration

use crate::autograd::gradients::{AddBackward, DivBackward, MulBackward, SubBackward};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform, TensorAutograd};
use crate::autograd::{AutogradContext, GradientFunction, GradientFunctionGeneric, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Helper function to record a binary operation in the computation graph
fn record_binary_op<T: FloatType>(
    op: Operation,
    grad_fn: Box<dyn GradientFunctionGeneric<T>>,
    self_tensor: &Tensor<T>,
    other_tensor: &Tensor<T>,
    result: &mut Tensor<T>,
) where
    Tensor<T>: TensorAutograd<T>,
{
    if !(self_tensor.requires_grad() || other_tensor.requires_grad()) || !AutogradContext::is_enabled() {
        return;
    }

    // Get or allocate node IDs for input tensors
    let self_node_id = self_tensor.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());
    let other_node_id = other_tensor.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());

    // Add node to computation graph (this allocates the result node ID internally)
    let result_node_id = AutogradContext::add_node_generic(
        op,
        vec![self_node_id, other_node_id],
        Some(grad_fn),
    );

    // Register tensors
    AutogradContext::register_tensor_generic(self_node_id, self_tensor.clone());
    AutogradContext::register_tensor_generic(other_node_id, other_tensor.clone());

    // Set node ID and requires_grad on result
    result.set_grad_node(result_node_id);
    result.set_requires_grad(true);
}

impl<T: FloatType> Tensor<T> {
    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
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
    fn add_metal(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                format!("Metal operations currently only support f16, got {}", std::any::type_name::<T>())
            ));
        }

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

        // Safety: We checked T::is_f16() above, so we can safely transmute to MetalBuffer<f16>
        let a_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(a_buf) };
        let b_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(b_buf) };

        // Create result buffer (f16)
        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        // Create local executor for this operation
        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_binary_op("add_f16", a_buf_f16, b_buf_f16, &result_buf)?;

        // Safety: We're working with f16, so transmute back to T (which is f16)
        let result_buf_t: MetalBuffer<T> = unsafe { std::mem::transmute(result_buf) };

        // Create result tensor
        self.new_from_pool(
            BufferHandle::Metal(result_buf_t),
            self.shape().clone(),
        )
    }

    /// CPU fallback for addition
    fn add_cpu(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        let b = other.to_vec();

        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let b_f16: Vec<f16> = unsafe { std::mem::transmute(b) };
        let result: Vec<f16> = a_f16.iter().zip(b_f16.iter()).map(|(&x, &y)| x + y).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        // Keep result on same device as self
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
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

    fn sub_metal(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
        let a_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(a_buf) };
        let b_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(b_buf) };
        executor.execute_binary_op("sub_f16", a_buf_f16, b_buf_f16, &result_buf)?;

        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    fn sub_cpu(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        let b = other.to_vec();

        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let b_f16: Vec<f16> = unsafe { std::mem::transmute(b) };
        let result: Vec<f16> = a_f16.iter().zip(b_f16.iter()).map(|(&x, &y)| x - y).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
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

    fn mul_metal(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
        let a_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(a_buf) };
        let b_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(b_buf) };
        executor.execute_binary_op("mul_f16", a_buf_f16, b_buf_f16, &result_buf)?;

        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    fn mul_cpu(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        let b = other.to_vec();

        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let b_f16: Vec<f16> = unsafe { std::mem::transmute(b) };
        let result: Vec<f16> = a_f16.iter().zip(b_f16.iter()).map(|(&x, &y)| x * y).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
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

    fn div_metal(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
        let a_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(a_buf) };
        let b_buf_f16: &MetalBuffer<half::f16> = unsafe { std::mem::transmute(b_buf) };
        executor.execute_binary_op("div_f16", a_buf_f16, b_buf_f16, &result_buf)?;

        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    fn div_cpu(&self, other: &Tensor<T>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        let b = other.to_vec();

        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let b_f16: Vec<f16> = unsafe { std::mem::transmute(b) };
        let result: Vec<f16> = a_f16.iter().zip(b_f16.iter()).map(|(&x, &y)| x / y).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise exponential: e^x
    pub fn exp(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.exp_metal()
        } else {
            self.exp_cpu()
        }
    }

    fn exp_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "exp_f16")
    }

    fn exp_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.exp())
    }

    /// Element-wise natural logarithm: log(x)
    pub fn log(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.log_metal()
        } else {
            self.log_cpu()
        }
    }

    fn log_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "log_f16")
    }

    fn log_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.ln())
    }

    /// Element-wise square root: sqrt(x)
    pub fn sqrt(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.sqrt_metal()
        } else {
            self.sqrt_cpu()
        }
    }

    fn sqrt_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "sqrt_f16")
    }

    fn sqrt_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.sqrt())
    }

    /// Element-wise power: x^exponent
    pub fn pow(&self, exponent: f32) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.pow_metal(exponent)
        } else {
            self.pow_cpu(exponent)
        }
    }

    fn pow_metal(&self, exponent: f32) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let device = match self.device() {
            Device::Metal(dev) => dev,
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };
        let exp_tensor = Tensor::<T>::from_vec_metal(device, vec![T::from_f32(exponent)], vec![1])?;
        super::helpers::execute_binary_metal_op(self, &exp_tensor, "pow_f16")
    }

    fn pow_cpu(&self, exponent: f32) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_binary_cpu_op(self, exponent, |x, exp| x.powf(exp))
    }

    /// Element-wise sine: sin(x)
    pub fn sin(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.sin_metal()
        } else {
            self.sin_cpu()
        }
    }

    fn sin_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "sin_f16")
    }

    fn sin_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.sin())
    }

    /// Element-wise cosine: cos(x)
    pub fn cos(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.cos_metal()
        } else {
            self.cos_cpu()
        }
    }

    fn cos_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "cos_f16")
    }

    fn cos_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.cos())
    }

    /// Element-wise tangent: tan(x)
    pub fn tan(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.tan_metal()
        } else {
            self.tan_cpu()
        }
    }

    fn tan_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "tan_f16")
    }

    fn tan_cpu(&self) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_cpu_op(self, |x| x.tan())
    }

    /// Element-wise addition with a scalar
    pub fn add_scalar(&self, scalar: half::f16) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.add_scalar_metal(scalar)
        } else {
            self.add_scalar_cpu(scalar)
        }
    }

    fn add_scalar_metal(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x + scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    fn add_scalar_cpu(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x + scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise subtraction with a scalar
    pub fn sub_scalar(&self, scalar: half::f16) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.sub_scalar_metal(scalar)
        } else {
            self.sub_scalar_cpu(scalar)
        }
    }

    fn sub_scalar_metal(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x - scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    fn sub_scalar_cpu(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x - scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise multiplication with a scalar
    pub fn mul_scalar(&self, scalar: half::f16) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.mul_scalar_metal(scalar)
        } else {
            self.mul_scalar_cpu(scalar)
        }
    }

    fn mul_scalar_metal(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x * scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    fn mul_scalar_cpu(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x * scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    /// Element-wise division with a scalar
    pub fn div_scalar(&self, scalar: half::f16) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.div_scalar_metal(scalar)
        } else {
            self.div_scalar_cpu(scalar)
        }
    }

    fn div_scalar_metal(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x / scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
        }
    }

    fn div_scalar_cpu(&self, scalar: half::f16) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        // Safety: We checked T::is_f16() above
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let result: Vec<f16> = a_f16.iter().map(|&x| x / scalar).collect();
        let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, result_t, self.dims().to_vec()),
            _ => Tensor::from_vec(result_t, self.dims().to_vec()),
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

        let a = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();
        let b = Tensor::<f16>::zeros(&device, vec![3, 2]).unwrap();

        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_exp() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)],
            vec![3],
        )
        .unwrap();

        let result = a.exp().unwrap();
        let values = result.to_vec();

        assert!((values[0].to_f32() - 1.0).abs() < 0.01);
        assert!((values[1].to_f32() - 2.718).abs() < 0.01);
        assert!((values[2].to_f32() - 7.389).abs() < 0.01);
    }

    #[test]
    fn test_log() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.718), f16::from_f32(7.389)],
            vec![3],
        )
        .unwrap();

        let result = a.log().unwrap();
        let values = result.to_vec();

        assert!((values[0].to_f32() - 0.0).abs() < 0.01);
        assert!((values[1].to_f32() - 1.0).abs() < 0.01);
        assert!((values[2].to_f32() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_sqrt() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(4.0), f16::from_f32(9.0)],
            vec![3],
        )
        .unwrap();

        let result = a.sqrt().unwrap();
        let expected = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pow() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let result = a.pow(2.0).unwrap();
        let expected = vec![f16::from_f32(4.0), f16::from_f32(9.0), f16::from_f32(16.0)];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_sin() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0), f16::from_f32(std::f32::consts::PI)],
            vec![3],
        )
        .unwrap();

        let result = a.sin().unwrap();
        let values = result.to_vec();

        assert!((values[0].to_f32() - 0.0).abs() < 0.01);
        assert!((values[1].to_f32() - 1.0).abs() < 0.01);
        assert!((values[2].to_f32() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_cos() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0), f16::from_f32(std::f32::consts::PI)],
            vec![3],
        )
        .unwrap();

        let result = a.cos().unwrap();
        let values = result.to_vec();

        assert!((values[0].to_f32() - 1.0).abs() < 0.01);
        assert!((values[1].to_f32() - 0.0).abs() < 0.01);
        assert!((values[2].to_f32() + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tan() {
        let device = get_test_device();
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 4.0)],
            vec![2],
        )
        .unwrap();

        let result = a.tan().unwrap();
        let values = result.to_vec();

        assert!((values[0].to_f32() - 0.0).abs() < 0.01);
        assert!((values[1].to_f32() - 1.0).abs() < 0.01);
    }
}
