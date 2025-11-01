//! Tensor creation methods

use crate::device::{BufferPool, Device, MetalBuffer, MetalDevice};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, FloatType, Tensor, TensorShape};
use crate::tensor::TensorAccessors;
use half::f16;
use std::marker::PhantomData;

/// Trait for creating tensors
pub trait TensorCreation<T: FloatType>: Sized {
    /// Create a new tensor from buffer and shape
    ///
    /// Automatically extracts buffer_pool from Metal device for efficient memory management.
    /// For Metal tensors, this enables automatic buffer recycling when the tensor is dropped.
    fn new(buffer: BufferHandle<T>, shape: TensorShape, device: Device) -> TensorResult<Self>;

    /// Create a new tensor with buffer pool support for automatic recycling (internal use)
    fn new_with_pool(
        buffer: BufferHandle<T>,
        shape: TensorShape,
        device: Device,
        buffer_pool: Option<BufferPool>,
    ) -> TensorResult<Self>;

    /// Create a new tensor from this tensor's buffer pool (internal use)
    /// Useful for operations that create output tensors
    fn new_from_pool(
        &self,
        buffer: BufferHandle<T>,
        shape: TensorShape,
    ) -> TensorResult<Self>;

    /// Create a tensor from vector on CPU
    fn from_vec(data: Vec<T>, shape: Vec<usize>) -> TensorResult<Self>;

    /// Create a tensor from vector on Metal device
    fn from_vec_gpu(
        device: &MetalDevice,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> TensorResult<Self>;

    /// Create a tensor from vector on Metal device using buffer pool
    fn from_vec_gpu_pooled(
        device: &MetalDevice,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> TensorResult<Self>;

    /// Create a tensor filled with zeros on Metal device
    fn zeros(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self>;

    /// Create a tensor filled with ones on Metal device
    fn ones(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self>;

    /// Create a scalar tensor
    fn scalar(device: &MetalDevice, value: T) -> TensorResult<Self>;
}

impl<T: FloatType> TensorCreation<T> for Tensor<T> {
    fn new(buffer: BufferHandle<T>, shape: TensorShape, device: Device) -> TensorResult<Self> {
        // Automatically get buffer_pool from Metal device
        let buffer_pool = match &device {
            Device::Metal(metal_device) => Some(metal_device.buffer_pool().clone()),
            #[allow(unreachable_patterns)]
            _ => None,
        };

        Self::new_with_pool(buffer, shape, device, buffer_pool)
    }

    fn new_with_pool(
        buffer: BufferHandle<T>,
        shape: TensorShape,
        device: Device,
        buffer_pool: Option<BufferPool>,
    ) -> TensorResult<Self> {
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
            buffer_pool,
            _phantom: PhantomData,
        })
    }

    fn new_from_pool(
        &self,
        buffer: BufferHandle<T>,
        shape: TensorShape,
    ) -> TensorResult<Self> {
        Self::new_with_pool(
            buffer,
            shape,
            self.device.clone(),
            self.buffer_pool.clone(),
        )
    }

    fn from_vec(data: Vec<T>, shape: Vec<usize>) -> TensorResult<Self> {
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

    fn from_vec_gpu(
        device: &MetalDevice,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);

        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }

        let metal_buffer = MetalBuffer::from_slice(device.metal_device(), &data)?;
        let buffer = BufferHandle::Metal(metal_buffer);

        let tensor = Self::new(buffer, shape, Device::Metal(device.clone()))?;

        // Force synchronization to ensure GPU buffer is fully initialized
        // This prevents race conditions when the tensor is used immediately after creation
        use crate::tensor::TensorIO;
        let _ = tensor.sync_and_read();

        Ok(tensor)
    }

    fn from_vec_gpu_pooled(
        device: &MetalDevice,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);

        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                actual: vec![data.len()],
            });
        }

        let data_f16: Vec<f16> = unsafe { std::mem::transmute(data) };
        let metal_buffer = MetalBuffer::from_vec_pooled(device.buffer_pool(), &data_f16)?;
        let buffer = BufferHandle::Metal(unsafe { std::mem::transmute(metal_buffer) });

        let tensor = Self::new_with_pool(
            buffer,
            shape,
            Device::Metal(device.clone()),
            Some(device.buffer_pool().clone()),
        )?;

        // Force synchronization to ensure GPU buffer is fully initialized
        use crate::tensor::TensorIO;
        let _ = tensor.sync_and_read();

        Ok(tensor)
    }

    fn zeros(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);
        // Use buffer pool for allocation
        let metal_buffer = MetalBuffer::<T>::zeros_pooled(device.buffer_pool(), shape.numel())?;
        let buffer = BufferHandle::Metal(unsafe { std::mem::transmute(metal_buffer) });

        Self::new_with_pool(
            buffer,
            shape,
            Device::Metal(device.clone()),
            Some(device.buffer_pool().clone()),
        )
    }

    fn ones(device: &MetalDevice, shape: Vec<usize>) -> TensorResult<Self> {
        let shape = TensorShape::new(shape);
        let metal_buffer = MetalBuffer::ones(device.metal_device(), shape.numel())?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, shape, Device::Metal(device.clone()))
    }

    fn scalar(device: &MetalDevice, value: T) -> TensorResult<Self> {
        Self::from_vec_gpu(device, vec![value], vec![1])
    }
}

// Note: from_vec_gpu_pooled supports both f16 and f32 through MetalBuffer::from_vec_pooled
