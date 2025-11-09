use crate::device::{MetalDevice, MetalBuffer};
use crate::tensor::{FloatType, Tensor, TensorCreation};
use crate::error::{TensorError, TensorResult};
use metal::{MTLResourceOptions, Buffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::marker::PhantomData;

/// Pre-allocated GPU buffer for creating tensors without allocation overhead
pub struct TensorBuffer {
    device: MetalDevice,
    buffer: Arc<Buffer>,
    capacity: usize,
    used: AtomicUsize,
}

impl TensorBuffer {
    /// Create a new tensor buffer with specified capacity (internal use only)
    pub(crate) fn new(device: MetalDevice, capacity: usize) -> Self {
        // Use StorageModeShared to allow CPU access for initialization
        let buffer = device.metal_device().new_buffer(
            capacity as u64,
            MTLResourceOptions::StorageModeShared
        );
        Self {
            device,
            buffer: Arc::new(buffer),
            capacity,
            used: AtomicUsize::new(0),
        }
    }

    /// Allocate space from this buffer
    /// Returns (offset, size) for the allocated region
    pub fn allocate_space(&self, size: usize) -> TensorResult<(usize, usize)> {
        let offset = self.used.fetch_add(size, Ordering::SeqCst);

        // Error on exhaustion (no fallback to BufferPool)
        if offset + size > self.capacity {
            // Rollback the allocation
            self.used.fetch_sub(size, Ordering::SeqCst);
            return Err(TensorError::BufferExhausted {
                requested: size,
                available: self.capacity.saturating_sub(offset),
            });
        }

        Ok((offset, size))
    }

    /// Get the underlying buffer
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Get the device
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Reset buffer offset to 0 (allows reusing from the beginning)
    pub fn reset(&self) {
        self.used.store(0, Ordering::SeqCst);
    }

    /// Get current usage in bytes
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }

    /// Get total capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get available bytes
    pub fn available_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.used.load(Ordering::SeqCst))
    }

    /// Create a tensor filled with zeros from this buffer
    pub fn zeros<T: FloatType>(&self, shape: Vec<usize>) -> TensorResult<Tensor<T>> {
        let numel: usize = shape.iter().product();
        let size = numel * std::mem::size_of::<T>();

        let (offset, _) = self.allocate_space(size)?;

        // Zero initialize using CPU (StorageModeShared allows direct access)
        unsafe {
            let ptr = self.buffer.contents().add(offset) as *mut u8;
            std::ptr::write_bytes(ptr, 0, size);
        }

        // Create sub-buffer view
        let sub_buffer: SubMetalBuffer<T> = SubMetalBuffer {
            base_buffer: self.buffer.clone(),
            offset,
            length: numel,
            _phantom: PhantomData,
        };

        // Create MetalBuffer wrapper
        let metal_buffer = MetalBuffer {
            buffer: sub_buffer.base_buffer.clone(),
            length: numel,
            _phantom: PhantomData,
            pool: None,
            size_class: None,
        };

        let buffer = crate::tensor::BufferHandle::Metal(metal_buffer);
        let shape = crate::tensor::TensorShape::new(shape);

        Tensor::new(buffer, shape, crate::device::Device::Metal(self.device.clone()))
    }

    /// Create a tensor filled with ones from this buffer
    pub fn ones<T: FloatType>(&self, shape: Vec<usize>) -> TensorResult<Tensor<T>> {
        let numel: usize = shape.iter().product();
        let size = numel * std::mem::size_of::<T>();

        let (offset, _) = self.allocate_space(size)?;

        // Create temporary CPU buffer with ones
        let ones_data = vec![T::one(); numel];

        // Copy to GPU buffer at offset
        unsafe {
            let ptr = self.buffer.contents().add(offset) as *mut T;
            std::ptr::copy_nonoverlapping(ones_data.as_ptr(), ptr, numel);
        }

        // Create MetalBuffer wrapper
        let metal_buffer = MetalBuffer {
            buffer: self.buffer.clone(),
            length: numel,
            _phantom: PhantomData,
            pool: None,
            size_class: None,
        };

        let buffer = crate::tensor::BufferHandle::Metal(metal_buffer);
        let shape = crate::tensor::TensorShape::new(shape);

        Tensor::new(buffer, shape, crate::device::Device::Metal(self.device.clone()))
    }
}

/// Sub-buffer view for TensorBuffer allocations (internal use)
#[allow(dead_code)]
struct SubMetalBuffer<T: FloatType> {
    base_buffer: Arc<Buffer>,
    offset: usize,
    length: usize,
    _phantom: PhantomData<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use crate::tensor::TensorAccessors;

    #[test]
    fn test_tensor_buffer_creation() {
        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(1024 * 1024); // 1MB

        assert_eq!(buffer.capacity(), 1024 * 1024);
        assert_eq!(buffer.used_bytes(), 0);
        assert_eq!(buffer.available_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_tensor_buffer_allocation() {
        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(1024);

        // Allocate 512 bytes
        let result = buffer.allocate_space(512);
        assert!(result.is_ok());
        let (offset, size) = result.unwrap();
        assert_eq!(offset, 0);
        assert_eq!(size, 512);
        assert_eq!(buffer.used_bytes(), 512);
        assert_eq!(buffer.available_bytes(), 512);

        // Allocate another 256 bytes
        let result = buffer.allocate_space(256);
        assert!(result.is_ok());
        let (offset, size) = result.unwrap();
        assert_eq!(offset, 512);
        assert_eq!(size, 256);
        assert_eq!(buffer.used_bytes(), 768);
        assert_eq!(buffer.available_bytes(), 256);
    }

    #[test]
    fn test_tensor_buffer_exhaustion() {
        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(1024);

        // Allocate 512 bytes - should succeed
        let result = buffer.allocate_space(512);
        assert!(result.is_ok());

        // Try to allocate 600 bytes - should fail (only 512 bytes left)
        let result = buffer.allocate_space(600);
        assert!(result.is_err());

        // Verify error type
        match result {
            Err(TensorError::BufferExhausted { requested, available }) => {
                assert_eq!(requested, 600);
                assert_eq!(available, 512);
            }
            _ => panic!("Expected BufferExhausted error"),
        }

        // Verify used bytes wasn't incremented on failure
        assert_eq!(buffer.used_bytes(), 512);
    }

    #[test]
    fn test_tensor_buffer_reset() {
        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(1024);

        // Allocate some space
        buffer.allocate_space(512).unwrap();
        assert_eq!(buffer.used_bytes(), 512);

        // Reset
        buffer.reset();
        assert_eq!(buffer.used_bytes(), 0);
        assert_eq!(buffer.available_bytes(), 1024);

        // Should be able to allocate again from beginning
        let result = buffer.allocate_space(256);
        assert!(result.is_ok());
        let (offset, _) = result.unwrap();
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_tensor_buffer_zeros() {
        use crate::tensor::{Tensor, TensorCreation};
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Create a zeros tensor from buffer
        let tensor: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();

        assert_eq!(tensor.dims(), &[10, 10]);
        assert_eq!(buffer.used_bytes(), 10 * 10 * 2); // f16 is 2 bytes
    }

    #[test]
    fn test_tensor_buffer_ones() {
        use crate::tensor::{Tensor, TensorCreation};
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Create a ones tensor from buffer
        let tensor: Tensor<f16> = buffer.ones(vec![5, 5]).unwrap();

        assert_eq!(tensor.dims(), &[5, 5]);
        assert_eq!(buffer.used_bytes(), 5 * 5 * 2); // f16 is 2 bytes
    }

    #[test]
    fn test_tensor_buffer_multiple_tensors() {
        use crate::tensor::{Tensor, TensorCreation};
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Create multiple tensors from the same buffer
        let tensor1: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();
        let tensor2: Tensor<f16> = buffer.ones(vec![5, 5]).unwrap();
        let tensor3: Tensor<f16> = buffer.zeros(vec![20, 20]).unwrap();

        assert_eq!(tensor1.dims(), &[10, 10]);
        assert_eq!(tensor2.dims(), &[5, 5]);
        assert_eq!(tensor3.dims(), &[20, 20]);

        let expected_used = (10*10 + 5*5 + 20*20) * 2; // f16 is 2 bytes
        assert_eq!(buffer.used_bytes(), expected_used);
    }
}
