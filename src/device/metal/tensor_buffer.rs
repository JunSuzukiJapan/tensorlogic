use crate::device::{MetalDevice, MetalBuffer};
use crate::tensor::{FloatType, Tensor, TensorCreation};
use crate::error::{TensorError, TensorResult};
use metal::{MTLResourceOptions, Buffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;
use std::collections::HashMap;

/// Shape key for tensor pool (shape dimensions as Vec)
type ShapeKey = Vec<usize>;

/// Pre-allocated GPU buffer for creating tensors without allocation overhead
pub struct TensorBuffer {
    device: MetalDevice,
    buffer: Arc<Buffer>,
    capacity: usize,
    used: AtomicUsize,
    /// Free list for recycled tensors (shape -> list of available offsets and sizes)
    free_lists: Arc<Mutex<HashMap<ShapeKey, Vec<(usize, usize)>>>>,
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
            free_lists: Arc::new(Mutex::new(HashMap::new())),
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

    /// Try to allocate from free list, or allocate new space
    fn allocate_or_reuse(&self, shape: &[usize], elem_size: usize) -> TensorResult<(usize, usize)> {
        let shape_key: ShapeKey = shape.to_vec();
        let size = shape.iter().product::<usize>() * elem_size;

        // Try to reuse from free list
        {
            let mut free_lists = self.free_lists.lock().unwrap();
            if let Some(free_list) = free_lists.get_mut(&shape_key) {
                if let Some((offset, reused_size)) = free_list.pop() {
                    return Ok((offset, reused_size));
                }
            }
        }

        // No free slot available, allocate new space
        self.allocate_space(size)
    }

    /// Pre-allocate tensor slots for a specific shape
    ///
    /// # Arguments
    /// * `shape` - Shape of tensors to pre-allocate
    /// * `count` - Number of tensor slots to pre-allocate
    /// * `elem_size` - Size of each element in bytes
    ///
    /// # Example
    /// ```
    /// let buf = device.new_tensor_buffer(10 * 1024 * 1024);
    /// buf.alloc(&[512], 5, 2)?;  // Pre-allocate 5 slots of [512] tensors (f16)
    /// ```
    pub fn alloc(&self, shape: &[usize], count: usize, elem_size: usize) -> TensorResult<()> {
        let shape_key: ShapeKey = shape.to_vec();
        let size = shape.iter().product::<usize>() * elem_size;

        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            let (offset, size) = self.allocate_space(size)?;
            offsets.push((offset, size));
        }

        // Add to free list
        let mut free_lists = self.free_lists.lock().unwrap();
        free_lists.insert(shape_key, offsets);

        Ok(())
    }

    /// Return a tensor slot to the free list for reuse
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor being recycled
    /// * `offset` - Buffer offset of the tensor
    /// * `size` - Size of the tensor in bytes
    ///
    /// Note: This does not free the GPU memory, it just marks the slot as available for reuse
    pub fn recycle(&self, shape: &[usize], offset: usize, size: usize) {
        let shape_key: ShapeKey = shape.to_vec();
        let mut free_lists = self.free_lists.lock().unwrap();
        free_lists.entry(shape_key)
            .or_insert_with(Vec::new)
            .push((offset, size));
    }

    /// Clear all free lists and reset buffer
    pub fn clear_all(&self) {
        let mut free_lists = self.free_lists.lock().unwrap();
        free_lists.clear();
        self.used.store(0, Ordering::SeqCst);
    }

    /// Clear free list for a specific shape
    pub fn clear_shape(&self, shape: &[usize]) {
        let shape_key: ShapeKey = shape.to_vec();
        let mut free_lists = self.free_lists.lock().unwrap();
        free_lists.remove(&shape_key);
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
    /// Reuses pre-allocated slots if available
    pub fn zeros<T: FloatType>(&self, shape: Vec<usize>) -> TensorResult<Tensor<T>> {
        let numel: usize = shape.iter().product();
        let elem_size = std::mem::size_of::<T>();
        let size = numel * elem_size;

        let (offset, _) = self.allocate_or_reuse(&shape, elem_size)?;

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
            device: self.device.clone(),
        };

        let buffer = crate::tensor::BufferHandle::Metal(metal_buffer);
        let shape = crate::tensor::TensorShape::new(shape);

        Tensor::new(buffer, shape, crate::device::Device::Metal(self.device.clone()))
    }

    /// Create a tensor filled with ones from this buffer
    /// Reuses pre-allocated slots if available
    pub fn ones<T: FloatType>(&self, shape: Vec<usize>) -> TensorResult<Tensor<T>> {
        let numel: usize = shape.iter().product();
        let elem_size = std::mem::size_of::<T>();

        let (offset, _) = self.allocate_or_reuse(&shape, elem_size)?;

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
            device: self.device.clone(),
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

    #[test]
    fn test_tensor_buffer_alloc() {
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Pre-allocate 5 slots for [10, 10] tensors (f16 = 2 bytes)
        buffer.alloc(&[10, 10], 5, std::mem::size_of::<f16>()).unwrap();

        let expected_used = 5 * 10 * 10 * 2; // 5 tensors * 100 elements * 2 bytes
        assert_eq!(buffer.used_bytes(), expected_used);
    }

    #[test]
    fn test_tensor_buffer_recycle() {
        use crate::tensor::{Tensor, TensorCreation};
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Pre-allocate 3 slots for [10, 10] tensors
        buffer.alloc(&[10, 10], 3, std::mem::size_of::<f16>()).unwrap();

        let initial_used = buffer.used_bytes();

        // Request a tensor - should reuse from free list
        let tensor1: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();
        assert_eq!(tensor1.dims(), &[10, 10]);

        // Used bytes should not increase (reused pre-allocated slot)
        assert_eq!(buffer.used_bytes(), initial_used);

        // Request another tensor - should also reuse
        let tensor2: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();
        assert_eq!(buffer.used_bytes(), initial_used);

        // Request a 4th tensor - should allocate new (only 3 were pre-allocated)
        let tensor3: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();
        let tensor4: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();

        // Now should have allocated a new slot
        assert!(buffer.used_bytes() > initial_used);
    }

    #[test]
    fn test_tensor_buffer_clear_all() {
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Allocate some slots
        buffer.alloc(&[10, 10], 3, std::mem::size_of::<f16>()).unwrap();
        buffer.alloc(&[5, 5], 2, std::mem::size_of::<f16>()).unwrap();

        assert!(buffer.used_bytes() > 0);

        // Clear all
        buffer.clear_all();

        assert_eq!(buffer.used_bytes(), 0);
        assert_eq!(buffer.available_bytes(), 10 * 1024);
    }

    #[test]
    fn test_tensor_buffer_clear_shape() {
        use crate::tensor::{Tensor, TensorCreation};
        use half::f16;

        let device = MetalDevice::new().unwrap();
        let buffer = device.new_tensor_buffer(10 * 1024); // 10KB

        // Pre-allocate different shapes
        buffer.alloc(&[10, 10], 2, std::mem::size_of::<f16>()).unwrap();
        buffer.alloc(&[5, 5], 2, std::mem::size_of::<f16>()).unwrap();

        let used_before = buffer.used_bytes();

        // Clear [10, 10] shape
        buffer.clear_shape(&[10, 10]);

        // Used bytes unchanged (clear_shape only removes from free list)
        assert_eq!(buffer.used_bytes(), used_before);

        // Request [10, 10] - should allocate new (free list cleared)
        let tensor: Tensor<f16> = buffer.zeros(vec![10, 10]).unwrap();
        assert!(buffer.used_bytes() > used_before);
    }
}
