//! Metal buffer management for generic floating-point data

use crate::device::{BufferPool, NeuralEngineBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::FloatType;
use half::f16;
use metal::{Buffer, Device as MTLDevice};
use std::marker::PhantomData;
use std::sync::Arc;

/// Metal buffer wrapper for generic floating-point data
#[derive(Debug, Clone)]
pub struct MetalBuffer<T: FloatType> {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) length: usize, // number of T elements
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: FloatType> MetalBuffer<T> {
    /// Create a new Metal buffer from slice
    pub fn from_slice(device: &MTLDevice, data: &[T]) -> TensorResult<Self> {
        let byte_length = data.len() * T::size_in_bytes();

        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length: data.len(),
            _phantom: PhantomData,
        })
    }

    /// Create a new uninitialized Metal buffer
    pub fn new_uninit(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        let buffer = device.new_buffer(
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length,
            _phantom: PhantomData,
        })
    }

    /// Create a new Metal buffer filled with zeros
    pub fn zeros(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let zeros = vec![T::zero(); length];
        Self::from_slice(device, &zeros)
    }

    /// Create a new Metal buffer filled with ones
    pub fn ones(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let ones = vec![T::one(); length];
        Self::from_slice(device, &ones)
    }

    // Pool methods are f16-only, see impl MetalBuffer<f16> below

    /// Get the buffer length (number of T elements)
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the buffer byte length
    pub fn byte_length(&self) -> usize {
        self.length * T::size_in_bytes()
    }

    /// Get the underlying Metal buffer
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Read data from buffer to Vec<T>
    pub fn to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.length).to_vec() }
    }

    /// Write data to buffer from slice
    pub fn write_from_slice(&mut self, data: &[T]) -> TensorResult<()> {
        if data.len() != self.length {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.length],
                actual: vec![data.len()],
            });
        }

        let ptr = self.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, self.length);
        }

        Ok(())
    }

    /// Get a mutable pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents_mut(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    /// Get a const pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    /// Convert to Neural Engine buffer (with data copy, f16 only)
    ///
    /// Note: This performs a data copy. Zero-copy conversion will be implemented in Phase 5.
    pub fn to_neural_engine(&self, shape: &[usize]) -> TensorResult<NeuralEngineBuffer>
    where
        T: 'static,
    {
        // Validate shape matches buffer size
        let total_elements: usize = shape.iter().product();
        if total_elements != self.length {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.length],
                actual: vec![total_elements],
            });
        }

        // For f16, use direct conversion
        if T::is_f16() {
            let data = self.to_vec();
            // Safety: We checked T::is_f16(), so T = f16
            let f16_data: &[f16] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f16, data.len())
            };
            return NeuralEngineBuffer::from_f16_slice(f16_data, shape);
        }

        // For f32, convert to f16 first
        let data = self.to_vec();
        let f16_data: Vec<f16> = data.iter().map(|x| f16::from_f32(x.to_f32())).collect();
        NeuralEngineBuffer::from_f16_slice(&f16_data, shape)
    }
}

// Pool methods for f16 only
impl MetalBuffer<f16> {
    /// Create a new uninitialized Metal buffer from pool (f16 only)
    pub fn new_uninit_pooled(pool: &BufferPool, length: usize) -> TensorResult<Self> {
        pool.allocate(length)
    }

    /// Create a new Metal buffer filled with zeros from pool (f16 only)
    pub fn zeros_pooled(pool: &BufferPool, length: usize) -> TensorResult<Self> {
        pool.allocate_zeros(length)
    }

    /// Create a new Metal buffer from slice using pool (f16 only)
    pub fn from_vec_pooled(pool: &BufferPool, data: &[f16]) -> TensorResult<Self> {
        let mut buffer = pool.allocate(data.len())?;
        buffer.write_from_slice(data)?;
        Ok(buffer)
    }
}

impl<T: FloatType> PartialEq for MetalBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> MTLDevice {
        MTLDevice::system_default().expect("No Metal device available")
    }

    #[test]
    fn test_create_from_slice() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let buffer = MetalBuffer::<f16>::from_slice(&device, &data).unwrap();
        assert_eq!(buffer.len(), 3);

        let read_data = buffer.to_vec();
        assert_eq!(read_data.len(), 3);
        assert_eq!(read_data[0], f16::from_f32(1.0));
        assert_eq!(read_data[1], f16::from_f32(2.0));
        assert_eq!(read_data[2], f16::from_f32(3.0));
    }

    #[test]
    fn test_zeros() {
        let device = get_test_device();
        let buffer = MetalBuffer::<f16>::zeros(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_ones() {
        let device = get_test_device();
        let buffer = MetalBuffer::<f16>::ones(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ONE));
    }

    #[test]
    fn test_write_from_slice() {
        let device = get_test_device();
        let mut buffer = MetalBuffer::<f16>::zeros(&device, 3).unwrap();

        let new_data = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        buffer.write_from_slice(&new_data).unwrap();

        let read_data = buffer.to_vec();
        assert_eq!(read_data, new_data);
    }

    #[test]
    fn test_metal_to_neural_engine_conversion() {
        let device = get_test_device();
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let shape = vec![2, 2];

        // Create Metal buffer
        let metal_buffer = MetalBuffer::<f16>::from_slice(&device, &data).unwrap();

        // Convert to Neural Engine
        let ne_buffer = metal_buffer.to_neural_engine(&shape).unwrap();

        // Verify data
        assert_eq!(ne_buffer.shape(), shape);
        assert_eq!(ne_buffer.count(), 4);

        let ne_data = ne_buffer.to_f16_vec();
        assert_eq!(ne_data.len(), 4);
        assert_eq!(ne_data[0].to_f32(), 1.0);
        assert_eq!(ne_data[3].to_f32(), 4.0);
    }

    #[test]
    fn test_metal_neural_engine_roundtrip() {
        let device = get_test_device();
        let original_data = vec![
            f16::from_f32(1.5),
            f16::from_f32(2.5),
            f16::from_f32(3.5),
        ];
        let shape = vec![3];

        // Metal -> Neural Engine -> Metal
        let metal1 = MetalBuffer::<f16>::from_slice(&device, &original_data).unwrap();
        let ne_buffer = metal1.to_neural_engine(&shape).unwrap();
        let metal2 = ne_buffer.to_metal_buffer(&device).unwrap();

        // Verify roundtrip preserves data
        let result = metal2.to_vec();
        assert_eq!(result.len(), original_data.len());
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val.to_f32(), original_data[i].to_f32());
        }
    }
}
