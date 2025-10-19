//! Metal buffer management for f16 data

use crate::error::{TensorError, TensorResult};
use half::f16;
use metal::{Buffer, Device as MTLDevice};
use std::sync::Arc;

/// Metal buffer wrapper for f16 data
#[derive(Debug, Clone)]
pub struct MetalBuffer {
    buffer: Arc<Buffer>,
    length: usize, // number of f16 elements
}

impl MetalBuffer {
    /// Create a new Metal buffer from f16 slice
    pub fn from_f16_slice(device: &MTLDevice, data: &[f16]) -> TensorResult<Self> {
        let byte_length = data.len() * std::mem::size_of::<f16>();

        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length: data.len(),
        })
    }

    /// Create a new uninitialized Metal buffer
    pub fn new_uninit(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * std::mem::size_of::<f16>();

        let buffer = device.new_buffer(
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length,
        })
    }

    /// Create a new Metal buffer filled with zeros
    pub fn zeros(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let zeros = vec![f16::ZERO; length];
        Self::from_f16_slice(device, &zeros)
    }

    /// Create a new Metal buffer filled with ones
    pub fn ones(device: &MTLDevice, length: usize) -> TensorResult<Self> {
        let ones = vec![f16::ONE; length];
        Self::from_f16_slice(device, &ones)
    }

    /// Get the buffer length (number of f16 elements)
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the buffer byte length
    pub fn byte_length(&self) -> usize {
        self.length * std::mem::size_of::<f16>()
    }

    /// Get the underlying Metal buffer
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Read data from buffer to Vec<f16>
    pub fn to_vec(&self) -> Vec<f16> {
        let ptr = self.buffer.contents() as *const f16;
        unsafe { std::slice::from_raw_parts(ptr, self.length).to_vec() }
    }

    /// Write data to buffer from slice
    pub fn write_from_slice(&mut self, data: &[f16]) -> TensorResult<()> {
        if data.len() != self.length {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.length],
                actual: vec![data.len()],
            });
        }

        let ptr = self.buffer.contents() as *mut f16;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, self.length);
        }

        Ok(())
    }

    /// Get a mutable pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents_mut(&self) -> *mut f16 {
        self.buffer.contents() as *mut f16
    }

    /// Get a const pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents(&self) -> *const f16 {
        self.buffer.contents() as *const f16
    }
}

impl PartialEq for MetalBuffer {
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

        let buffer = MetalBuffer::from_f16_slice(&device, &data).unwrap();
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
        let buffer = MetalBuffer::zeros(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_ones() {
        let device = get_test_device();
        let buffer = MetalBuffer::ones(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ONE));
    }

    #[test]
    fn test_write_from_slice() {
        let device = get_test_device();
        let mut buffer = MetalBuffer::zeros(&device, 3).unwrap();

        let new_data = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        buffer.write_from_slice(&new_data).unwrap();

        let read_data = buffer.to_vec();
        assert_eq!(read_data, new_data);
    }
}
