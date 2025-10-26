//! Shared buffer for zero-copy conversion between Metal and Neural Engine

use crate::device::{MetalBuffer, NeuralEngineBuffer};
use crate::error::{TensorError, TensorResult};
use half::f16;
use metal::{Buffer as MTLBuffer, MTLResourceOptions};
use objc2::rc::Retained;
use objc2::ClassType;
use objc2_core_ml::MLMultiArray;
use objc2_foundation::NSNumber;

/// A buffer that can be accessed by both Metal GPU and Neural Engine without copying data.
///
/// This structure holds both an MLMultiArray (for Neural Engine) and a Metal buffer
/// that points to the same underlying memory, enabling zero-copy data sharing.
pub struct SharedBuffer {
    /// The MLMultiArray that owns the memory
    ml_array: Retained<MLMultiArray>,

    /// Metal buffer created with bytesNoCopy, pointing to ml_array's data
    metal_buffer: MTLBuffer,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Total number of elements
    count: usize,
}

impl SharedBuffer {
    /// Create a new SharedBuffer from f16 data
    ///
    /// The data is copied into an MLMultiArray, and then a Metal buffer is created
    /// that points to the same memory without copying.
    #[allow(deprecated)]
    pub fn from_f16_slice(
        metal_device: &metal::DeviceRef,
        data: &[f16],
        shape: Vec<usize>,
    ) -> TensorResult<Self> {
        // Validate shape
        let total_elements: usize = shape.iter().product();
        if total_elements != data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![total_elements],
                actual: vec![data.len()],
            });
        }

        // Create MLMultiArray
        let ns_shape: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|&dim| NSNumber::new_usize(dim))
            .collect();

        let ns_shape_array =
            objc2_foundation::NSArray::from_vec(ns_shape);

        // Float16 = 65552 in CoreML
        let data_type = unsafe { std::mem::transmute(65552i64) };

        let ml_array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &ns_shape_array,
                data_type,
            )
        }
        .map_err(|_| TensorError::InvalidOperation("Failed to create MLMultiArray".to_string()))?;

        // Copy data into MLMultiArray
        unsafe {
            let dest_ptr = ml_array.dataPointer().as_ptr() as *mut f16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest_ptr, data.len());
        }

        // Create Metal buffer pointing to the same memory (zero-copy)
        let metal_buffer = unsafe {
            let ptr = ml_array.dataPointer().as_ptr();
            let byte_length = (total_elements * std::mem::size_of::<f16>()) as u64;

            // Use MTLResourceStorageModeShared for CPU-GPU shared memory
            metal_device.new_buffer_with_bytes_no_copy(
                ptr,
                byte_length,
                MTLResourceOptions::StorageModeShared,
                None, // No deallocator - MLMultiArray owns the memory
            )
        };

        Ok(Self {
            ml_array,
            metal_buffer,
            shape,
            count: total_elements,
        })
    }

    /// Get a reference to the MLMultiArray (for Neural Engine operations)
    pub fn ml_array(&self) -> &Retained<MLMultiArray> {
        &self.ml_array
    }

    /// Get a reference to the Metal buffer (for GPU operations)
    pub fn metal_buffer(&self) -> &MTLBuffer {
        &self.metal_buffer
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the element count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Convert to f16 vector (reads from the shared memory)
    #[allow(deprecated)]
    pub fn to_f16_vec(&self) -> Vec<f16> {
        let mut result = vec![f16::ZERO; self.count];

        unsafe {
            let src_ptr = self.ml_array.dataPointer().as_ptr() as *const f16;
            std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr(), self.count);
        }

        result
    }

    /// Create a MetalBuffer view (zero-copy)
    pub fn as_metal_buffer(&self) -> MetalBuffer<half::f16> {
        use std::sync::Arc;
        use std::marker::PhantomData;
        MetalBuffer {
            buffer: Arc::new(self.metal_buffer.clone()),
            length: self.count,
            _phantom: PhantomData,
        }
    }

    /// Create a NeuralEngineBuffer view (zero-copy)
    pub fn as_neural_engine_buffer(&self) -> NeuralEngineBuffer {
        NeuralEngineBuffer {
            array: self.ml_array.clone(),
        }
    }

    /// Write f16 data to the shared buffer
    #[allow(deprecated)]
    pub fn write_from_slice(&mut self, data: &[f16]) -> TensorResult<()> {
        if data.len() != self.count {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.count],
                actual: vec![data.len()],
            });
        }

        unsafe {
            let dest_ptr = self.ml_array.dataPointer().as_ptr() as *mut f16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest_ptr, data.len());
        }

        Ok(())
    }

    /// Get a mutable pointer to the shared memory
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    /// from both Metal and Neural Engine.
    #[allow(deprecated)]
    pub unsafe fn data_ptr_mut(&self) -> *mut f16 {
        self.ml_array.dataPointer().as_ptr() as *mut f16
    }

    /// Get a const pointer to the shared memory
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    #[allow(deprecated)]
    pub unsafe fn data_ptr(&self) -> *const f16 {
        self.ml_array.dataPointer().as_ptr() as *const f16
    }
}

// SharedBuffer is Send because:
// - MLMultiArray owns the memory
// - MTLBuffer is just a reference to that memory
// - Both are thread-safe when properly synchronized
unsafe impl Send for SharedBuffer {}

// SharedBuffer is Sync because access to the underlying memory
// is managed through proper synchronization (Metal command buffers, etc.)
unsafe impl Sync for SharedBuffer {}

impl Clone for SharedBuffer {
    fn clone(&self) -> Self {
        // Create a new SharedBuffer with copied data
        let data = self.to_f16_vec();

        // Get Metal device from the existing buffer
        let metal_device = self.metal_buffer.device();

        Self::from_f16_slice(metal_device, &data, self.shape.clone())
            .expect("Failed to clone SharedBuffer")
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
    fn test_shared_buffer_creation() {
        let device = get_test_device();
        let metal_device = device.metal_device();

        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];

        let shared = SharedBuffer::from_f16_slice(metal_device, &data, vec![2, 2]).unwrap();

        assert_eq!(shared.shape(), &[2, 2]);
        assert_eq!(shared.count(), 4);
        assert_eq!(shared.to_f16_vec(), data);
    }

    #[test]
    fn test_zero_copy_metal_neural_engine() {
        let device = get_test_device();
        let metal_device = device.metal_device();

        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let mut shared = SharedBuffer::from_f16_slice(metal_device, &data, vec![3]).unwrap();

        // Modify through shared buffer
        let new_data = vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)];
        shared.write_from_slice(&new_data).unwrap();

        // Verify both views see the same data (zero-copy)
        assert_eq!(shared.to_f16_vec(), new_data);

        // Verify Metal buffer points to the same data
        let metal_view = shared.as_metal_buffer();
        assert_eq!(metal_view.to_vec(), new_data);
    }

    #[test]
    fn test_shared_buffer_clone() {
        let device = get_test_device();
        let metal_device = device.metal_device();

        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let shared = SharedBuffer::from_f16_slice(metal_device, &data, vec![2]).unwrap();

        let cloned = shared.clone();
        assert_eq!(cloned.to_f16_vec(), data);
        assert_eq!(cloned.shape(), shared.shape());
    }
}
