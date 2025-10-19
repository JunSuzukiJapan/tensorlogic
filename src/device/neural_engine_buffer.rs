//! Neural Engine buffer implementation using CoreML MLMultiArray

use crate::error::{TensorError, TensorResult};
use half::f16;
use objc2::rc::Retained;
use objc2::ClassType;
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};

/// Neural Engine buffer wrapper for MLMultiArray with f16 data
pub struct NeuralEngineBuffer {
    array: Retained<MLMultiArray>,
}

impl NeuralEngineBuffer {
    /// Create a new Neural Engine buffer from f16 slice
    #[allow(deprecated)]
    pub fn from_f16_slice(data: &[f16], shape: &[usize]) -> TensorResult<Self> {
        // Convert shape to NSArray<NSNumber>
        let ns_shape: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|&dim| NSNumber::numberWithUnsignedLong(dim as u64))
            .collect();

        let shape_array = NSArray::from_vec(ns_shape);

        // Create MLMultiArray with Float16 data type
        let ml_array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape_array,
                MLMultiArrayDataType::Float16,
            )
            .map_err(|_| TensorError::NeuralEngineError("Failed to create MLMultiArray".to_string()))?
        };

        // Copy f16 data to MLMultiArray
        unsafe {
            let dest_ptr = ml_array.dataPointer().as_ptr() as *mut f16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest_ptr, data.len());
        }

        Ok(Self { array: ml_array })
    }

    /// Get the underlying MLMultiArray
    pub fn ml_array(&self) -> &Retained<MLMultiArray> {
        &self.array
    }

    /// Convert to f16 vector
    #[allow(deprecated)]
    pub fn to_f16_vec(&self) -> Vec<f16> {
        let count = unsafe { self.array.count() } as usize;
        let mut result = vec![f16::ZERO; count];

        unsafe {
            let src_ptr = self.array.dataPointer().as_ptr() as *const f16;
            std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr(), count);
        }

        result
    }

    /// Get the shape of the buffer
    pub fn shape(&self) -> Vec<usize> {
        let ns_shape = unsafe { self.array.shape() };
        let count = ns_shape.len();

        (0..count)
            .map(|i| {
                ns_shape.get(i).unwrap().unsignedLongValue() as usize
            })
            .collect()
    }

    /// Get total element count
    pub fn count(&self) -> usize {
        (unsafe { self.array.count() }) as usize
    }
}

impl Clone for NeuralEngineBuffer {
    #[allow(deprecated)]
    fn clone(&self) -> Self {
        // Create new MLMultiArray with same shape and data type
        let shape = unsafe { self.array.shape() };
        let data_type = unsafe { self.array.dataType() };

        let new_array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape,
                data_type,
            )
            .unwrap_or_else(|_| panic!("Failed to clone MLMultiArray"))
        };

        // Copy data
        let count = unsafe { self.array.count() } as usize;
        unsafe {
            let src_ptr = self.array.dataPointer().as_ptr() as *const u8;
            let dest_ptr = new_array.dataPointer().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, count * 2); // f16 = 2 bytes
        }

        Self { array: new_array }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_neural_engine_buffer() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let shape = vec![3];

        let buffer = NeuralEngineBuffer::from_f16_slice(&data, &shape).unwrap();
        assert_eq!(buffer.count(), 3);
        assert_eq!(buffer.shape(), vec![3]);
    }

    #[test]
    fn test_neural_engine_buffer_roundtrip() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let shape = vec![2, 2];

        let buffer = NeuralEngineBuffer::from_f16_slice(&data, &shape).unwrap();
        let result = buffer.to_f16_vec();

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[3].to_f32(), 4.0);
    }

    #[test]
    fn test_neural_engine_buffer_clone() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let shape = vec![2];

        let buffer = NeuralEngineBuffer::from_f16_slice(&data, &shape).unwrap();
        let cloned = buffer.clone();

        assert_eq!(cloned.count(), buffer.count());
        assert_eq!(cloned.shape(), buffer.shape());
        assert_eq!(cloned.to_f16_vec(), buffer.to_f16_vec());
    }
}
