//! Buffer handle for different device types

use crate::device::{MetalBuffer, NeuralEngineBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::FloatType;
use half::f16;

/// Handle to tensor data on different devices
#[derive(Debug, Clone)]
pub enum BufferHandle<T: FloatType> {
    /// Metal GPU buffer
    Metal(MetalBuffer<T>),

    /// Neural Engine buffer (CoreML MLMultiArray, f16 only)
    /// Neural Engine always uses f16 internally, conversion happens automatically
    NeuralEngine(NeuralEngineBuffer),

    /// CPU buffer - avoid if possible, for control flow only
    CPU(Vec<T>),
}

impl<T: FloatType> BufferHandle<T> {
    /// Get the number of elements in the buffer
    pub fn len(&self) -> usize {
        match self {
            BufferHandle::Metal(buf) => buf.len(),
            BufferHandle::NeuralEngine(buf) => buf.count(),
            BufferHandle::CPU(vec) => vec.len(),
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read data from buffer to CPU Vec<T>
    pub fn to_cpu_vec(&self) -> Vec<T> {
        match self {
            BufferHandle::Metal(buf) => buf.to_vec(),
            BufferHandle::NeuralEngine(buf) => {
                // Neural Engine always uses f16, convert to T if needed
                let f16_data = buf.to_f16_vec();
                if T::is_f16() {
                    // Safety: We checked T::is_f16(), so T = f16
                    unsafe {
                        std::slice::from_raw_parts(f16_data.as_ptr() as *const T, f16_data.len()).to_vec()
                    }
                } else {
                    // Convert f16 to T (e.g., f16 to f32)
                    f16_data.iter().map(|x| T::from_f32(x.to_f32())).collect()
                }
            }
            BufferHandle::CPU(vec) => vec.clone(),
        }
    }

    /// Get device type as string
    pub fn device_type(&self) -> &'static str {
        match self {
            BufferHandle::Metal(_) => "Metal",
            BufferHandle::NeuralEngine(_) => "NeuralEngine",
            BufferHandle::CPU(_) => "CPU",
        }
    }

    /// Check if buffer is on Metal
    pub fn is_metal(&self) -> bool {
        matches!(self, BufferHandle::Metal(_))
    }

    /// Check if buffer is on Neural Engine
    pub fn is_neural_engine(&self) -> bool {
        matches!(self, BufferHandle::NeuralEngine(_))
    }

    /// Check if buffer is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, BufferHandle::CPU(_))
    }

    /// Get reference to Metal buffer if available
    pub fn as_metal(&self) -> TensorResult<&MetalBuffer<T>> {
        match self {
            BufferHandle::Metal(buf) => Ok(buf),
            _ => Err(TensorError::DeviceConversionError(format!(
                "Buffer is not on Metal device (found: {})",
                self.device_type()
            ))),
        }
    }

    /// Get mutable reference to Metal buffer if available
    pub fn as_metal_mut(&mut self) -> TensorResult<&mut MetalBuffer<T>> {
        match self {
            BufferHandle::Metal(buf) => Ok(buf),
            _ => Err(TensorError::DeviceConversionError(format!(
                "Buffer is not on Metal device (found: {})",
                self.device_type()
            ))),
        }
    }

    /// Get reference to CPU vec if available
    pub fn as_cpu(&self) -> TensorResult<&Vec<T>> {
        match self {
            BufferHandle::CPU(vec) => Ok(vec),
            _ => Err(TensorError::DeviceConversionError(format!(
                "Buffer is not on CPU (found: {})",
                self.device_type()
            ))),
        }
    }

    /// Get reference to Neural Engine buffer if available
    pub fn as_neural_engine(&self) -> TensorResult<&NeuralEngineBuffer> {
        match self {
            BufferHandle::NeuralEngine(buf) => Ok(buf),
            _ => Err(TensorError::DeviceConversionError(format!(
                "Buffer is not on Neural Engine (found: {})",
                self.device_type()
            ))),
        }
    }

}

impl<T: FloatType + PartialEq> PartialEq for BufferHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BufferHandle::Metal(a), BufferHandle::Metal(b)) => a == b,
            (BufferHandle::NeuralEngine(a), BufferHandle::NeuralEngine(b)) => {
                // Compare by converting to CPU vectors
                a.to_f16_vec() == b.to_f16_vec()
            }
            (BufferHandle::CPU(a), BufferHandle::CPU(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::Device as MTLDevice;

    #[test]
    fn test_cpu_buffer() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let handle = BufferHandle::<f16>::CPU(data.clone());

        assert!(handle.is_cpu());
        assert!(!handle.is_metal());
        assert_eq!(handle.len(), 2);
        assert_eq!(handle.to_cpu_vec(), data);
    }

    #[test]
    fn test_metal_buffer() {
        let device = MTLDevice::system_default().unwrap();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let metal_buf = MetalBuffer::<f16>::from_f16_slice(&device, &data).unwrap();
        let handle = BufferHandle::<f16>::Metal(metal_buf);

        assert!(handle.is_metal());
        assert!(!handle.is_cpu());
        assert_eq!(handle.len(), 2);
    }
}
