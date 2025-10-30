//! Core Tensor type
//!
//! All tensor methods are now organized into separate traits:
//! - TensorCreation: Creation methods (new, from_vec, zeros, etc.)
//! - TensorAccessors: Accessor methods (shape, dims, buffer, etc.)
//! - TensorTransform: Transformation methods (reshape, flatten)
//! - TensorIO: I/O methods (to_cpu, to_metal, save, load)
//! - TensorAutograd: Automatic differentiation methods (backward, grad, etc.)

use crate::autograd::NodeId;
use crate::device::{BufferPool, Device};
use crate::tensor::{BufferHandle, FloatType, TensorShape, TensorIO};
use half::f16;
use std::marker::PhantomData;
use std::sync::Arc;

/// Tensor data structure with generic floating-point type
#[derive(Debug, Clone)]
pub struct Tensor<T: FloatType = f16> {
    /// Tensor shape
    pub(crate) shape: TensorShape,

    /// Strides for memory layout (row-major)
    pub(crate) strides: Vec<usize>,

    /// Data buffer (on GPU or CPU)
    pub(crate) buffer: BufferHandle<T>,

    /// Device location
    pub(crate) device: Device,

    /// Gradient (for auto-differentiation)
    pub(crate) grad: Option<Box<Tensor<T>>>,

    /// Whether gradient computation is required
    pub(crate) requires_grad: bool,

    /// Computation graph node ID (if part of a computation graph)
    pub(crate) grad_node: Option<NodeId>,

    /// Version counter for gradient accumulation tracking
    pub(crate) version: u64,

    /// Buffer pool reference for automatic buffer recycling (Metal only)
    /// Cloning BufferPool is cheap - it just clones Arc pointers to shared data
    pub(crate) buffer_pool: Option<BufferPool>,

    /// Phantom data to hold the type parameter
    pub(crate) _phantom: PhantomData<T>,
}

/// Automatic buffer recycling when tensor is dropped
impl<T: FloatType> Drop for Tensor<T> {
    fn drop(&mut self) {
        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            if let BufferHandle::Metal(metal_buffer) = &self.buffer {
                let has_pool = self.buffer_pool.is_some();
                eprintln!("[Drop] Tensor drop: size={}, has_pool={}", metal_buffer.len(), has_pool);
            }
        }

        // Only recycle Metal buffers that have a buffer pool
        // Both f16 and f32 buffers can be recycled (BufferPool supports both)
        if T::is_f16() {
            if let (BufferHandle::Metal(metal_buffer), Some(pool)) =
                (&self.buffer, &self.buffer_pool)
            {
                // Only recycle if this is the last reference to the buffer
                // Arc::strong_count returns the number of strong references
                let ref_count = Arc::strong_count(&metal_buffer.buffer);

                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!("[Drop] ref_count={}", ref_count);
                }

                if ref_count == 1 {
                    // Clone the buffer for recycling (the clone shares the Arc)
                    // Need to convert to MetalBuffer<f16> for recycling
                    if let BufferHandle::Metal(mb) = &self.buffer {
                        // Safety: We checked T::is_f16(), so this is MetalBuffer<f16>
                        let f16_buffer: &crate::device::MetalBuffer<f16> = unsafe {
                            std::mem::transmute(mb)
                        };
                        let recycled = pool.recycle(f16_buffer.clone());
                        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                            eprintln!("[Drop] Buffer recycled: success={}", recycled);
                        }
                    }
                }
            }
        }
    }
}

impl<T: FloatType + PartialEq> PartialEq for Tensor<T> {
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
    use crate::device::MetalDevice;
    use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_create_from_vec() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let tensor = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![3]).unwrap();

        assert_eq!(tensor.dims(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_zeros() {
        let device = get_test_device();
        let tensor = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec();
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_ones() {
        let device = get_test_device();
        let tensor = Tensor::<f16>::ones(&device, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec();
        assert!(data.iter().all(|&x| x == f16::ONE));
    }

    #[test]
    fn test_reshape() {
        let device = get_test_device();
        let tensor = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(vec![6, 1]).unwrap();

        assert_eq!(reshaped.dims(), &[6, 1]);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_flatten() {
        let device = get_test_device();
        let tensor = Tensor::<f16>::zeros(&device, vec![2, 3, 4]).unwrap();
        let flat = tensor.flatten().unwrap();

        assert_eq!(flat.dims(), &[24]);
    }

    #[test]
    fn test_to_cpu() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let tensor = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2]).unwrap();

        let cpu_tensor = tensor.to_cpu().unwrap();
        assert!(cpu_tensor.buffer().is_cpu());
        assert_eq!(cpu_tensor.to_vec(), data);
    }

    #[test]
    fn test_save_load() {
        use crate::device::Device;
        use std::fs;

        let metal_device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let tensor = Tensor::<f16>::from_vec_gpu(&metal_device, data.clone(), vec![2, 2]).unwrap();

        let path = "/tmp/test_tensor.bin";
        tensor.save(path).unwrap();

        let device = Device::Metal(metal_device);
        let loaded = Tensor::<f16>::load(&device, path).unwrap();
        assert_eq!(loaded.dims(), tensor.dims());
        assert_eq!(loaded.to_vec(), data);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_f32_tensor_creation() {
        let device = get_test_device();
        let data = vec![1.0f32, 2.0f32, 3.0f32];

        let tensor = Tensor::<f32>::from_vec_gpu(&device, data.clone(), vec![3]).unwrap();

        assert_eq!(tensor.dims(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_f32_zeros() {
        let device = get_test_device();
        let tensor = Tensor::<f32>::zeros(&device, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec();
        assert!(data.iter().all(|&x| x == 0.0f32));
    }
}
