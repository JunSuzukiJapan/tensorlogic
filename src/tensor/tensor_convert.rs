//! Type conversion operations for tensors

use crate::tensor::{Tensor, BufferHandle, TensorAccessors, TensorCreation};
use crate::error::{TensorError, TensorResult};
use crate::device::{Device, MetalBuffer};
use half::f16;

/// Trait for converting between tensor types
pub trait TensorConvert: Sized {
    /// Convert f16 tensor to f32 tensor
    fn to_f32(&self) -> TensorResult<Tensor<f32>>;

    /// Convert f32 tensor to f16 tensor
    fn to_f16(&self) -> TensorResult<Tensor<f16>>;
}

impl TensorConvert for Tensor<f16> {
    fn to_f32(&self) -> TensorResult<Tensor<f32>> {
        match self.device() {
            Device::Metal(device) => {
                // Get f16 buffer
                let f16_buf = self.buffer().as_metal()?;
                let f16_data = f16_buf.to_vec();

                // Convert to f32
                let f32_data: Vec<f32> = f16_data.iter()
                    .map(|&x| x.to_f32())
                    .collect();

                // Create f32 buffer with pooled memory
                let f32_buf = MetalBuffer::from_vec_pooled(&device, &f32_data)?;

                // Create new tensor
                Tensor::new(
                    BufferHandle::Metal(unsafe { std::mem::transmute(f32_buf) }),
                    self.shape().clone(),
                    Device::Metal(device.clone()),
                )
            }
            Device::CPU => {
                return Err(TensorError::InvalidOperation(
                    "CPU tensor conversion not yet implemented".to_string()
                ));
            }
            Device::NeuralEngine => {
                return Err(TensorError::InvalidOperation(
                    "NeuralEngine tensor conversion not yet implemented".to_string()
                ));
            }
        }
    }

    fn to_f16(&self) -> TensorResult<Tensor<f16>> {
        // Already f16, just clone
        Ok(self.clone())
    }
}

impl TensorConvert for Tensor<f32> {
    fn to_f32(&self) -> TensorResult<Tensor<f32>> {
        // Already f32, just clone
        Ok(self.clone())
    }

    fn to_f16(&self) -> TensorResult<Tensor<f16>> {
        match self.device() {
            Device::Metal(device) => {
                // Get f32 buffer
                let f32_buf = self.buffer().as_metal()?;
                let f32_data = f32_buf.to_vec();

                // Convert to f16
                let f16_data: Vec<f16> = f32_data.iter()
                    .map(|&x| f16::from_f32(x))
                    .collect();

                // Create f16 buffer with pooled memory
                let f16_buf = MetalBuffer::from_vec_pooled(&device, &f16_data)?;

                // Create new tensor
                Tensor::new(
                    BufferHandle::Metal(unsafe { std::mem::transmute(f16_buf) }),
                    self.shape().clone(),
                    Device::Metal(device.clone()),
                )
            }
            Device::CPU => {
                return Err(TensorError::InvalidOperation(
                    "CPU tensor conversion not yet implemented".to_string()
                ));
            }
            Device::NeuralEngine => {
                return Err(TensorError::InvalidOperation(
                    "NeuralEngine tensor conversion not yet implemented".to_string()
                ));
            }
        }
    }
}
