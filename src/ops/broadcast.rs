//! Broadcasting operations for tensors

use crate::device::{Device, MetalBuffer};
use crate::tensor::FloatType;
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor, TensorShape};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Broadcast this tensor to a target shape
    pub fn broadcast_to(&self, target_shape: &TensorShape) -> TensorResult<Self> {
        // Check if broadcasting is needed
        if !self.shape().needs_broadcast(target_shape) {
            return Ok(self.clone());
        }

        // Check if broadcasting is valid
        if !self.shape().can_broadcast_to(target_shape) {
            return Err(TensorError::ShapeMismatch {
                expected: target_shape.dims().to_vec(),
                actual: self.shape().dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.broadcast_to_metal(target_shape),
            Device::CPU => self.broadcast_to_cpu(target_shape),
            Device::NeuralEngine => self.broadcast_to_cpu(target_shape), // Fallback
        }
    }

    /// CPU implementation of broadcast
    fn broadcast_to_cpu(&self, target_shape: &TensorShape) -> TensorResult<Self> {
        let input = self.to_vec();
        let input_dims = self.shape().dims();

        let target_numel = target_shape.numel();
        let mut output = vec![f16::ZERO; target_numel];

        // Compute strides for input and output
        let input_strides = self.shape().compute_strides();
        let target_strides = target_shape.compute_strides();

        // Align dimensions from the right
        let rank_diff = target_shape.rank() - self.shape().rank();

        for target_idx in 0..target_numel {
            // Compute multi-dimensional index for target
            let mut target_coords = vec![0; target_shape.rank()];
            let mut remaining = target_idx;
            for i in 0..target_shape.rank() {
                target_coords[i] = remaining / target_strides[i];
                remaining %= target_strides[i];
            }

            // Map to input index
            let mut input_idx = 0;
            for i in rank_diff..target_shape.rank() {
                let input_i = i - rank_diff;
                let coord = if input_dims[input_i] == 1 {
                    0
                } else {
                    target_coords[i]
                };
                input_idx += coord * input_strides[input_i];
            }

            output[target_idx] = input[input_idx];
        }

        Tensor::from_vec(output, target_shape.dims().to_vec())
    }

    /// Metal GPU implementation of broadcast
    fn broadcast_to_metal(&self, target_shape: &TensorShape) -> TensorResult<Self> {
        // For now, fallback to CPU
        // TODO: Implement efficient Metal kernel for broadcasting
        let cpu_result = self.to_cpu()?.broadcast_to_cpu(target_shape)?;

        // Convert back to Metal
        let device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError(
                "Expected Metal device".to_string(),
            )),
        };

        let metal_buf = MetalBuffer::from_f16_slice(
            device.metal_device(),
            &cpu_result.to_vec(),
        )?;

        Tensor::new(
            BufferHandle::Metal(metal_buf),
            target_shape.clone(),
            Device::Metal(device),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_broadcast_1d_to_2d() {
        // [3] -> [2, 3]
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ],
            vec![3],
        )
        .unwrap();

        let target_shape = TensorShape::new(vec![2, 3]);
        let b = a.broadcast_to(&target_shape).unwrap();

        let result = b.to_vec();
        assert_eq!(
            result,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ]
        );
    }

    #[test]
    fn test_broadcast_column_to_matrix() {
        // [2, 1] -> [2, 3]
        let a = Tensor::from_vec(
            vec![f16::from_f32(10.0), f16::from_f32(20.0)],
            vec![2, 1],
        )
        .unwrap();

        let target_shape = TensorShape::new(vec![2, 3]);
        let b = a.broadcast_to(&target_shape).unwrap();

        let result = b.to_vec();
        assert_eq!(
            result,
            vec![
                f16::from_f32(10.0),
                f16::from_f32(10.0),
                f16::from_f32(10.0),
                f16::from_f32(20.0),
                f16::from_f32(20.0),
                f16::from_f32(20.0),
            ]
        );
    }

    #[test]
    fn test_broadcast_scalar_to_vector() {
        // [1] -> [5]
        let a = Tensor::from_vec(vec![f16::from_f32(42.0)], vec![1]).unwrap();

        let target_shape = TensorShape::new(vec![5]);
        let b = a.broadcast_to(&target_shape).unwrap();

        let result = b.to_vec();
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&x| x == f16::from_f32(42.0)));
    }

    #[test]
    fn test_broadcast_error() {
        // [3] cannot broadcast to [2] (incompatible)
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ],
            vec![3],
        )
        .unwrap();

        let target_shape = TensorShape::new(vec![2]);
        assert!(a.broadcast_to(&target_shape).is_err());
    }

    #[test]
    fn test_broadcast_gpu() {
        let device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ],
            vec![3],
        )
        .unwrap();

        let target_shape = TensorShape::new(vec![2, 3]);
        let b = a.broadcast_to(&target_shape).unwrap();

        let result = b.to_vec();
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[3], f16::from_f32(1.0));
    }
}
