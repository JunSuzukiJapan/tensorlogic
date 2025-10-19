//! Element-wise tensor operations (placeholders for Metal kernels)

use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;
use half::f16;

impl Tensor {
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> TensorResult<Self> {
        // Check shape compatibility
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        // TODO: Implement Metal kernel for add
        // For now, fallback to CPU
        self.add_cpu(other)
    }

    /// CPU fallback for addition
    fn add_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        // Keep result on same device as self
        match self.device() {
            crate::device::Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, result, self.dims().to_vec())
            }
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        // TODO: Implement Metal kernel
        self.sub_cpu(other)
    }

    fn sub_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        match self.device() {
            crate::device::Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, result, self.dims().to_vec())
            }
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        // TODO: Implement Metal kernel
        self.mul_cpu(other)
    }

    fn mul_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        match self.device() {
            crate::device::Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, result, self.dims().to_vec())
            }
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        // TODO: Implement Metal kernel
        self.div_cpu(other)
    }

    fn div_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let result: Vec<f16> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x / y)
            .collect();

        match self.device() {
            crate::device::Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, result, self.dims().to_vec())
            }
            _ => Tensor::from_vec(result, self.dims().to_vec()),
        }
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
    fn test_add() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![3],
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        let expected = vec![f16::from_f32(5.0), f16::from_f32(7.0), f16::from_f32(9.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_sub() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(5.0), f16::from_f32(7.0), f16::from_f32(9.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        let expected = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_mul() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)],
            vec![3],
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        let expected = vec![f16::from_f32(10.0), f16::from_f32(18.0), f16::from_f32(28.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_div() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(5.0)],
            vec![3],
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        let expected = vec![f16::from_f32(5.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_shape_mismatch() {
        let device = get_test_device();

        let a = Tensor::zeros(&device, vec![2, 3]).unwrap();
        let b = Tensor::zeros(&device, vec![3, 2]).unwrap();

        assert!(a.add(&b).is_err());
    }
}
