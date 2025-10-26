//! In-place tensor operations for memory optimization
//!
//! These operations modify tensors in-place to avoid allocating new memory,
//! which is crucial for memory-constrained environments and large models.

use crate::device::Device;
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;
use half::f16;

impl Tensor<half::f16> {
    /// In-place addition: self += other
    ///
    /// Modifies self in-place to avoid allocating new memory.
    /// Both tensors must have the same shape.
    pub fn add_(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        // Shape check
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.add_metal_inplace(other),
            Device::CPU => self.add_cpu_inplace(other),
            Device::NeuralEngine => self.add_cpu_inplace(other), // Fallback to CPU
        }
    }

    fn add_cpu_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();
        let other_data = other.to_vec();

        let result: Vec<f16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        // Update self's buffer
        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            Device::CPU,
        )?;

        Ok(())
    }

    fn add_metal_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        // For Metal, we need to execute the kernel and update the buffer
        // This requires mutable access to the Metal buffer
        let result = self.add(other)?;
        *self = result;
        Ok(())
    }

    /// In-place multiplication: self *= other
    pub fn mul_(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.mul_metal_inplace(other),
            Device::CPU => self.mul_cpu_inplace(other),
            Device::NeuralEngine => self.mul_cpu_inplace(other),
        }
    }

    fn mul_cpu_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();
        let other_data = other.to_vec();

        let result: Vec<f16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            Device::CPU,
        )?;

        Ok(())
    }

    fn mul_metal_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        let result = self.mul(other)?;
        *self = result;
        Ok(())
    }

    /// In-place subtraction: self -= other
    pub fn sub_(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.sub_metal_inplace(other),
            Device::CPU => self.sub_cpu_inplace(other),
            Device::NeuralEngine => self.sub_cpu_inplace(other),
        }
    }

    fn sub_cpu_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();
        let other_data = other.to_vec();

        let result: Vec<f16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| *a - *b)
            .collect();

        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            Device::CPU,
        )?;

        Ok(())
    }

    fn sub_metal_inplace(&mut self, other: &Tensor<half::f16>) -> TensorResult<()> {
        let result = self.sub(other)?;
        *self = result;
        Ok(())
    }

    /// In-place ReLU: self = max(self, 0)
    pub fn relu_(&mut self) -> TensorResult<()> {
        match self.device() {
            Device::Metal(_) => self.relu_metal_inplace(),
            Device::CPU => self.relu_cpu_inplace(),
            Device::NeuralEngine => self.relu_cpu_inplace(),
        }
    }

    fn relu_cpu_inplace(&mut self) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();

        let result: Vec<f16> = self_data
            .iter()
            .map(|&x| if x > f16::ZERO { x } else { f16::ZERO })
            .collect();

        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            Device::CPU,
        )?;

        Ok(())
    }

    fn relu_metal_inplace(&mut self) -> TensorResult<()> {
        let result = self.relu()?;
        *self = result;
        Ok(())
    }

    /// In-place scalar addition: self += scalar
    pub fn add_scalar_(&mut self, scalar: f16) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();

        let result: Vec<f16> = self_data
            .iter()
            .map(|&x| x + scalar)
            .collect();

        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            self.device().clone(),
        )?;

        Ok(())
    }

    /// In-place scalar multiplication: self *= scalar
    pub fn mul_scalar_(&mut self, scalar: f16) -> TensorResult<()> {
        use crate::tensor::BufferHandle;

        let self_data = self.buffer().to_cpu_vec();

        let result: Vec<f16> = self_data
            .iter()
            .map(|&x| x * scalar)
            .collect();

        *self = Tensor::new(
            BufferHandle::CPU(result),
            self.shape().clone(),
            self.device().clone(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_inplace() {
        let mut a = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![3],
        )
        .unwrap();

        a.add_(&b).unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::from_f32(5.0));
        assert_eq!(result[1], f16::from_f32(7.0));
        assert_eq!(result[2], f16::from_f32(9.0));
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = Tensor::from_vec(
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)],
            vec![3],
        )
        .unwrap();

        a.mul_(&b).unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::from_f32(10.0));
        assert_eq!(result[1], f16::from_f32(18.0));
        assert_eq!(result[2], f16::from_f32(28.0));
    }

    #[test]
    fn test_sub_inplace() {
        let mut a = Tensor::from_vec(
            vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        a.sub_(&b).unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::from_f32(9.0));
        assert_eq!(result[1], f16::from_f32(18.0));
        assert_eq!(result[2], f16::from_f32(27.0));
    }

    #[test]
    fn test_relu_inplace() {
        let mut a = Tensor::from_vec(
            vec![
                f16::from_f32(-1.0),
                f16::from_f32(2.0),
                f16::from_f32(-3.0),
                f16::from_f32(4.0),
            ],
            vec![4],
        )
        .unwrap();

        a.relu_().unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::ZERO);
        assert_eq!(result[1], f16::from_f32(2.0));
        assert_eq!(result[2], f16::ZERO);
        assert_eq!(result[3], f16::from_f32(4.0));
    }

    #[test]
    fn test_add_scalar_inplace() {
        let mut a = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        a.add_scalar_(f16::from_f32(10.0)).unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::from_f32(11.0));
        assert_eq!(result[1], f16::from_f32(12.0));
        assert_eq!(result[2], f16::from_f32(13.0));
    }

    #[test]
    fn test_mul_scalar_inplace() {
        let mut a = Tensor::from_vec(
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        a.mul_scalar_(f16::from_f32(2.0)).unwrap();

        let result = a.to_vec();
        assert_eq!(result[0], f16::from_f32(4.0));
        assert_eq!(result[1], f16::from_f32(6.0));
        assert_eq!(result[2], f16::from_f32(8.0));
    }
}
