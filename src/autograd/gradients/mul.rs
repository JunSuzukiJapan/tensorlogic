use crate::tensor::FloatType;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use super::prelude::*;
use crate::autograd::GradientFunction;
use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};

/// Mul演算の勾配関数
///
/// c = a * b の場合:
/// ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * b
/// ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * a
pub struct MulBackward {
    a: Tensor<half::f16>,
    b: Tensor<half::f16>,
    a_shape: TensorShape,
    b_shape: TensorShape,
}

impl MulBackward {
    pub fn new(a: Tensor, b: Tensor<half::f16>) -> Self {
        let a_shape = a.shape().clone();
        let b_shape = b.shape().clone();
        Self {
            a,
            b,
            a_shape,
            b_shape,
        }
    }
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        // ∂L/∂a = grad_output * b
        let grad_a = grad_output.mul(&self.b)?;

        // ∂L/∂b = grad_output * a
        let grad_b = grad_output.mul(&self.a)?;

        // ブロードキャストされている場合は次元を縮約
        let grad_a = reduce_grad_for_broadcast(&grad_a, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_b, &self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use half::f16;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_mul_backward_same_shape() {
        let device = get_test_device();

        let a = <Tensor<half::f16>>::from_vec_metal(
            &device,
            vec![
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
                half::f16::from_f32(5.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let b = <Tensor<half::f16>>::from_vec_metal(
            &device,
            vec![
                half::f16::from_f32(6.0),
                half::f16::from_f32(7.0),
                half::f16::from_f32(8.0),
                half::f16::from_f32(9.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let grad_output = <Tensor<half::f16>>::from_vec_metal(
            &device,
            vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let backward = MulBackward::new(a.clone(), b.clone());
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);

        // grad_a = grad_output * b = [1,1,1,1] * [6,7,8,9] = [6,7,8,9]
        assert_eq!(grads[0].to_vec(), b.to_vec());

        // grad_b = grad_output * a = [1,1,1,1] * [2,3,4,5] = [2,3,4,5]
        assert_eq!(grads[1].to_vec(), a.to_vec());
    }
}
