use crate::tensor::FloatType;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use super::prelude::*;
use crate::autograd::GradientFunction;
use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};

/// Add演算の勾配関数
///
/// c = a + b の場合:
/// ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * 1 = grad_output
/// ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * 1 = grad_output
pub struct AddBackward<T: FloatType> {
    a_shape: TensorShape,
    b_shape: TensorShape,
}

impl<T: FloatType> AddBackward<T> {
    pub fn new(a_shape: TensorShape, b_shape: TensorShape) -> Self {
        Self { a_shape, b_shape }
    }
}

impl<T: FloatType> GradientFunction for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        // 加算の勾配は単純にgrad_outputを両方の入力に伝播
        let grad_a = grad_output.clone();
        let grad_b = grad_output.clone();

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
    fn test_add_backward_same_shape() {
        let device = get_test_device();
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let a_shape = TensorShape::new(vec![2, 2]);
        let b_shape = TensorShape::new(vec![2, 2]);

        let backward = AddBackward::new(a_shape, b_shape);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), grad_output.to_vec());
        assert_eq!(grads[1].to_vec(), grad_output.to_vec());
    }

    #[test]
    fn test_add_backward_broadcast_scalar() {
        let device = get_test_device();
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let a_shape = TensorShape::new(vec![2, 2]);
        let b_shape = TensorShape::new(vec![1]); // スカラー

        let backward = AddBackward::new(a_shape, b_shape);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].dims(), &[2, 2]);
        assert_eq!(grads[1].dims(), &[1]);
        // grad_b = sum of all elements = 1 + 2 + 3 + 4 = 10
        assert_eq!(grads[1].to_vec()[0], f16::from_f32(10.0));
    }
}
