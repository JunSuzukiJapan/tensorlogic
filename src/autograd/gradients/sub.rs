use crate::tensor::FloatType;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use super::prelude::*;
use crate::autograd::GradientFunction;
use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};
use half::f16;

/// Sub演算の勾配関数
///
/// c = a - b の場合:
/// ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * 1 = grad_output
/// ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * (-1) = -grad_output
pub struct SubBackward {
    a_shape: TensorShape,
    b_shape: TensorShape,
}

impl SubBackward {
    pub fn new(a_shape: TensorShape, b_shape: TensorShape) -> Self {
        Self { a_shape, b_shape }
    }
}

impl GradientFunction for SubBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        // ∂L/∂a = grad_output
        let grad_a = grad_output.clone();

        // ∂L/∂b = -grad_output
        let grad_output_data = grad_output.to_vec();
        let neg_grad_data: Vec<half::f16> = grad_output_data.iter().map(|&x| -x).collect();
        let grad_b = Tensor<half::f16>::from_vec(neg_grad_data, grad_output.dims().to_vec())?;

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
    fn test_sub_backward_same_shape() {
        let device = get_test_device();
        let grad_output = Tensor<half::f16>::from_vec_metal(
            &device,
            vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let a_shape = TensorShape::new(vec![2, 2]);
        let b_shape = TensorShape::new(vec![2, 2]);

        let backward = SubBackward::new(a_shape, b_shape);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), grad_output.to_vec());

        // grad_b should be -grad_output
        let expected_grad_b = vec![
            half::f16::from_f32(-1.0),
            half::f16::from_f32(-2.0),
            half::f16::from_f32(-3.0),
            half::f16::from_f32(-4.0),
        ];
        assert_eq!(grads[1].to_vec(), expected_grad_b);
    }
}
