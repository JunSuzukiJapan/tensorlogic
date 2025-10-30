use crate::tensor::FloatType;
use crate::tensor::TensorAutograd;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use super::prelude::*;
use crate::autograd::GradientFunctionGeneric;
use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};

/// Div演算の勾配関数
///
/// c = a / b の場合:
/// ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * (1/b) = grad_output / b
/// ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * (-a/b²) = -grad_output * a / b²
pub struct DivBackward<T: FloatType> {
    a: Tensor<T>,
    b: Tensor<T>,
    a_shape: TensorShape,
    b_shape: TensorShape,
    _phantom: PhantomData<T>,
}

impl<T: FloatType> DivBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        let a_shape = a.shape().clone();
        let b_shape = b.shape().clone();
        Self {
            a,
            b,
            a_shape,
            b_shape,
            _phantom: PhantomData,
        }
    }
}

impl<T: FloatType> GradientFunctionGeneric<T> for DivBackward<T>
where
    Tensor<T>: TensorAutograd<T>,
{
    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> TensorResult<Vec<Tensor<T>>> {
        // ∂L/∂a = grad_output / b
        let grad_a = grad_output.div(&self.b)?;

        // ∂L/∂b = -grad_output * a / b²
        let grad_output_mul_a = grad_output.mul(&self.a)?;
        let b_squared = self.b.mul(&self.b)?;
        let grad_b_positive = grad_output_mul_a.div(&b_squared)?;

        // 符号を反転 - ジェネリックで処理
        let grad_b_data = grad_b_positive.to_vec();
        let neg_grad_b_data: Vec<T> = grad_b_data.iter().map(|&x| T::zero() - x).collect();
        let grad_b = Tensor::<T>::from_vec(neg_grad_b_data, grad_b_positive.dims().to_vec())?;

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
    fn test_div_backward_simple() {
        let device = get_test_device();

        // a = [4.0, 6.0], b = [2.0, 3.0]
        // c = a / b = [2.0, 2.0]
        let a = <Tensor<half::f16>>::from_vec_gpu(
            &device,
            vec![half::f16::from_f32(4.0), half::f16::from_f32(6.0)],
            vec![2],
        )
        .unwrap();

        let b = <Tensor<half::f16>>::from_vec_gpu(
            &device,
            vec![half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
            vec![2],
        )
        .unwrap();

        let grad_output = <Tensor<half::f16>>::from_vec_gpu(
            &device,
            vec![half::f16::from_f32(1.0), half::f16::from_f32(1.0)],
            vec![2],
        )
        .unwrap();

        let backward = DivBackward::new(a.clone(), b.clone());
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);

        // grad_a = grad_output / b = [1.0, 1.0] / [2.0, 3.0] = [0.5, 0.333...]
        let grad_a_expected = vec![half::f16::from_f32(0.5), half::f16::from_f32(1.0 / 3.0)];
        let grad_a_actual = grads[0].to_vec();
        for (actual, expected) in grad_a_actual.iter().zip(grad_a_expected.iter()) {
            assert!((actual.to_f32() - expected.to_f32()).abs() < 0.01);
        }

        // grad_b = -grad_output * a / b² = -[1.0, 1.0] * [4.0, 6.0] / [4.0, 9.0]
        //        = -[4.0/4.0, 6.0/9.0] = -[1.0, 0.666...] = [-1.0, -0.666...]
        let grad_b_expected = vec![half::f16::from_f32(-1.0), half::f16::from_f32(-6.0 / 9.0)];
        let grad_b_actual = grads[1].to_vec();
        for (actual, expected) in grad_b_actual.iter().zip(grad_b_expected.iter()) {
            assert!((actual.to_f32() - expected.to_f32()).abs() < 0.01);
        }
    }
}
