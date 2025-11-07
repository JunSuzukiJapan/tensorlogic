use crate::tensor::FloatType;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use crate::autograd::GradientFunctionGeneric;
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
    _phantom: PhantomData<T>,
}

impl<T: FloatType> AddBackward<T> {
    pub fn new(a_shape: TensorShape, b_shape: TensorShape) -> Self {
        Self {
            a_shape,
            b_shape,
            _phantom: PhantomData,
        }
    }
}

impl<T: FloatType> GradientFunctionGeneric<T> for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> TensorResult<Vec<Tensor<T>>> {
        // 加算の勾配は単純にgrad_outputを両方の入力に伝播
        let grad_a = grad_output.clone();
        let grad_b = grad_output.clone();

        // ブロードキャストされている場合は次元を縮約
        let grad_a = reduce_grad_for_broadcast(&grad_a, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_b, &self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}
