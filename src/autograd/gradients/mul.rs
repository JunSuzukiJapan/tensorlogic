use crate::tensor::FloatType;
use crate::tensor::TensorAutograd;
use crate::autograd::gradients::reduce_grad_for_broadcast;
use std::marker::PhantomData;
use super::prelude::*;
use crate::autograd::GradientFunctionGeneric;
use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};

/// Mul演算の勾配関数
///
/// c = a * b の場合:
/// ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * b
/// ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * a
pub struct MulBackward<T: FloatType> {
    a: Tensor<T>,
    b: Tensor<T>,
    a_shape: TensorShape,
    b_shape: TensorShape,
    _phantom: PhantomData<T>,
}

impl<T: FloatType> MulBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        use crate::device::Device;

        let a_shape = a.shape().clone();
        let b_shape = b.shape().clone();

        // Ensure GPU operations complete before storing tensors
        if let Device::Metal(ref device) = a.device() {
            device.flush_if_needed().ok();
            device.wait_until_completed().ok();
        }
        if let Device::Metal(ref device) = b.device() {
            device.flush_if_needed().ok();
            device.wait_until_completed().ok();
        }

        Self {
            a,
            b,
            a_shape,
            b_shape,
            _phantom: PhantomData,
        }
    }
}

impl<T: FloatType> GradientFunctionGeneric<T> for MulBackward<T>
where
    Tensor<T>: TensorAutograd<T>,
{
    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> TensorResult<Vec<Tensor<T>>> {
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

        let a = <Tensor<half::f16>>::from_vec_gpu(
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

        let b = <Tensor<half::f16>>::from_vec_gpu(
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

        let grad_output = <Tensor<half::f16>>::from_vec_gpu(
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
        assert_eq!(grads[0].sync_and_read(), b.sync_and_read());

        // grad_b = grad_output * a = [1,1,1,1] * [2,3,4,5] = [2,3,4,5]
        assert_eq!(grads[1].sync_and_read(), a.sync_and_read());
    }
}
