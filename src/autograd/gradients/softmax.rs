use crate::autograd::GradientFunction;
use super::prelude::*;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

/// Softmax演算の勾配関数
///
/// Softmax: y_i = exp(x_i) / Σ_j exp(x_j)
///
/// ∂y_i/∂x_j = y_i * (δ_ij - y_j)
/// ∂L/∂x_i = Σ_j (∂L/∂y_j * ∂y_j/∂x_i)
///         = grad_output_i * y_i - y_i * Σ_j (grad_output_j * y_j)
pub struct SoftmaxBackward {
    output: Tensor,
}

impl SoftmaxBackward {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_output_data = grad_output.to_vec();
        let output_data = self.output.to_vec();

        // Σ_j (grad_output_j * y_j)
        let sum_grad_y: f16 = grad_output_data
            .iter()
            .zip(output_data.iter())
            .map(|(&g, &y)| g * y)
            .fold(f16::ZERO, |acc, x| acc + x);

        // grad_input_i = grad_output_i * y_i - y_i * sum_grad_y
        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(output_data.iter())
            .map(|(&g_i, &y_i)| g_i * y_i - y_i * sum_grad_y)
            .collect();

        let grad_input = Tensor::from_vec(grad_input_data, grad_output.dims().to_vec())?;
        Ok(vec![grad_input])
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
    fn test_softmax_backward() {
        let device = get_test_device();

        // Softmax output (already computed): [0.1, 0.2, 0.7]
        // (これは入力 [1.0, 2.0, 3.0] のsoftmax出力の近似値)
        let output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(0.09003057),
                f16::from_f32(0.24472848),
                f16::from_f32(0.66524095),
            ],
            vec![3],
        )
        .unwrap();

        // grad_output = [1.0, 0.0, 0.0] (one-hot gradient)
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            vec![3],
        )
        .unwrap();

        let backward = SoftmaxBackward::new(output.clone());
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].dims(), &[3]);

        let grad_input = grads[0].to_vec();

        // Softmax勾配の性質：Σ_i grad_input_i = 0 (合計は0になる)
        let sum: f32 = grad_input.iter().map(|x| x.to_f32()).sum();
        assert!((sum).abs() < 0.01, "Sum of softmax gradient should be ~0, got {}", sum);

        // grad_input[0] は正（対応するgrad_outputが1のため）
        assert!(grad_input[0].to_f32() > 0.0);

        // 他の成分は負
        assert!(grad_input[1].to_f32() < 0.0);
        assert!(grad_input[2].to_f32() < 0.0);
    }
}
