use crate::autograd::GradientFunction;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

/// GELU演算の勾配関数
///
/// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// ∂GELU/∂x = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * derivative_of_inner
pub struct GELUBackward {
    input: Tensor,
}

impl GELUBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradientFunction for GELUBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_output_data = grad_output.to_vec();
        let input_data = self.input.to_vec();

        let sqrt_2_over_pi = f16::from_f32((2.0_f32 / std::f32::consts::PI).sqrt());

        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(input_data.iter())
            .map(|(&grad_out, &x)| {
                let x_f32 = x.to_f32();
                let x3 = x_f32 * x_f32 * x_f32;
                let inner = sqrt_2_over_pi.to_f32() * (x_f32 + 0.044715 * x3);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;

                // ∂GELU/∂x = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * derivative_of_inner
                let derivative_of_inner =
                    sqrt_2_over_pi.to_f32() * (1.0 + 3.0 * 0.044715 * x_f32 * x_f32);
                let gelu_derivative =
                    0.5 * (1.0 + tanh_val) + 0.5 * x_f32 * sech2 * derivative_of_inner;

                f16::from_f32(grad_out.to_f32() * gelu_derivative)
            })
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
    fn test_gelu_backward_basic() {
        let device = get_test_device();

        // input = [0.0, 1.0]
        let input = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.0), f16::from_f32(1.0)],
            vec![2],
        )
        .unwrap();

        // grad_output = [1.0, 1.0]
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(1.0)],
            vec![2],
        )
        .unwrap();

        let backward = GELUBackward::new(input);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].dims(), &[2]);

        // 勾配が計算されていることを確認（具体的な値は数値微分で検証が望ましい）
        let grad_values = grads[0].to_vec();
        assert!(grad_values[0].to_f32() > 0.0); // x=0での勾配は正
        assert!(grad_values[1].to_f32() > 0.0); // x=1での勾配は正
    }
}
