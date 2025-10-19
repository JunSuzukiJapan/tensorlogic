use crate::autograd::GradientFunction;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

/// ReLU演算の勾配関数
///
/// y = ReLU(x) = max(0, x) の場合:
/// ∂y/∂x = 1 if x > 0, else 0
pub struct ReLUBackward {
    input: Tensor,
}

impl ReLUBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradientFunction for ReLUBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_output_data = grad_output.to_vec();
        let input_data = self.input.to_vec();

        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(input_data.iter())
            .map(|(&grad_out, &input_val)| {
                if input_val > f16::ZERO {
                    grad_out
                } else {
                    f16::ZERO
                }
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
    fn test_relu_backward() {
        let device = get_test_device();

        // input = [-1.0, 0.0, 1.0, 2.0]
        let input = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        // grad_output = [1.0, 1.0, 1.0, 1.0]
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                f16::from_f32(1.0),
            ],
            vec![4],
        )
        .unwrap();

        let backward = ReLUBackward::new(input);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 1);

        // grad_input = [0.0, 0.0, 1.0, 1.0] (mask where input > 0)
        let expected = vec![
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];
        assert_eq!(grads[0].to_vec(), expected);
    }
}
