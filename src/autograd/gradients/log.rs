use crate::autograd::GradientFunction;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;

pub struct LogBackward {
    input: Tensor<half::f16>,
}

impl LogBackward {
    pub fn new(input: Tensor<half::f16>) -> Self {
        Self { input }
    }
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl LogBackward {
    fn backward_metal(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let input_buf = self.input.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "log_backward_f16",
            grad_output,
            &[input_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let grad_out = grad_output.sync_and_read();
        let input = self.input.sync_and_read();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(input.iter())
            .map(|(g, x)| *g / *x)
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                <Tensor<half::f16>>::from_vec_gpu(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => <Tensor<half::f16>>::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
