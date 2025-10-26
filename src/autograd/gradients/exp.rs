use crate::tensor::FloatType;
use crate::autograd::GradientFunction;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;

pub struct ExpBackward {
    output: Tensor<half::f16>, // exp(x)
}

impl ExpBackward {
    pub fn new(output: Tensor<half::f16>) -> Self {
        Self { output }
    }
}

impl GradientFunction for ExpBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.output.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl ExpBackward {
    fn backward_metal(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let output_buf = self.output.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "exp_backward_f16",
            grad_output,
            &[output_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let grad_out = grad_output.to_vec();
        let output = self.output.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(output.iter())
            .map(|(g, o)| *g * *o)
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                Tensor<half::f16>::from_vec_metal(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => Tensor<half::f16>::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
