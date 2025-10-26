use crate::tensor::FloatType;
use crate::autograd::GradientFunction;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;

pub struct LogBackward<T: FloatType> {
    input: Tensor<T>,
}

impl<T: FloatType> LogBackward<T> {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl<T: FloatType> GradientFunction for LogBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl<T: FloatType> LogBackward<T> {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let input_buf = self.input.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "log_backward_f16",
            grad_output,
            &[input_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_out = grad_output.to_vec();
        let input = self.input.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(input.iter())
            .map(|(g, x)| *g / *x)
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => Tensor::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
