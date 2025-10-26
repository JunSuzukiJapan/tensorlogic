use crate::autograd::GradientFunction;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

pub struct SigmoidBackward<T: FloatType> {
    output: Tensor<T>, // σ(x)
}

impl<T: FloatType> SigmoidBackward<T> {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl<T: FloatType> GradientFunction for SigmoidBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.output.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl<T: FloatType> SigmoidBackward<T> {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let output_buf = self.output.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "sigmoid_backward_f16",
            grad_output,
            &[output_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_out = grad_output.to_vec();
        let output = self.output.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(output.iter())
            .map(|(g, sigmoid)| {
                let s = sigmoid.to_f32();
                let grad = g.to_f32() * s * (1.0 - s);
                f16::from_f32(grad)
            })
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => Tensor::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}

pub struct TanhBackward<T: FloatType> {
    output: Tensor<T>, // tanh(x)
}

impl<T: FloatType> TanhBackward<T> {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }
}

impl<T: FloatType> GradientFunction for TanhBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.output.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl<T: FloatType> TanhBackward<T> {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let output_buf = self.output.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "tanh_backward_f16",
            grad_output,
            &[output_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_out = grad_output.to_vec();
        let output = self.output.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(output.iter())
            .map(|(g, tanh_val)| {
                let t = tanh_val.to_f32();
                let grad = g.to_f32() * (1.0 - t * t);
                f16::from_f32(grad)
            })
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => Tensor::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
