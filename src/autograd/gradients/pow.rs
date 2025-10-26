use crate::tensor::FloatType;
use crate::autograd::GradientFunction;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::{Device, MetalBuffer};
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

pub struct PowBackward<T: FloatType> {
    input: Tensor<T>,
    exponent: f32,
}

impl<T: FloatType> PowBackward<T> {
    pub fn new(input: Tensor, exponent: f32) -> Self {
        Self { input, exponent }
    }
}

impl<T: FloatType> GradientFunction for PowBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl<T: FloatType> PowBackward<T> {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let input_buf = self.input.buffer().as_metal()?;

        let device = match grad_output.device() {
            Device::Metal(dev) => dev,
            _ => return Err(crate::error::TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            )),
        };

        let exponent_buf = MetalBuffer::from_f16_slice(
            device.metal_device(),
            &[f16::from_f32(self.exponent)],
        )?;

        super::metal_helper::execute_parametric_metal_gradient(
            "pow_backward_f16",
            grad_output,
            &[input_buf],
            &[&exponent_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_out = grad_output.to_vec();
        let input = self.input.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(input.iter())
            .map(|(g, x)| {
                let x_f32 = x.to_f32();
                let grad = g.to_f32() * self.exponent * x_f32.powf(self.exponent - 1.0);
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
