use crate::tensor::FloatType;
use crate::autograd::GradientFunction;
use super::prelude::*;
use crate::device::{Device, MetalBuffer};
use crate::error::TensorResult;
use crate::tensor::Tensor;

pub struct PowBackward {
    input: Tensor<half::f16>,
    exponent: f32,
}

impl PowBackward {
    pub fn new(input: Tensor<half::f16>, exponent: f32) -> Self {
        Self { input, exponent }
    }
}

impl GradientFunction for PowBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl PowBackward {
    fn backward_metal(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let input_buf = self.input.buffer().as_metal()?;

        let device = match grad_output.device() {
            Device::Metal(dev) => dev,
            _ => return Err(crate::error::TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            )),
        };

        let exponent_buf = MetalBuffer::<half::f16>::from_slice(
            device.metal_device(),
            &[half::f16::from_f32(self.exponent)],
        )?;

        super::metal_helper::execute_parametric_metal_gradient(
            "pow_backward_f16",
            grad_output,
            &[input_buf],
            &[&exponent_buf],
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor<half::f16>) -> TensorResult<Tensor<half::f16>> {
        let grad_out = grad_output.sync_and_read();
        let input = self.input.sync_and_read();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(input.iter())
            .map(|(g, x)| {
                let x_f32 = x.to_f32();
                let grad = g.to_f32() * self.exponent * x_f32.powf(self.exponent - 1.0);
                half::f16::from_f32(grad)
            })
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                <Tensor<half::f16>>::from_vec_gpu(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => <Tensor<half::f16>>::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
