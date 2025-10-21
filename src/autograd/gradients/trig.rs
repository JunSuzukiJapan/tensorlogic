use crate::autograd::GradientFunction;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

pub struct SinBackward {
    input: Tensor,
}

impl SinBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradientFunction for SinBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl SinBackward {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let input_buf = self.input.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "sin_backward_f16",
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
            .map(|(g, x)| {
                let grad = g.to_f32() * x.to_f32().cos();
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

pub struct CosBackward {
    input: Tensor,
}

impl CosBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradientFunction for CosBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl CosBackward {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let input_buf = self.input.buffer().as_metal()?;
        super::metal_helper::execute_simple_metal_gradient(
            "cos_backward_f16",
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
            .map(|(g, x)| {
                let grad = g.to_f32() * (-x.to_f32().sin());
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
