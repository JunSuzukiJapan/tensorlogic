use crate::autograd::GradientFunction;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;

/// Backward for concatenation - splits gradient to match input shapes
pub struct ConcatBackward {
    input_shapes: Vec<Vec<usize>>,
    dim: usize,
}

impl ConcatBackward {
    pub fn new(input_shapes: Vec<Vec<usize>>, dim: usize) -> Self {
        Self { input_shapes, dim }
    }
}

impl GradientFunction for ConcatBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        // For simplicity, implement CPU version that splits the gradient
        let grad_data = grad_output.to_vec();
        let mut gradients = Vec::new();

        // Calculate strides
        let dims = grad_output.dims();
        let mut stride = 1;
        let mut strides = vec![0; dims.len()];
        for i in (0..dims.len()).rev() {
            strides[i] = stride;
            stride *= dims[i];
        }

        let mut offset_at_dim = 0;
        for input_shape in &self.input_shapes {
            let size_at_dim = input_shape[self.dim];
            let input_numel: usize = input_shape.iter().product();
            let mut grad_input_data = vec![half::f16::ZERO; input_numel];

            // Copy data for this slice
            for idx in 0..input_numel {
                // Convert flat index to multidimensional coordinates
                let mut coords = vec![0; input_shape.len()];
                let mut remaining = idx;
                for d in 0..input_shape.len() {
                    coords[d] = remaining / strides[d];
                    remaining %= strides[d];
                }

                // Adjust coordinate at concat dim
                coords[self.dim] += offset_at_dim;

                // Calculate flat index in grad_output
                let mut output_idx = 0;
                for d in 0..dims.len() {
                    output_idx += coords[d] * strides[d];
                }

                grad_input_data[idx] = grad_data[output_idx];
            }

            let grad_tensor = match grad_output.device() {
                Device::Metal(dev) => {
                    <Tensor<half::f16>>::from_vec_metal(dev, grad_input_data, input_shape.clone())?
                }
                _ => <Tensor<half::f16>>::from_vec(grad_input_data, input_shape.clone())?,
            };
            gradients.push(grad_tensor);

            offset_at_dim += size_at_dim;
        }

        Ok(gradients)
    }
}

/// Backward for transpose - just transpose the gradient back
pub struct TransposeBackward;

impl TransposeBackward {
    pub fn new() -> Self {
        Self
    }
}

impl GradientFunction for TransposeBackward {
    fn backward(&self, grad_output: &Tensor<half::f16>, _inputs: &[&Tensor<half::f16>]) -> TensorResult<Vec<Tensor<half::f16>>> {
        // Transpose is self-inverse, so just transpose the gradient
        let grad_input = grad_output.transpose()?;
        Ok(vec![grad_input])
    }
}
