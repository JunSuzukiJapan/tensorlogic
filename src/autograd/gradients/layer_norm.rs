use crate::tensor::FloatType;
use crate::autograd::GradientFunction;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::Device;
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;

/// Backward for layer normalization
/// Returns gradients for [input, weight, bias] (weight and bias can be None)
pub struct LayerNormBackward<T: FloatType> {
    input: Tensor<T>,
    normalized_shape: Vec<usize>,
    weight: Option<Tensor>,
    #[allow(dead_code)]
    mean: Tensor<T>,      // Saved from forward pass
    inv_std: Tensor<T>,   // Saved from forward pass
    normalized: Tensor<T>, // Saved normalized values
    #[allow(dead_code)]
    eps: f32,
}

impl<T: FloatType> LayerNormBackward<T> {
    pub fn new(
        input: Tensor<T>,
        normalized_shape: Vec<usize>,
        weight: Option<Tensor>,
        mean: Tensor<T>,
        inv_std: Tensor<T>,
        normalized: Tensor<T>,
        eps: f32,
    ) -> Self {
        Self {
            input,
            normalized_shape,
            weight,
            mean,
            inv_std,
            normalized,
            eps,
        }
    }
}

impl<T: FloatType> GradientFunction for LayerNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        // For now, implement CPU version only
        self.backward_cpu(grad_output)
    }
}

impl<T: FloatType> LayerNormBackward<T> {
    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Vec<Tensor<f16>>> {
        let grad_out = grad_output.to_vec();
        let normalized = self.normalized.to_vec();
        let inv_std = self.inv_std.to_vec();

        let normalized_size: usize = self.normalized_shape.iter().product();
        let batch_size = self.input.numel() / normalized_size;

        let mut grad_input = vec![f16::ZERO; self.input.numel()];
        let mut grad_weight = if self.weight.is_some() {
            Some(vec![f16::ZERO; normalized_size])
        } else {
            None
        };
        let mut grad_bias = if self.weight.is_some() {
            Some(vec![f16::ZERO; normalized_size])
        } else {
            None
        };

        let weight_vec = self.weight.as_ref().map(|w| w.to_vec());

        for batch_idx in 0..batch_size {
            let offset = batch_idx * normalized_size;
            let grad_out_slice = &grad_out[offset..offset + normalized_size];
            let norm_slice = &normalized[offset..offset + normalized_size];

            let inv_std_val = inv_std[batch_idx].to_f32();

            // Compute gradient statistics
            let mean_grad: f32 =
                grad_out_slice.iter().map(|g| g.to_f32()).sum::<f32>() / normalized_size as f32;

            let mean_grad_norm: f32 = grad_out_slice
                .iter()
                .zip(norm_slice.iter())
                .map(|(g, n)| g.to_f32() * n.to_f32())
                .sum::<f32>()
                / normalized_size as f32;

            // Compute input gradient
            for i in 0..normalized_size {
                let grad = grad_out_slice[i].to_f32();
                let norm = norm_slice[i].to_f32();

                let grad_normalized = if let Some(ref w) = weight_vec {
                    grad * w[i].to_f32()
                } else {
                    grad
                };

                // d_input = (grad - mean_grad - normalized * mean_grad_norm) * inv_std
                let d_input = (grad_normalized - mean_grad - norm * mean_grad_norm) * inv_std_val;
                grad_input[offset + i] = f16::from_f32(d_input);

                // Accumulate weight and bias gradients
                if let Some(ref mut gw) = grad_weight {
                    gw[i] = f16::from_f32(gw[i].to_f32() + grad * norm);
                }
                if let Some(ref mut gb) = grad_bias {
                    gb[i] = f16::from_f32(gb[i].to_f32() + grad);
                }
            }
        }

        let mut gradients = vec![];

        // Input gradient
        let grad_input_tensor = match self.input.device() {
            Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, grad_input, self.input.dims().to_vec())?
            }
            _ => Tensor::from_vec(grad_input, self.input.dims().to_vec())?,
        };
        gradients.push(grad_input_tensor);

        // Weight gradient (if weight exists)
        if let Some(gw) = grad_weight {
            let grad_weight_tensor = match self.input.device() {
                Device::Metal(dev) => {
                    Tensor::from_vec_metal(dev, gw, self.normalized_shape.clone())?
                }
                _ => Tensor::from_vec(gw, self.normalized_shape.clone())?,
            };
            gradients.push(grad_weight_tensor);
        }

        // Bias gradient (if bias exists)
        if let Some(gb) = grad_bias {
            let grad_bias_tensor = match self.input.device() {
                Device::Metal(dev) => {
                    Tensor::from_vec_metal(dev, gb, self.normalized_shape.clone())?
                }
                _ => Tensor::from_vec(gb, self.normalized_shape.clone())?,
            };
            gradients.push(grad_bias_tensor);
        }

        Ok(gradients)
    }
}
