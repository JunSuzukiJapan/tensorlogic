//! Reduction operations for tensors

use crate::device::Device;
use crate::error::{TensorError, TensorResult};
use crate::tensor::{Tensor, TensorShape};
use half::f16;

impl Tensor {
    /// Sum all elements in the tensor
    pub fn sum(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) | Device::NeuralEngine => {
                // Fallback to CPU for now
                self.to_cpu()?.sum_cpu()
            }
            Device::CPU => self.sum_cpu(),
        }
    }

    fn sum_cpu(&self) -> TensorResult<f16> {
        let data = self.to_vec();
        let mut sum = f16::ZERO;
        for &val in &data {
            sum += val;
        }
        Ok(sum)
    }

    /// Sum along a specific dimension
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        if dim >= self.shape().rank() {
            return Err(TensorError::InvalidDimension { dim });
        }

        match self.device() {
            Device::Metal(_) | Device::NeuralEngine => {
                self.to_cpu()?.sum_dim_cpu(dim, keepdim)
            }
            Device::CPU => self.sum_dim_cpu(dim, keepdim),
        }
    }

    fn sum_dim_cpu(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        let input = self.to_vec();
        let input_dims = self.shape().dims();
        let input_strides = self.shape().compute_strides();

        // Compute output shape
        let mut output_dims = input_dims.to_vec();
        if keepdim {
            output_dims[dim] = 1;
        } else {
            output_dims.remove(dim);
        }

        let output_shape = TensorShape::new(output_dims.clone());
        let output_numel = output_shape.numel();
        let mut output = vec![f16::ZERO; output_numel];

        let dim_size = input_dims[dim];

        // Iterate over output elements
        for out_idx in 0..output_numel {
            // Compute multi-dimensional index for output
            let mut out_coords = vec![0; output_dims.len()];
            let mut remaining = out_idx;
            let out_strides = output_shape.compute_strides();
            for i in 0..output_dims.len() {
                out_coords[i] = remaining / out_strides[i];
                remaining %= out_strides[i];
            }

            // Map to input coordinates (insert reduction dimension)
            let mut input_coords = if keepdim {
                out_coords.clone()
            } else {
                let mut coords = Vec::with_capacity(input_dims.len());
                for (i, &val) in out_coords.iter().enumerate() {
                    if i == dim {
                        coords.push(0); // Will be replaced in loop
                    }
                    coords.push(val);
                }
                if dim == input_dims.len() - 1 {
                    coords.push(0);
                }
                coords
            };

            // Sum over the reduction dimension
            let mut sum = f16::ZERO;
            for d in 0..dim_size {
                input_coords[dim] = d;

                // Convert to linear index
                let mut input_idx = 0;
                for (i, &coord) in input_coords.iter().enumerate() {
                    input_idx += coord * input_strides[i];
                }

                sum += input[input_idx];
            }

            output[out_idx] = sum;
        }

        Tensor::from_vec(output, output_dims)
    }

    /// Mean of all elements
    pub fn mean(&self) -> TensorResult<f16> {
        let sum = self.sum()?;
        let count = self.numel() as f32;
        Ok(f16::from_f32(sum.to_f32() / count))
    }

    /// Mean along a specific dimension
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> TensorResult<Self> {
        let sum_result = self.sum_dim(dim, keepdim)?;
        let dim_size = self.shape().dims()[dim] as f32;

        // Divide by dimension size
        let data = sum_result.to_vec();
        let mean_data: Vec<f16> = data
            .iter()
            .map(|&x| f16::from_f32(x.to_f32() / dim_size))
            .collect();

        Tensor::from_vec(mean_data, sum_result.shape().dims().to_vec())
    }

    /// Maximum value in the tensor
    pub fn max(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) | Device::NeuralEngine => self.to_cpu()?.max_cpu(),
            Device::CPU => self.max_cpu(),
        }
    }

    fn max_cpu(&self) -> TensorResult<f16> {
        let data = self.to_vec();
        if data.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Cannot compute max of empty tensor".to_string(),
            ));
        }

        let mut max_val = data[0];
        for &val in &data[1..] {
            if val > max_val {
                max_val = val;
            }
        }
        Ok(max_val)
    }

    /// Minimum value in the tensor
    pub fn min(&self) -> TensorResult<f16> {
        match self.device() {
            Device::Metal(_) | Device::NeuralEngine => self.to_cpu()?.min_cpu(),
            Device::CPU => self.min_cpu(),
        }
    }

    fn min_cpu(&self) -> TensorResult<f16> {
        let data = self.to_vec();
        if data.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }

        let mut min_val = data[0];
        for &val in &data[1..] {
            if val < min_val {
                min_val = val;
            }
        }
        Ok(min_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![4],
        )
        .unwrap();

        let sum = a.sum().unwrap();
        assert_eq!(sum, f16::from_f32(10.0));
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![4],
        )
        .unwrap();

        let mean = a.mean().unwrap();
        assert_eq!(mean, f16::from_f32(2.5));
    }

    #[test]
    fn test_max() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(5.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let max = a.max().unwrap();
        assert_eq!(max, f16::from_f32(5.0));
    }

    #[test]
    fn test_min() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(5.0),
                f16::from_f32(1.0),
                f16::from_f32(3.0),
                f16::from_f32(2.0),
            ],
            vec![4],
        )
        .unwrap();

        let min = a.min().unwrap();
        assert_eq!(min, f16::from_f32(1.0));
    }

    #[test]
    fn test_sum_dim() {
        // Matrix [2, 3]
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        // Sum along dim 0 (rows) -> [3]
        let sum0 = a.sum_dim(0, false).unwrap();
        assert_eq!(sum0.shape().dims(), &[3]);
        let result0 = sum0.to_vec();
        assert_eq!(result0[0], f16::from_f32(5.0)); // 1+4
        assert_eq!(result0[1], f16::from_f32(7.0)); // 2+5
        assert_eq!(result0[2], f16::from_f32(9.0)); // 3+6

        // Sum along dim 1 (columns) -> [2]
        let sum1 = a.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        let result1 = sum1.to_vec();
        assert_eq!(result1[0], f16::from_f32(6.0)); // 1+2+3
        assert_eq!(result1[1], f16::from_f32(15.0)); // 4+5+6
    }

    #[test]
    fn test_mean_dim() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        // Mean along dim 1
        let mean1 = a.mean_dim(1, false).unwrap();
        assert_eq!(mean1.shape().dims(), &[2]);
        let result = mean1.to_vec();
        assert_eq!(result[0], f16::from_f32(3.0)); // (2+4)/2
        assert_eq!(result[1], f16::from_f32(7.0)); // (6+8)/2
    }
}
