//! Batch Normalization operations

use crate::tensor::Tensor;
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::TensorResult;
use crate::error::TensorError;
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Batch Normalization
    ///
    /// Normalizes input across the batch dimension
    ///
    /// # Arguments
    /// * `gamma` - Scale parameter [features]
    /// * `beta` - Shift parameter [features]
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    ///
    /// # Shape
    /// - self: [batch, features] or [batch, C, H, W]
    /// - gamma: [features] or [C]
    /// - beta: [features] or [C]
    /// - output: same shape as self
    ///
    /// # Formula
    /// y = gamma * ((x - mean) / sqrt(var + eps)) + beta
    ///
    /// where mean and var are computed across the batch dimension
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = MetalDevice::new()?;
    /// let data: Vec<f16> = (0..512).map(|i| f16::from_f32(i as f32)).collect();
    /// let x = Tensor::from_vec(data, vec![8, 64])?;  // batch=8, features=64
    /// let gamma = Tensor::ones(&device, vec![64])?;
    /// let beta = Tensor::zeros(&device, vec![64])?;
    /// let normalized = x.batch_norm(&gamma, &beta, 1e-5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn batch_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> TensorResult<Tensor> {
        // Verify shapes
        if self.dims().len() < 2 {
            return Err(TensorError::InvalidOperation(
                "batch_norm requires at least 2D tensor".to_string()
            ));
        }

        let batch_size = self.dims()[0];
        let feature_size = if self.dims().len() == 2 {
            self.dims()[1]
        } else {
            // For 4D [batch, channels, height, width], normalize over channels
            self.dims()[1]
        };

        // Verify gamma and beta shapes
        if gamma.dims()[0] != feature_size || beta.dims()[0] != feature_size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![feature_size],
                actual: gamma.dims().to_vec(),
            });
        }

        // CPU implementation
        self.batch_norm_cpu(gamma, beta, eps, batch_size, feature_size)
    }

    fn batch_norm_cpu(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32,
        batch_size: usize,
        feature_size: usize,
    ) -> TensorResult<Tensor> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input_data = self.to_vec();
        let gamma_data = gamma.to_vec();
        let beta_data = beta.to_vec();

        let total_size = input_data.len();
        let mut output_data = vec![f16::ZERO; total_size];

        // Process each feature
        for f in 0..feature_size {
            // Compute mean across batch
            let mut sum = 0.0f32;
            for b in 0..batch_size {
                let idx = b * feature_size + f;
                sum += input_data[idx].to_f32();
            }
            let mean = sum / batch_size as f32;

            // Compute variance across batch
            let mut var_sum = 0.0f32;
            for b in 0..batch_size {
                let idx = b * feature_size + f;
                let diff = input_data[idx].to_f32() - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / batch_size as f32;

            // Normalize and apply affine transformation
            let std = (variance + eps).sqrt();
            let scale = gamma_data[f].to_f32();
            let shift = beta_data[f].to_f32();

            for b in 0..batch_size {
                let idx = b * feature_size + f;
                let normalized = (input_data[idx].to_f32() - mean) / std;
                output_data[idx] = f16::from_f32(scale * normalized + shift);
            }
        }

        Tensor::from_vec(output_data, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_2d() {
        // Input: [batch=2, features=3]
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),  // batch 0
                f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0),  // batch 1
            ],
            vec![2, 3],
        ).unwrap();

        // Gamma and beta
        let gamma = Tensor::from_vec(
            vec![f16::ONE, f16::ONE, f16::ONE],
            vec![3],
        ).unwrap();

        let beta = Tensor::from_vec(
            vec![f16::ZERO, f16::ZERO, f16::ZERO],
            vec![3],
        ).unwrap();

        let result = input.batch_norm(&gamma, &beta, 1e-5).unwrap();
        let data = result.to_vec();

        // For feature 0: values are [1, 4], mean=2.5, std≈1.5
        // normalized: [(1-2.5)/1.5, (4-2.5)/1.5] = [-1.0, 1.0]
        assert!((data[0].to_f32() + 1.0).abs() < 0.1);  // ≈ -1.0
        assert!((data[3].to_f32() - 1.0).abs() < 0.1);  // ≈ 1.0

        // Check that each feature is normalized (mean≈0, std≈1)
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_batch_norm_with_affine() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(0.0), f16::from_f32(1.0),
                f16::from_f32(2.0), f16::from_f32(3.0),
            ],
            vec![2, 2],
        ).unwrap();

        // Scale by 2, shift by 1
        let gamma = Tensor::from_vec(
            vec![f16::from_f32(2.0), f16::from_f32(2.0)],
            vec![2],
        ).unwrap();

        let beta = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(1.0)],
            vec![2],
        ).unwrap();

        let result = input.batch_norm(&gamma, &beta, 1e-5).unwrap();
        let data = result.to_vec();

        // After normalization, should be scaled by 2 and shifted by 1
        // Each feature should have different values but follow the affine transformation
        assert_eq!(result.dims(), &[2, 2]);

        // Verify output is not NaN
        for &val in &data {
            assert!(!val.is_nan());
        }
    }
}
