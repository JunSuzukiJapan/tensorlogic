//! Dropout operations

use crate::tensor::Tensor;
use crate::TensorResult;
use half::f16;
use rand::Rng;

impl Tensor {
    /// Dropout regularization
    ///
    /// Randomly zeros out elements with probability `p` during training.
    /// Scales remaining elements by 1/(1-p) to maintain expected value.
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 to 1.0)
    /// * `training` - If false, returns input unchanged (inference mode)
    ///
    /// # Formula (training mode)
    /// y_i = {
    ///   0           with probability p
    ///   x_i/(1-p)   with probability (1-p)
    /// }
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data: Vec<f16> = (0..640).map(|i| f16::from_f32(i as f32)).collect();
    /// let x = Tensor::from_vec(data, vec![10, 64])?;
    /// let dropped = x.dropout(0.5, true)?;  // 50% dropout during training
    /// let inference = x.dropout(0.5, false)?;  // No dropout during inference
    /// # Ok(())
    /// # }
    /// ```
    pub fn dropout(&self, p: f32, training: bool) -> TensorResult<Tensor> {
        // Validation
        if p < 0.0 || p > 1.0 {
            return Err(crate::error::TensorError::InvalidOperation(
                format!("dropout probability must be in [0, 1], got {}", p)
            ));
        }

        // Inference mode: return input unchanged
        if !training || p == 0.0 {
            return Ok(self.clone());
        }

        // Training mode with p > 0
        self.dropout_cpu(p)
    }

    fn dropout_cpu(&self, p: f32) -> TensorResult<Tensor> {
        let input_data = self.to_vec();
        let size = input_data.len();

        let mut rng = rand::rng();
        let scale = 1.0 / (1.0 - p);

        let output_data: Vec<f16> = input_data
            .iter()
            .map(|&val| {
                let random: f32 = rng.gen();  // Random number in [0, 1)
                if random < p {
                    f16::ZERO  // Drop
                } else {
                    f16::from_f32(val.to_f32() * scale)  // Keep and scale
                }
            })
            .collect();

        Tensor::from_vec(output_data, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_inference_mode() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0),
                f16::from_f32(3.0), f16::from_f32(4.0),
            ],
            vec![2, 2],
        ).unwrap();

        // Inference mode: should return unchanged
        let result = input.dropout(0.5, false).unwrap();
        let input_data = input.to_vec();
        let result_data = result.to_vec();

        for i in 0..input_data.len() {
            assert_eq!(input_data[i], result_data[i]);
        }
    }

    #[test]
    fn test_dropout_zero_probability() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0),
                f16::from_f32(3.0), f16::from_f32(4.0),
            ],
            vec![2, 2],
        ).unwrap();

        // p=0.0: should return unchanged
        let result = input.dropout(0.0, true).unwrap();
        let input_data = input.to_vec();
        let result_data = result.to_vec();

        for i in 0..input_data.len() {
            assert_eq!(input_data[i], result_data[i]);
        }
    }

    #[test]
    fn test_dropout_training_mode() {
        let input = Tensor::from_vec(
            vec![f16::from_f32(1.0); 100],
            vec![100],
        ).unwrap();

        // Training mode with p=0.5
        let result = input.dropout(0.5, true).unwrap();
        let result_data = result.to_vec();

        // Count zeros (dropped elements)
        let zero_count = result_data.iter().filter(|&&x| x == f16::ZERO).count();

        // Count non-zeros (kept elements)
        let non_zero_count = result_data.iter().filter(|&&x| x != f16::ZERO).count();

        // Should have roughly 50% zeros and 50% non-zeros
        // Allow some variance due to randomness (30% to 70%)
        assert!(zero_count >= 30 && zero_count <= 70);
        assert!(non_zero_count >= 30 && non_zero_count <= 70);

        // Non-zero values should be scaled by 1/(1-p) = 2.0
        for &val in &result_data {
            if val != f16::ZERO {
                // Should be approximately 2.0 (1.0 * scale)
                assert!((val.to_f32() - 2.0).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_dropout_expected_value() {
        // Test that expected value is preserved
        let input = Tensor::from_vec(
            vec![f16::from_f32(1.0); 1000],
            vec![1000],
        ).unwrap();

        // Apply dropout multiple times and check average
        let mut sum = 0.0f32;
        let trials = 10;

        for _ in 0..trials {
            let result = input.dropout(0.5, true).unwrap();
            let data = result.to_vec();
            let mean: f32 = data.iter().map(|x| x.to_f32()).sum::<f32>() / data.len() as f32;
            sum += mean;
        }

        let avg_mean = sum / trials as f32;

        // Expected value should be close to 1.0 (original value)
        assert!((avg_mean - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_dropout_invalid_probability() {
        let input = Tensor::from_vec(
            vec![f16::ONE; 10],
            vec![10],
        ).unwrap();

        // p < 0 should error
        assert!(input.dropout(-0.1, true).is_err());

        // p > 1 should error
        assert!(input.dropout(1.1, true).is_err());
    }
}
