//! Masking operations for attention mechanisms

use crate::tensor::Tensor;
use crate::tensor::FloatType;
use crate::TensorResult;
use crate::error::TensorError;
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Apply attention mask to attention scores
    ///
    /// Replaces masked positions with a large negative value (-10000.0)
    /// so they become ~0 after softmax
    ///
    /// # Arguments
    /// * `mask` - Boolean-like tensor where 0.0 means mask (ignore), 1.0 means keep
    ///
    /// # Shape
    /// - self: [batch, seq_len, seq_len] or [seq_len, seq_len]
    /// - mask: same shape as self
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Causal mask for autoregressive models
    /// let scores = Tensor::from_vec(vec![f16::from_f32(1.0), f16::from_f32(2.0),
    ///                                     f16::from_f32(3.0), f16::from_f32(4.0)], vec![2, 2])?;
    /// let mask = Tensor::from_vec(vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE], vec![2, 2])?;
    /// let masked = scores.apply_attention_mask(&mask)?;
    /// // masked[0,1] will be -10000.0 (masked out)
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_attention_mask(&self, mask: &Tensor<T>) -> TensorResult<Tensor> {
        // Verify shapes match
        if self.dims() != mask.dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: mask.dims().to_vec(),
            });
        }

        // Create large negative value for masked positions
        let mask_value = f16::from_f32(-10000.0);

        // For each element: if mask == 0, use mask_value, else use original value
        let self_data = self.to_vec();
        let mask_data = mask.to_vec();

        let result_data: Vec<f16> = self_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&val, &mask_val)| {
                if mask_val == f16::ZERO {
                    mask_value
                } else {
                    val
                }
            })
            .collect();

        Tensor::from_vec(result_data, self.dims().to_vec())
    }

    /// Create a causal mask for autoregressive attention
    ///
    /// Returns a lower triangular matrix of 1s and 0s
    /// Used in decoder self-attention to prevent attending to future positions
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Tensor of shape [seq_len, seq_len] where:
    /// - Upper triangle (including diagonal) = 1.0 (allow attention)
    /// - Lower triangle = 0.0 (mask out)
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mask = Tensor::causal_mask(3)?;
    /// // [[1, 0, 0],
    /// //  [1, 1, 0],
    /// //  [1, 1, 1]]
    /// # Ok(())
    /// # }
    /// ```
    pub fn causal_mask(seq_len: usize) -> TensorResult<Tensor> {
        let mut data = Vec::with_capacity(seq_len * seq_len);

        for i in 0..seq_len {
            for j in 0..seq_len {
                // Allow attention to positions <= current position
                if j <= i {
                    data.push(f16::ONE);
                } else {
                    data.push(f16::ZERO);
                }
            }
        }

        Tensor::from_vec(data, vec![seq_len, seq_len])
    }

    /// Create a padding mask
    ///
    /// Masks out padding tokens in a batch of sequences
    ///
    /// # Arguments
    /// * `lengths` - Actual length of each sequence in the batch
    /// * `max_len` - Maximum sequence length (padded length)
    ///
    /// # Returns
    /// Tensor of shape [batch_size, max_len] where:
    /// - 1.0 for real tokens
    /// - 0.0 for padding tokens
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mask = Tensor::padding_mask(&[2, 3], 4)?;
    /// // [[1, 1, 0, 0],  // first sequence has length 2
    /// //  [1, 1, 1, 0]]  // second sequence has length 3
    /// # Ok(())
    /// # }
    /// ```
    pub fn padding_mask(lengths: &[usize], max_len: usize) -> TensorResult<Tensor> {
        let batch_size = lengths.len();
        let mut data = Vec::with_capacity(batch_size * max_len);

        for &len in lengths {
            for pos in 0..max_len {
                if pos < len {
                    data.push(f16::ONE);
                } else {
                    data.push(f16::ZERO);
                }
            }
        }

        Tensor::from_vec(data, vec![batch_size, max_len])
    }

    /// Combine multiple masks (logical AND)
    ///
    /// Useful for combining causal mask + padding mask
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let causal = Tensor::causal_mask(4)?;
    /// // Create matching 4x4 padding mask (not 2x4)
    /// let padding_data = vec![f16::ONE; 16];  // All ones for simplicity
    /// let padding = Tensor::from_vec(padding_data, vec![4, 4])?;
    /// let combined = causal.combine_masks(&padding)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn combine_masks(&self, other: &Tensor<T>) -> TensorResult<Tensor> {
        if self.dims() != other.dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        let result_data: Vec<f16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| {
                // Logical AND: both must be non-zero
                if a != f16::ZERO && b != f16::ZERO {
                    f16::ONE
                } else {
                    f16::ZERO
                }
            })
            .collect();

        Tensor::from_vec(result_data, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_attention_mask() {
        let scores = Tensor::from_vec(
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0),
                f16::from_f32(3.0), f16::from_f32(4.0),
            ],
            vec![2, 2],
        ).unwrap();

        let mask = Tensor::from_vec(
            vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let result = scores.apply_attention_mask(&mask).unwrap();
        let data = result.to_vec();

        assert_eq!(data[0], f16::from_f32(1.0));
        assert_eq!(data[1], f16::from_f32(-10000.0)); // masked
        assert_eq!(data[2], f16::from_f32(3.0));
        assert_eq!(data[3], f16::from_f32(4.0));
    }

    #[test]
    fn test_causal_mask() {
        let mask = Tensor::causal_mask(3).unwrap();
        let data = mask.to_vec();

        // Expected: [[1, 0, 0],
        //            [1, 1, 0],
        //            [1, 1, 1]]
        assert_eq!(data[0], f16::ONE);  // [0,0]
        assert_eq!(data[1], f16::ZERO); // [0,1]
        assert_eq!(data[2], f16::ZERO); // [0,2]

        assert_eq!(data[3], f16::ONE);  // [1,0]
        assert_eq!(data[4], f16::ONE);  // [1,1]
        assert_eq!(data[5], f16::ZERO); // [1,2]

        assert_eq!(data[6], f16::ONE);  // [2,0]
        assert_eq!(data[7], f16::ONE);  // [2,1]
        assert_eq!(data[8], f16::ONE);  // [2,2]
    }

    #[test]
    fn test_padding_mask() {
        let mask = Tensor::padding_mask(&[2, 3], 4).unwrap();
        let data = mask.to_vec();

        // Expected: [[1, 1, 0, 0],
        //            [1, 1, 1, 0]]

        // First sequence (length 2)
        assert_eq!(data[0], f16::ONE);
        assert_eq!(data[1], f16::ONE);
        assert_eq!(data[2], f16::ZERO);
        assert_eq!(data[3], f16::ZERO);

        // Second sequence (length 3)
        assert_eq!(data[4], f16::ONE);
        assert_eq!(data[5], f16::ONE);
        assert_eq!(data[6], f16::ONE);
        assert_eq!(data[7], f16::ZERO);
    }

    #[test]
    fn test_combine_masks() {
        let mask1 = Tensor::from_vec(
            vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let mask2 = Tensor::from_vec(
            vec![f16::ONE, f16::ONE, f16::ZERO, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let combined = mask1.combine_masks(&mask2).unwrap();
        let data = combined.to_vec();

        // Logical AND
        assert_eq!(data[0], f16::ONE);  // 1 & 1 = 1
        assert_eq!(data[1], f16::ZERO); // 0 & 1 = 0
        assert_eq!(data[2], f16::ZERO); // 1 & 0 = 0
        assert_eq!(data[3], f16::ONE);  // 1 & 1 = 1
    }
}
