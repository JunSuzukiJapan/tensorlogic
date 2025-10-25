//! TensorLike trait for operations on tensor-like arrays

use crate::error::TensorResult;
use crate::tensor::Tensor;

/// Trait for tensor-like data structures
///
/// Provides common operations needed for token ID arrays and tensors.
/// This allows using the same interface for both f16 tensors and i64 token arrays.
pub trait TensorLike: Clone {
    /// Get the shape of the array/tensor
    fn shape(&self) -> Vec<usize>;

    /// Get number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get total number of elements
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Convert to Vec<i64> for token IDs
    /// For tensors, this rounds f16 values to nearest integer
    fn to_token_ids(&self) -> TensorResult<Vec<i64>>;
}

/// Token ID array - preserves integer precision without f16 conversion
#[derive(Debug, Clone, PartialEq)]
pub struct TokenIdArray {
    data: Vec<i64>,
    shape: Vec<usize>,
}

impl TokenIdArray {
    /// Create new token ID array from vector
    pub fn new(data: Vec<i64>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    /// Create from single token ID
    pub fn from_single(id: i64) -> Self {
        Self::new(vec![id])
    }

    /// Concatenate two token ID arrays
    pub fn concat(&self, other: &TokenIdArray) -> TensorResult<TokenIdArray> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(crate::error::TensorError::ShapeMismatch {
                expected: vec![1],
                actual: vec![self.ndim(), other.ndim()],
            });
        }

        let mut new_data = self.data.clone();
        new_data.extend_from_slice(&other.data);
        Ok(TokenIdArray::new(new_data))
    }

    /// Push a token ID to the array
    pub fn push(&mut self, id: i64) {
        self.data.push(id);
        self.shape[0] = self.data.len();
    }

    /// Get slice of token IDs
    pub fn slice(&self, start: usize, end: usize) -> TensorResult<TokenIdArray> {
        if start >= self.data.len() || end > self.data.len() || start >= end {
            return Err(crate::error::TensorError::ShapeMismatch {
                expected: vec![start, end],
                actual: vec![self.data.len()],
            });
        }
        Ok(TokenIdArray::new(self.data[start..end].to_vec()))
    }

    /// Get reference to underlying data
    pub fn data(&self) -> &[i64] {
        &self.data
    }

    /// Get single element at index
    pub fn get(&self, index: usize) -> Option<i64> {
        self.data.get(index).copied()
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl TensorLike for TokenIdArray {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn to_token_ids(&self) -> TensorResult<Vec<i64>> {
        Ok(self.data.clone())
    }
}

impl TensorLike for Tensor {
    fn shape(&self) -> Vec<usize> {
        self.dims().to_vec()
    }

    fn to_token_ids(&self) -> TensorResult<Vec<i64>> {
        // Convert tensor to f32 values and round to i64
        let values = self.to_vec_f32();
        Ok(values.into_iter().map(|v| v.round() as i64).collect())
    }
}

impl From<i64> for TokenIdArray {
    fn from(id: i64) -> Self {
        TokenIdArray::from_single(id)
    }
}

impl From<Vec<i64>> for TokenIdArray {
    fn from(ids: Vec<i64>) -> Self {
        TokenIdArray::new(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_array_concat() {
        let arr1 = TokenIdArray::new(vec![1, 2, 3]);
        let arr2 = TokenIdArray::new(vec![4, 5]);
        let result = arr1.concat(&arr2).unwrap();

        assert_eq!(result.data(), &[1, 2, 3, 4, 5]);
        assert_eq!(result.shape(), vec![5]);
    }

    #[test]
    fn test_token_array_push() {
        let mut arr = TokenIdArray::new(vec![1, 2]);
        arr.push(3);

        assert_eq!(arr.data(), &[1, 2, 3]);
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_token_array_slice() {
        let arr = TokenIdArray::new(vec![10, 20, 30, 40, 50]);
        let sliced = arr.slice(1, 4).unwrap();

        assert_eq!(sliced.data(), &[20, 30, 40]);
    }

    #[test]
    fn test_no_precision_loss() {
        // Test large token IDs that would lose precision in f16
        let large_id = 20358_i64;
        let arr = TokenIdArray::from_single(large_id);

        assert_eq!(arr.get(0), Some(large_id));
        assert_eq!(arr.to_token_ids().unwrap()[0], large_id);
    }
}
