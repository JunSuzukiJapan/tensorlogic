//! Tensor shape utilities

use crate::error::{TensorError, TensorResult};

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }

    /// Check if shapes are compatible for broadcasting
    pub fn can_broadcast_to(&self, other: &TensorShape) -> bool {
        if self.rank() > other.rank() {
            return false;
        }

        for (a, b) in self.dims.iter().rev().zip(other.dims.iter().rev()) {
            if *a != *b && *a != 1 {
                return false;
            }
        }

        true
    }

    /// Compute the broadcasted shape of two shapes
    pub fn broadcast_with(&self, other: &TensorShape) -> TensorResult<TensorShape> {
        let max_rank = self.rank().max(other.rank());
        let mut result_dims = vec![1; max_rank];

        // Align shapes from the right
        let self_offset = max_rank - self.rank();
        let other_offset = max_rank - other.rank();

        for i in 0..max_rank {
            let self_dim = if i >= self_offset {
                self.dims[i - self_offset]
            } else {
                1
            };

            let other_dim = if i >= other_offset {
                other.dims[i - other_offset]
            } else {
                1
            };

            if self_dim == other_dim {
                result_dims[i] = self_dim;
            } else if self_dim == 1 {
                result_dims[i] = other_dim;
            } else if other_dim == 1 {
                result_dims[i] = self_dim;
            } else {
                return Err(TensorError::ShapeMismatch {
                    expected: self.dims.clone(),
                    actual: other.dims.clone(),
                });
            }
        }

        Ok(TensorShape::new(result_dims))
    }

    /// Check if this shape needs broadcasting to match target shape
    pub fn needs_broadcast(&self, target: &TensorShape) -> bool {
        self.dims != target.dims
    }

    /// Compute strides for this shape (row-major / C-contiguous)
    pub fn compute_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.rank()];

        for i in (0..self.rank().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }

        strides
    }

    /// Check if two shapes are equal for element-wise operations
    pub fn is_same(&self, other: &TensorShape) -> bool {
        self.dims == other.dims
    }

    /// Reshape to a new shape (must have same number of elements)
    pub fn reshape(&self, new_dims: Vec<usize>) -> TensorResult<TensorShape> {
        let new_shape = TensorShape::new(new_dims);

        if self.numel() != new_shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims.clone(),
                actual: new_shape.dims.clone(),
            });
        }

        Ok(new_shape)
    }

    /// Get shape for matrix multiplication
    pub fn matmul_shape(&self, other: &TensorShape) -> TensorResult<TensorShape> {
        // Support 1D and 2D tensors
        match (self.rank(), other.rank()) {
            (2, 2) => {
                // Matrix x Matrix
                if self.dims[1] != other.dims[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.dims[0], self.dims[1], other.dims[1]],
                        actual: vec![self.dims[0], self.dims[1], other.dims[0], other.dims[1]],
                    });
                }
                Ok(TensorShape::new(vec![self.dims[0], other.dims[1]]))
            }
            (2, 1) => {
                // Matrix x Vector
                if self.dims[1] != other.dims[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.dims[0], self.dims[1]],
                        actual: vec![other.dims[0]],
                    });
                }
                Ok(TensorShape::new(vec![self.dims[0]]))
            }
            (1, 2) => {
                // Vector x Matrix
                if self.dims[0] != other.dims[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.dims[0]],
                        actual: vec![other.dims[0], other.dims[1]],
                    });
                }
                Ok(TensorShape::new(vec![other.dims[1]]))
            }
            (1, 1) => {
                // Vector dot product
                if self.dims[0] != other.dims[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.dims[0]],
                        actual: vec![other.dims[0]],
                    });
                }
                Ok(TensorShape::new(vec![1]))
            }
            _ => Err(TensorError::InvalidOperation(format!(
                "Unsupported matmul shapes: {:?} @ {:?}",
                self.dims, other.dims
            ))),
        }
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for TensorShape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_shape_strides() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        let strides = shape.compute_strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_reshape() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        let reshaped = shape.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.dims(), &[6, 4]);
    }

    #[test]
    fn test_matmul_shape() {
        let a = TensorShape::new(vec![2, 3]);
        let b = TensorShape::new(vec![3, 4]);
        let c = a.matmul_shape(&b).unwrap();
        assert_eq!(c.dims(), &[2, 4]);
    }

    #[test]
    fn test_matmul_shape_error() {
        let a = TensorShape::new(vec![2, 3]);
        let b = TensorShape::new(vec![4, 5]);
        assert!(a.matmul_shape(&b).is_err());
    }
}
