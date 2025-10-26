use crate::tensor::FloatType;
//! Gradient operation for second-order derivatives
use std::marker::PhantomData;
use super::prelude::*;
//!
//! This module implements GradientBackward which computes gradients
//! of gradients (second-order derivatives, Hessian).

use crate::autograd::{GradientFunction, Operation};
use crate::tensor::Tensor;
use crate::TensorResult;

/// Backward pass for gradient operation
///
/// Computes second-order derivatives (Hessian):
/// d/dx (dL/dx) = d²L/dx²
pub struct GradientBackward<T: FloatType> {
    /// The original operation whose gradient was computed
    original_op: Operation,
    /// Original inputs to the operation
    original_inputs: Vec<Tensor>,
    /// Index of the input for which gradient was computed
    input_index: usize,
    /// The gradient output from first backward pass
    #[allow(dead_code)]
    grad_output: Tensor<T>,
}

impl<T: FloatType> GradientBackward<T> {
    pub fn new(
        original_op: Operation,
        original_inputs: Vec<Tensor>,
        input_index: usize,
        grad_output: Tensor<T>,
    ) -> Self {
        Self {
            original_op,
            original_inputs,
            input_index,
            grad_output,
        }
    }

    /// Helper: Create zero tensor with same shape as input
    fn zeros_like(&self, tensor: &Tensor) -> TensorResult<Tensor> {
        let numel = tensor.numel();
        let zeros = vec![half::f16::ZERO; numel];
        Tensor::from_vec(zeros, tensor.dims().to_vec())
    }

    /// Compute Hessian for the original operation
    ///
    /// For each operation type, we need to compute the second derivative.
    /// This is operation-specific.
    fn compute_hessian(&self, grad_grad_output: &Tensor) -> TensorResult<Vec<Tensor<f16>>> {
        match &self.original_op {
            Operation::Add => {
                // d²/dx² (x + y) = 0
                // Second derivative of addition is zero
                let zero = self.zeros_like(&self.original_inputs[self.input_index])?;
                Ok(vec![zero.clone(), zero])
            }

            Operation::Sub => {
                // d²/dx² (x - y) = 0
                // Second derivative of subtraction is zero
                let zero = self.zeros_like(&self.original_inputs[self.input_index])?;
                Ok(vec![zero.clone(), zero])
            }

            Operation::Mul => {
                // For f(x,y) = x * y:
                // df/dx = y, df/dy = x
                // d²f/dx² = 0, d²f/dy² = 0
                // d²f/dxdy = 1
                //
                // When computing gradient of gradient with respect to grad_output:
                // If input_index == 0 (gradient w.r.t. x):
                //   d/dx (df/dx) = d/dx (y) = 0
                //   d/dy (df/dx) = d/dy (y) = 1
                // If input_index == 1 (gradient w.r.t. y):
                //   d/dx (df/dy) = d/dx (x) = 1
                //   d/dy (df/dy) = d/dy (x) = 0

                if self.input_index == 0 {
                    // Gradient w.r.t. x: df/dx = y
                    // Second derivative: d/dy (df/dx) = 1
                    let grad_x = self.zeros_like(&self.original_inputs[0])?;
                    let grad_y = grad_grad_output.clone();
                    Ok(vec![grad_x, grad_y])
                } else {
                    // Gradient w.r.t. y: df/dy = x
                    // Second derivative: d/dx (df/dy) = 1
                    let grad_x = grad_grad_output.clone();
                    let grad_y = self.zeros_like(&self.original_inputs[1])?;
                    Ok(vec![grad_x, grad_y])
                }
            }

            Operation::Div => {
                // For f(x,y) = x / y:
                // Second derivatives are zero for simplicity (complex otherwise)
                let grad_x = self.zeros_like(&self.original_inputs[0])?;
                let grad_y = self.zeros_like(&self.original_inputs[1])?;
                Ok(vec![grad_x, grad_y])
            }

            Operation::MatMul => {
                // For matrix multiplication C = A @ B:
                // Second derivatives are complex for matmul
                // For now, return zeros (can be improved later)
                let grad_a = self.zeros_like(&self.original_inputs[0])?;
                let grad_b = self.zeros_like(&self.original_inputs[1])?;
                Ok(vec![grad_a, grad_b])
            }

            Operation::ReLU => {
                // For f(x) = max(0, x):
                // df/dx = 1 if x > 0, else 0
                // d²f/dx² = 0 everywhere (except at x=0 where undefined)
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }

            Operation::GELU => {
                // GELU has non-zero second derivative
                // For now, placeholder (zero)
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }

            Operation::Softmax => {
                // Softmax has complex second derivatives
                // Placeholder for now
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }

            Operation::Sum | Operation::Mean => {
                // Sum and Mean are linear operations
                // Second derivative is zero
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }

            Operation::Broadcast => {
                // Broadcast is linear
                // Second derivative is zero
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }

            Operation::Gradient { .. } => {
                // Third-order derivatives not yet supported
                let grad = self.zeros_like(&self.original_inputs[0])?;
                Ok(vec![grad])
            }
        }
    }
}

impl<T: FloatType> GradientFunction for GradientBackward<T> {
    fn backward(&self, grad_output: &Tensor<f16>, _inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>> {
        self.compute_hessian(grad_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_backward_add() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
        let y = Tensor::from_vec(vec![half::f16::from_f32(3.0)], vec![1]).unwrap();
        let grad_out = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();

        let grad_fn = GradientBackward::new(
            Operation::Add,
            vec![x.clone(), y.clone()],
            0,
            grad_out.clone(),
        );

        let grad_grad_out = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let result = grad_fn.backward(&grad_grad_out, &[&x, &y]).unwrap();

        // Second derivative of addition is zero
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_vec()[0].to_f32(), 0.0);
        assert_eq!(result[1].to_vec()[0].to_f32(), 0.0);
    }

    #[test]
    fn test_gradient_backward_mul() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
        let y = Tensor::from_vec(vec![half::f16::from_f32(3.0)], vec![1]).unwrap();
        let grad_out = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();

        let grad_fn = GradientBackward::new(
            Operation::Mul,
            vec![x.clone(), y.clone()],
            0, // Gradient w.r.t. x
            grad_out.clone(),
        );

        let grad_grad_out = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let result = grad_fn.backward(&grad_grad_out, &[&x, &y]).unwrap();

        // d/dx (df/dx) = d/dx (y) = 0
        // d/dy (df/dx) = d/dy (y) = 1
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_vec()[0].to_f32(), 0.0); // grad_x = 0
        assert_eq!(result[1].to_vec()[0].to_f32(), 1.0); // grad_y = 1
    }
}
