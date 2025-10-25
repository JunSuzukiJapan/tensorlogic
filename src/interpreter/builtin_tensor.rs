//! Basic tensor operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;
use crate::error::TensorError;
use half::f16;

impl Interpreter {
    pub(super) fn eval_tensor_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "shape" => Some(self.eval_shape(args)),
            "ones" => Some(self.eval_ones(args)),
            "reshape" => Some(self.eval_reshape(args)),
            "transpose" => Some(self.eval_transpose(args)),
            "broadcast_to" => Some(self.eval_broadcast_to(args)),
            "concat" => Some(self.eval_concat(args)),
            "rope" => Some(self.eval_rope(args)),
            "zeros" | "flatten" | "permute" |
            "gather" | "scatter" | "chunk" | "split" |
            "squeeze" | "unsqueeze" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Tensor function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// shape(tensor) -> tensor
    /// Returns a 1D tensor containing the dimensions of the input tensor
    fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("shape() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Get dimensions
        let dims = tensor.dims();

        // Convert dimensions to f16 vector
        let shape_data: Vec<f16> = dims.iter().map(|&d| f16::from_f32(d as f32)).collect();

        // Create 1D tensor with the dimensions
        let device = tensor.device().clone();
        let shape_tensor = match &device {
            crate::device::Device::Metal(metal_device) => {
                Tensor::from_vec_metal(metal_device, shape_data, vec![dims.len()])
            }
            crate::device::Device::CPU => {
                Tensor::from_vec(shape_data, vec![dims.len()])
            }
            crate::device::Device::NeuralEngine => {
                // Fallback to CPU for NeuralEngine
                Tensor::from_vec(shape_data, vec![dims.len()])
            }
        }.map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(shape_tensor))
    }

    /// ones(shape) -> tensor
    /// Creates a tensor filled with ones
    fn eval_ones(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("ones() expects 1 argument (shape array), got {}", args.len())
            ));
        }

        // Evaluate shape argument - should be an array literal like [4, 2048]
        let shape_val = self.eval_expr(&args[0])?;

        // Extract shape from array literal
        let shape = match shape_val {
            Value::Tensor(ref t) => {
                // Extract the values and convert to usize
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("ones() expects shape as array, got {:?}", shape_val)
            )),
        };

        // Calculate total number of elements
        let numel: usize = shape.iter().product();

        // Create vector filled with ones
        let data = vec![f16::ONE; numel];

        // Create tensor on Metal device
        let device = self.env.metal_device();
        let tensor = Tensor::from_vec_metal(device, data, shape)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(tensor))
    }

    /// reshape(tensor, new_shape) -> tensor
    /// Reshapes a tensor to a new shape
    fn eval_reshape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("reshape() expects 2 arguments (tensor, new_shape), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Evaluate new_shape argument
        let shape_val = self.eval_expr(&args[1])?;
        let new_shape = match shape_val {
            Value::Tensor(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("reshape() expects new_shape as array, got {:?}", shape_val)
            )),
        };

        // Verify total number of elements matches
        let old_numel: usize = tensor.dims().iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(RuntimeError::TensorError(
                TensorError::ShapeMismatch {
                    expected: vec![old_numel],
                    actual: vec![new_numel],
                }
            ));
        }

        // Reshape the tensor
        let reshaped = tensor.reshape(new_shape)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(reshaped))
    }

    /// transpose(tensor) -> tensor
    /// Transpose a 2D tensor (swap dimensions 0 and 1)
    fn eval_transpose(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("transpose() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Transpose the tensor
        let transposed = tensor.transpose()
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(transposed))
    }

    /// broadcast_to(tensor, target_shape) -> tensor
    /// Broadcast tensor to target shape
    fn eval_broadcast_to(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("broadcast_to() expects 2 arguments (tensor, target_shape), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Evaluate target_shape argument
        let shape_val = self.eval_expr(&args[1])?;
        let target_dims = match shape_val {
            Value::Tensor(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("broadcast_to() expects target_shape as array, got {:?}", shape_val)
            )),
        };

        // Create TensorShape from dimensions
        let target_shape = crate::tensor::TensorShape::new(target_dims);

        // Broadcast the tensor
        let broadcasted = tensor.broadcast_to(&target_shape)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(broadcasted))
    }

    /// concat(tensor1, tensor2, dim) -> tensor
    /// Concatenates two tensors along the specified dimension
    ///
    /// # Arguments
    /// - tensor1: First tensor
    /// - tensor2: Second tensor
    /// - dim: Dimension along which to concatenate (as f16 scalar tensor)
    ///
    /// # Example
    /// ```tensorlogic
    /// let a = zeros(device, [2, 3])
    /// let b = zeros(device, [2, 3])
    /// let c = concat(a, b, 0.0)  // Result: [4, 3]
    /// ```
    fn eval_concat(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("concat() takes exactly 3 arguments (tensor1, tensor2, dim), got {}", args.len())
            ));
        }

        // Evaluate first tensor
        let tensor1_val = self.eval_expr(&args[0])?;
        let tensor1 = match tensor1_val {
            Value::Tensor(ref t) => t,
            _ => return Err(RuntimeError::TypeError(
                format!("concat() expects first argument to be a tensor, got {:?}", tensor1_val)
            )),
        };

        // Evaluate second tensor
        let tensor2_val = self.eval_expr(&args[1])?;
        let tensor2 = match tensor2_val {
            Value::Tensor(ref t) => t,
            _ => return Err(RuntimeError::TypeError(
                format!("concat() expects second argument to be a tensor, got {:?}", tensor2_val)
            )),
        };

        // Evaluate dim argument - accept both Float and Tensor
        let dim_val = self.eval_expr(&args[2])?;
        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Tensor(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("concat() expects dim as scalar, got tensor with {} elements", t.numel())
                    ));
                }
                t.to_vec_f32()[0] as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("concat() expects dim as scalar (float or tensor), got {:?}", dim_val)
            )),
        };

        // Call Tensor::concat with two tensors
        let tensors = vec![tensor1, tensor2];
        let result = Tensor::concat(&tensors, dim)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }

    /// rope(tensor) -> tensor
    /// Apply Rotary Position Embedding (RoPE) to the tensor
    ///
    /// RoPE is used in LLaMA and other modern LLMs for position encoding.
    /// Input tensor should be of shape [..., seq_len, n_heads, head_dim]
    ///
    /// # Arguments
    /// * `tensor` - Input tensor (typically Q or K after reshaping to multi-head format)
    ///
    /// # Returns
    /// Tensor with RoPE applied, same shape as input
    ///
    /// # Example
    /// ```ignore
    /// let Q_heads = reshape(Q, [seq_len, 32.0, 64.0])
    /// let Q_rope = rope(Q_heads)  // Apply rotary position embedding
    /// ```
    fn eval_rope(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("rope() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Apply RoPE
        let result = tensor.rope()
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }
}
