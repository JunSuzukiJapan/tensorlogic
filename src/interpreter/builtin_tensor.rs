//! Basic tensor operations for TensorLogic interpreter

use super::*;
use crate::interpreter::value::ToValue;
use crate::tensor::Tensor;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::error::TensorError;
use half::f16;

impl Interpreter {
    pub(super) fn eval_tensor_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            // Tensor properties and creation
            "shape" => Some(self.eval_shape(args)),
            "ones" => Some(self.eval_ones(args)),
            "reshape" => Some(self.eval_reshape(args)),
            "transpose" => Some(self.eval_transpose(args)),
            "broadcast_to" => Some(self.eval_broadcast_to(args)),
            "concat" => Some(self.eval_concat(args)),
            "rope" => Some(self.eval_rope(args)),
            "slice" => Some(self.eval_slice(args)),

            // Arithmetic operations (for method chaining)
            "add" => Some(self.eval_add_method(args)),
            "sub" => Some(self.eval_sub_method(args)),
            "mul" => Some(self.eval_mul_method(args)),
            "div" => Some(self.eval_div_method(args)),

            // Reduction operations (for method chaining)
            "sum" => Some(self.eval_sum_method(args)),
            "mean" => Some(self.eval_mean_method(args)),
            "max" => Some(self.eval_max_method(args)),
            "min" => Some(self.eval_min_method(args)),
            "argmax" => Some(self.eval_argmax_method(args)),
            "argmin" => Some(self.eval_argmin_method(args)),

            // Tensor shape operations
            "zeros" => Some(self.eval_zeros(args)),
            "flatten" => Some(self.eval_flatten(args)),
            "squeeze" => Some(self.eval_squeeze(args)),
            "unsqueeze" => Some(self.eval_unsqueeze(args)),
            "permute" => Some(self.eval_permute(args)),

            // Advanced indexing operations
            "gather" => Some(self.eval_gather(args)),
            "scatter" => Some(self.eval_scatter(args)),

            // Split operations
            "chunk" => Some(self.eval_chunk(args)),
            "split" => Some(self.eval_split(args)),

            // No longer any unimplemented tensor operations

            _ => None,
        }
    }

    /// shape(tensor) -> tensor
    /// Returns a 1D tensor containing the dimensions of the input tensor
    fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("shape() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        Ok(match val {
            Value::TensorF16(tensor) => {
                let dims = tensor.dims();
                let shape_data: Vec<f16> = dims.iter().map(|&d| f16::from_f32(d as f32)).collect();
                let device = tensor.device().clone();
                let shape_tensor = match &device {
                    crate::device::Device::Metal(metal_device) => {
                        Tensor::from_vec_metal(metal_device, shape_data, vec![dims.len()])
                    }
                    crate::device::Device::CPU => {
                        Tensor::from_vec(shape_data, vec![dims.len()])
                    }
                    crate::device::Device::NeuralEngine => {
                        Tensor::from_vec(shape_data, vec![dims.len()])
                    }
                }.map_err(|e| RuntimeError::TensorError(e))?;
                shape_tensor.to_value()
            }
            Value::TensorF32(tensor) => {
                let dims = tensor.dims();
                let shape_data: Vec<f32> = dims.iter().map(|&d| d as f32).collect();
                let device = tensor.device().clone();
                let shape_tensor = match &device {
                    crate::device::Device::Metal(metal_device) => {
                        Tensor::from_vec_metal(metal_device, shape_data, vec![dims.len()])
                    }
                    crate::device::Device::CPU => {
                        Tensor::from_vec(shape_data, vec![dims.len()])
                    }
                    crate::device::Device::NeuralEngine => {
                        Tensor::from_vec(shape_data, vec![dims.len()])
                    }
                }.map_err(|e| RuntimeError::TensorError(e))?;
                shape_tensor.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
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
            Value::TensorF16(ref t) => {
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

        Ok(Value::TensorF16(tensor))
    }

    /// reshape(tensor, new_shape) -> tensor
    /// Reshapes a tensor to a new shape
    fn eval_reshape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("reshape() expects 2 arguments (tensor, new_shape), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;
        let shape_val = self.eval_expr(&args[1])?;

        // Extract new_shape from shape_val
        let new_shape = match shape_val {
            Value::TensorF16(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            Value::TensorF32(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("reshape() expects new_shape as tensor, got {:?}", shape_val)
            )),
        };

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
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
                tensor.reshape(new_shape).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(tensor) => {
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
                tensor.reshape(new_shape).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
    }

    /// transpose(tensor) -> tensor
    /// Transpose a 2D tensor (swap dimensions 0 and 1)
    fn eval_transpose(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("transpose() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                tensor.transpose().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(tensor) => {
                tensor.transpose().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
    }

    /// broadcast_to(tensor, target_shape) -> tensor
    /// Broadcast tensor to target shape
    fn eval_broadcast_to(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("broadcast_to() expects 2 arguments (tensor, target_shape), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;
        let shape_val = self.eval_expr(&args[1])?;

        // Extract target_dims from shape_val
        let target_dims = match shape_val {
            Value::TensorF16(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            Value::TensorF32(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("broadcast_to() expects target_shape as tensor, got {:?}", shape_val)
            )),
        };

        let target_shape = crate::tensor::TensorShape::new(target_dims);

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                tensor.broadcast_to(&target_shape).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(tensor) => {
                tensor.broadcast_to(&target_shape).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
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

        // Evaluate first argument
        let val1 = self.eval_expr(&args[0])?;
        let val2 = self.eval_expr(&args[1])?;

        // Handle TokenIdArray concatenation (for token sequences)
        if let (Value::TokenIdArray(arr1), Value::TokenIdArray(arr2)) = (&val1, &val2) {
            let result = arr1.concat(arr2)
                .map_err(|e| RuntimeError::TypeError(format!("TokenIdArray concat failed: {}", e)))?;
            return Ok(Value::TokenIdArray(result));
        }

        // Handle Tensor concatenation - support both f16 and f32
        use crate::interpreter::value::ToValue;

        // Evaluate dim argument first
        let dim_val = self.eval_expr(&args[2])?;
        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            Value::TensorF32(ref t) if t.numel() == 1 => t.to_vec()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                format!("concat() expects dim as scalar, got {:?}", dim_val)
            )),
        };

        // Process based on tensor types
        match (val1, val2) {
            (Value::TensorF16(tensor1), Value::TensorF16(tensor2)) => {
                let tensors = vec![&tensor1, &tensor2];
                let output = crate::tensor::Tensor::concat(&tensors[..], dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(output.to_value())
            }
            (Value::TensorF32(tensor1), Value::TensorF32(tensor2)) => {
                let tensors = vec![&tensor1, &tensor2];
                let output = crate::tensor::Tensor::concat(&tensors[..], dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(output.to_value())
            }
            _ => Err(RuntimeError::TypeError(
                "concat() requires both tensors to be the same type (both f16 or both f32)".to_string()
            ))
        }
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
    /// let Q_rope = rope(Q_heads)  // Apply rotary position embedding with position_offset=0
    /// let Q_rope = rope(Q_heads, 29)  // Apply RoPE starting at position 29 (for KV cache)
    /// ```
    fn eval_rope(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() < 1 || args.len() > 2 {
            return Err(RuntimeError::TypeError(
                format!("rope() expects 1 or 2 arguments (tensor, [position_offset]), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        // Get position offset (default 0 for backward compatibility)
        let position_offset = if args.len() == 2 {
            let offset_val = self.eval_expr(&args[1])?;
            match offset_val {
                Value::Float(f) => f as usize,
                Value::Integer(i) => i as usize,
                Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
                Value::TensorF32(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
                Value::TokenIdArray(ref arr) if arr.len() == 1 => arr.get(0).unwrap() as usize,
                _ => return Err(RuntimeError::TypeError(
                    "rope() position_offset must be a scalar integer".to_string()
                )),
            }
        } else {
            0  // Default to 0 if not provided
        };

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                tensor.rope(position_offset).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(tensor) => {
                tensor.rope(position_offset).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
    }

    /// slice(tensor, row, col_start, col_end) -> tensor
    /// Extract a slice from a 2D tensor (specific row, column range)
    ///
    /// # Arguments
    /// * `tensor` - Input 2D tensor
    /// * `row` - Row index to extract
    /// * `col_start` - Starting column index (inclusive)
    /// * `col_end` - Ending column index (exclusive)
    ///
    /// # Returns
    /// 1D tensor containing the specified slice
    ///
    /// # Example
    /// ```ignore
    /// let data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    /// let row0_cols = slice(data, 0, 0, 3)  // [1, 2, 3]
    /// let row1_cols = slice(data, 1, 1, 4)  // [6, 7, 8]
    /// ```
    fn eval_slice(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // Support both 3-arg (1D slice) and 4-arg (2D slice) variants
        if args.len() == 3 {
            // 1D slice: slice(array, start, end)
            return self.eval_slice_1d(args);
        } else if args.len() != 4 {
            return Err(RuntimeError::TypeError(
                format!("slice() expects 3 arguments (array, start, end) for 1D or 4 arguments (tensor, row, col_start, col_end) for 2D, got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Check tensor is 2D
        let dims = tensor.dims();
        if dims.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("slice() expects 2D tensor, got {}D tensor", dims.len())
            ));
        }

        let rows = dims[0];
        let cols = dims[1];

        // Evaluate row argument
        let row_val = self.eval_expr(&args[1])?;
        let row = match row_val {
            Value::Float(f) => f as usize,
            Value::TensorF16(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("slice() expects row as scalar, got tensor with {} elements", t.numel())
                    ));
                }
                t.to_vec_f32()[0] as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("slice() expects row as scalar, got {:?}", row_val)
            )),
        };

        // Evaluate col_start argument
        let col_start_val = self.eval_expr(&args[2])?;
        let col_start = match col_start_val {
            Value::Float(f) => f as usize,
            Value::TensorF16(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("slice() expects col_start as scalar, got tensor with {} elements", t.numel())
                    ));
                }
                t.to_vec_f32()[0] as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("slice() expects col_start as scalar, got {:?}", col_start_val)
            )),
        };

        // Evaluate col_end argument
        let col_end_val = self.eval_expr(&args[3])?;
        let col_end = match col_end_val {
            Value::Float(f) => f as usize,
            Value::TensorF16(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("slice() expects col_end as scalar, got tensor with {} elements", t.numel())
                    ));
                }
                t.to_vec_f32()[0] as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("slice() expects col_end as scalar, got {:?}", col_end_val)
            )),
        };

        // Validate indices
        if row >= rows {
            return Err(RuntimeError::TypeError(
                format!("slice() row index {} out of bounds (tensor has {} rows)", row, rows)
            ));
        }
        if col_start >= cols || col_end > cols {
            return Err(RuntimeError::TypeError(
                format!("slice() column range [{}, {}) out of bounds (tensor has {} columns)", col_start, col_end, cols)
            ));
        }
        if col_start >= col_end {
            return Err(RuntimeError::TypeError(
                format!("slice() invalid range: col_start ({}) must be < col_end ({})", col_start, col_end)
            ));
        }

        // Get tensor data
        let data = tensor.to_vec();

        // Extract the slice
        let offset = row * cols + col_start;
        let length = col_end - col_start;
        let slice_data: Vec<f16> = data[offset..offset + length].to_vec();

        // Create 1D tensor with the slice
        let device = tensor.device().clone();
        let result_tensor = match &device {
            crate::device::Device::Metal(metal_device) => {
                Tensor::from_vec_metal(metal_device, slice_data, vec![length])
            }
            crate::device::Device::CPU => {
                Tensor::from_vec(slice_data, vec![length])
            }
            crate::device::Device::NeuralEngine => {
                // Fallback to CPU for NeuralEngine
                Tensor::from_vec(slice_data, vec![length])
            }
        }.map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(result_tensor))
    }

    /// 1D slice: slice(array, start, end)
    fn eval_slice_1d(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // Evaluate array argument
        let array_val = self.eval_expr(&args[0])?;

        // Get start index
        let start_val = self.eval_expr(&args[1])?;
        let start = match start_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            Value::TokenIdArray(ref arr) if arr.len() == 1 => arr.get(0).unwrap() as usize,
            _ => return Err(RuntimeError::TypeError(
                format!("slice() start index must be scalar, got {:?}", start_val)
            )),
        };

        // Get end index
        let end_val = self.eval_expr(&args[2])?;
        let end = match end_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            Value::TokenIdArray(ref arr) if arr.len() == 1 => arr.get(0).unwrap() as usize,
            _ => return Err(RuntimeError::TypeError(
                format!("slice() end index must be scalar, got {:?}", end_val)
            )),
        };

        // Handle TokenIdArray
        if let Value::TokenIdArray(arr) = &array_val {
            let sliced = arr.slice(start, end)
                .map_err(|e| RuntimeError::TypeError(format!("TokenIdArray slice failed: {}", e)))?;
            return Ok(Value::TokenIdArray(sliced));
        }

        // Handle Tensor (1D)
        let tensor = array_val.as_tensor()?;
        let dims = tensor.dims();

        if dims.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("1D slice() expects 1D tensor or TokenIdArray, got {}D tensor", dims.len())
            ));
        }

        let len = dims[0];

        // Validate indices
        if start >= len || end > len || start >= end {
            return Err(RuntimeError::TypeError(
                format!("slice() indices out of bounds: start={}, end={}, length={}", start, end, len)
            ));
        }

        // Extract slice from tensor
        let data = tensor.to_vec();
        let slice_data: Vec<_> = data[start..end].to_vec();
        let length = end - start;

        // Create result tensor on same device
        let result_tensor = match tensor.device() {
            crate::device::Device::Metal(metal_device) => {
                Tensor::from_vec_metal(metal_device, slice_data, vec![length])
            }
            crate::device::Device::CPU => {
                Tensor::from_vec(slice_data, vec![length])
            }
            crate::device::Device::NeuralEngine => {
                Tensor::from_vec(slice_data, vec![length])
            }
        }.map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(result_tensor))
    }

    /// add(tensor1, tensor2) -> tensor
    /// Adds two tensors element-wise (supports method chaining)
    fn eval_add_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("add() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.add(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.add(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "add() requires two tensors of the same type".to_string()
            ))
        }
    }

    /// sub(tensor1, tensor2) -> tensor
    /// Subtracts tensor2 from tensor1 element-wise (supports method chaining)
    fn eval_sub_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("sub() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.sub(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.sub(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "sub() requires two tensors of the same type".to_string()
            ))
        }
    }

    /// mul(tensor1, tensor2_or_scalar) -> tensor
    /// Multiplies tensors element-wise or by scalar (supports method chaining)
    fn eval_mul_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("mul() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.mul(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.mul(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            (Value::TensorF16(t), Value::Float(scalar)) | (Value::Float(scalar), Value::TensorF16(t)) => {
                let result = t.mul_scalar(f16::from_f32(scalar as f32))
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            // TODO: Implement mul_scalar for f32 tensors
            (Value::TensorF32(_), Value::Float(_)) | (Value::Float(_), Value::TensorF32(_)) => {
                Err(RuntimeError::NotImplemented(
                    "mul_scalar for f32 tensors not yet implemented".to_string()
                ))
            }
            _ => Err(RuntimeError::TypeError(
                "mul() requires two tensors or a tensor and a scalar".to_string()
            ))
        }
    }

    /// div(tensor1, tensor2_or_scalar) -> tensor
    /// Divides tensor1 by tensor2 or scalar element-wise (supports method chaining)
    fn eval_div_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("div() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.div(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.div(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "div() requires two tensors of the same type".to_string()
            ))
        }
    }

    /// sum(tensor) -> scalar
    /// Computes the sum of all elements in the tensor
    fn eval_sum_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("sum() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.sum()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Float(result.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.sum()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Float(result.to_f32() as f64))
            }
            _ => Err(RuntimeError::TypeError(
                "sum() expects a tensor".to_string()
            ))
        }
    }

    /// mean(tensor) -> scalar
    /// Computes the mean (average) of all elements in the tensor
    fn eval_mean_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("mean() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.mean()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Float(result.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.mean()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Float(result.to_f32() as f64))
            }
            _ => Err(RuntimeError::TypeError(
                "mean() expects a tensor".to_string()
            ))
        }
    }

    /// zeros(shape) -> tensor
    /// Creates a tensor filled with zeros
    fn eval_zeros(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("zeros() expects 1 argument (shape array), got {}", args.len())
            ));
        }

        // Evaluate shape argument
        let shape_val = self.eval_expr(&args[0])?;

        // Extract shape from array literal
        let shape = match shape_val {
            Value::TensorF16(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("zeros() expects shape as array, got {:?}", shape_val)
            )),
        };

        // Calculate total number of elements
        let numel: usize = shape.iter().product();

        // Create vector filled with zeros
        let data = vec![f16::ZERO; numel];

        // Create tensor on Metal device
        let device = self.env.metal_device();
        let tensor = Tensor::from_vec_metal(device, data, shape)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(tensor))
    }

    /// flatten(tensor) -> tensor
    /// Flattens a tensor to 1D
    fn eval_flatten(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("flatten() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let numel = tensor.numel();
                let result = tensor.reshape(vec![numel])
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            Value::TensorF32(tensor) => {
                let numel = tensor.numel();
                let result = tensor.reshape(vec![numel])
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "flatten() expects a tensor".to_string()
            ))
        }
    }

    /// squeeze(tensor) -> tensor
    /// Removes dimensions of size 1 from the tensor shape
    fn eval_squeeze(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.is_empty() || args.len() > 2 {
            return Err(RuntimeError::TypeError(
                format!("squeeze() expects 1-2 arguments (tensor, optional dim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        // Parse optional dim parameter
        let dim = if args.len() >= 2 {
            match self.eval_expr(&args[1])? {
                Value::Integer(i) => Some(i as usize),
                Value::Float(f) => Some(f as usize),
                _ => None,
            }
        } else {
            None
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.squeeze(dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.squeeze(dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "squeeze() expects a tensor".to_string()
            ))
        }
    }

    /// unsqueeze(tensor, dim) -> tensor
    /// Adds a dimension of size 1 at the specified position
    fn eval_unsqueeze(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("unsqueeze() expects 2 arguments (tensor, dim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let dim_val = self.eval_expr(&args[1])?;

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "unsqueeze() dim must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let mut dims = tensor.dims().to_vec();

                if dim > dims.len() {
                    return Err(RuntimeError::TypeError(
                        format!("unsqueeze() dim {} out of range for tensor with {} dimensions", dim, dims.len())
                    ));
                }

                dims.insert(dim, 1);

                let result = tensor.reshape(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            Value::TensorF32(tensor) => {
                let mut dims = tensor.dims().to_vec();

                if dim > dims.len() {
                    return Err(RuntimeError::TypeError(
                        format!("unsqueeze() dim {} out of range for tensor with {} dimensions", dim, dims.len())
                    ));
                }

                dims.insert(dim, 1);

                let result = tensor.reshape(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "unsqueeze() expects a tensor".to_string()
            ))
        }
    }

    /// max(tensor) -> scalar
    /// Returns the maximum value in the tensor
    fn eval_max_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("max() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let data = tensor.to_vec();
                let max_val = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(max_val.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let data = tensor.to_vec();
                let max_val = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(max_val.to_f32() as f64))
            }
            _ => Err(RuntimeError::TypeError(
                "max() expects a tensor".to_string()
            ))
        }
    }

    /// min(tensor) -> scalar
    /// Returns the minimum value in the tensor
    fn eval_min_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("min() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let data = tensor.to_vec();
                let min_val = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(min_val.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let data = tensor.to_vec();
                let min_val = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(min_val.to_f32() as f64))
            }
            _ => Err(RuntimeError::TypeError(
                "min() expects a tensor".to_string()
            ))
        }
    }

    /// argmax(tensor) -> integer
    /// Returns the index of the maximum value in the flattened tensor
    fn eval_argmax_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.is_empty() || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("argmax() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        // Parse optional dim parameter
        let dim = if args.len() >= 2 {
            match self.eval_expr(&args[1])? {
                Value::Integer(i) => Some(i as usize),
                Value::Float(f) => Some(f as usize),
                _ => None,
            }
        } else {
            None
        };

        // Parse optional keepdim parameter
        let keepdim = if args.len() >= 3 {
            match self.eval_expr(&args[2])? {
                Value::Integer(i) => i != 0,
                Value::Float(f) => f != 0.0,
                _ => false,
            }
        } else {
            false
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.argmax(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.to_vec();
                    Ok(Value::Integer(data[0].to_f32() as i64))
                } else {
                    Ok(Value::TensorF16(result))
                }
            }
            Value::TensorF32(tensor) => {
                let result = tensor.argmax(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.to_vec_f32();
                    Ok(Value::Integer(data[0] as i64))
                } else {
                    Ok(Value::TensorF32(result))
                }
            }
            _ => Err(RuntimeError::TypeError(
                "argmax() expects a tensor".to_string()
            ))
        }
    }

    /// argmin(tensor, [dim], [keepdim]) -> integer or tensor
    /// Returns the index of the minimum value
    fn eval_argmin_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.is_empty() || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("argmin() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        // Parse optional dim parameter
        let dim = if args.len() >= 2 {
            match self.eval_expr(&args[1])? {
                Value::Integer(i) => Some(i as usize),
                Value::Float(f) => Some(f as usize),
                _ => None,
            }
        } else {
            None
        };

        // Parse optional keepdim parameter
        let keepdim = if args.len() >= 3 {
            match self.eval_expr(&args[2])? {
                Value::Integer(i) => i != 0,
                Value::Float(f) => f != 0.0,
                _ => false,
            }
        } else {
            false
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.argmin(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.to_vec();
                    Ok(Value::Integer(data[0].to_f32() as i64))
                } else {
                    Ok(Value::TensorF16(result))
                }
            }
            Value::TensorF32(tensor) => {
                let result = tensor.argmin(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.to_vec_f32();
                    Ok(Value::Integer(data[0] as i64))
                } else {
                    Ok(Value::TensorF32(result))
                }
            }
            _ => Err(RuntimeError::TypeError(
                "argmin() expects a tensor".to_string()
            ))
        }
    }

    /// permute(tensor, dims) -> tensor
    /// Permutes the dimensions of the tensor according to the given order
    /// Example: permute([2, 3, 4], [2, 0, 1]) -> [4, 2, 3]
    fn eval_permute(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("permute() expects 2 arguments (tensor, dims), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let dims_val = self.eval_expr(&args[1])?;

        // Extract dims from array
        let dims = match dims_val {
            Value::TensorF16(ref t) => {
                let data = t.to_vec_f32();
                data.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(RuntimeError::TypeError(
                format!("permute() expects dims as array, got {:?}", dims_val)
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.permute(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.permute(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "permute() expects a tensor".to_string()
            ))
        }
    }

    /// gather(tensor, dim, index) -> tensor
    /// Gathers values along an axis specified by dim
    fn eval_gather(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("gather() expects 3 arguments (tensor, dim, index), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let dim_val = self.eval_expr(&args[1])?;
        let index_val = self.eval_expr(&args[2])?;

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "gather() dim must be a scalar".to_string()
            )),
        };

        match (val, index_val) {
            (Value::TensorF16(tensor), Value::TensorF16(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(tensor), Value::TensorF32(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "gather() requires tensor and index of the same type".to_string()
            ))
        }
    }

    /// scatter(tensor, dim, index, src) -> tensor
    /// Scatters values along an axis specified by dim
    fn eval_scatter(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 4 {
            return Err(RuntimeError::TypeError(
                format!("scatter() expects 4 arguments (tensor, dim, index, src), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let dim_val = self.eval_expr(&args[1])?;
        let index_val = self.eval_expr(&args[2])?;
        let src_val = self.eval_expr(&args[3])?;

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "scatter() dim must be a scalar".to_string()
            )),
        };

        // Note: scatter expects indices as &Tensor<f16> regardless of the main tensor type
        match (val, src_val) {
            (Value::TensorF16(tensor), Value::TensorF16(src)) => {
                // Extract index as f16 tensor
                let index = match index_val {
                    Value::TensorF16(idx) => idx,
                    _ => return Err(RuntimeError::TypeError(
                        "scatter() requires index as f16 tensor".to_string()
                    )),
                };
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(tensor), Value::TensorF32(src)) => {
                // Extract index as f16 tensor (indices are always f16)
                let index = match index_val {
                    Value::TensorF16(idx) => idx,
                    _ => return Err(RuntimeError::TypeError(
                        "scatter() requires index as f16 tensor".to_string()
                    )),
                };
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "scatter() requires tensor and src of the same type".to_string()
            ))
        }
    }

    /// chunk(tensor, chunks, dim) -> list of tensors
    /// Splits a tensor into a specific number of chunks along a dimension
    fn eval_chunk(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("chunk() expects 3 arguments (tensor, chunks, dim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let chunks_val = self.eval_expr(&args[1])?;
        let dim_val = self.eval_expr(&args[2])?;

        let chunks = match chunks_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "chunk() chunks must be a scalar".to_string()
            )),
        };

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "chunk() dim must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.chunk(chunks, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                // Return as array of tensors
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF16(t))
                    .collect();
                // For now, return the first chunk as we don't have array support
                // TODO: Implement proper array return type
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF16(tensor)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.chunk(chunks, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF32(t))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF32(tensor)))
            }
            _ => Err(RuntimeError::TypeError(
                "chunk() expects a tensor".to_string()
            ))
        }
    }

    /// split(tensor, split_size, dim) -> list of tensors
    /// Splits a tensor into chunks of a specific size along a dimension
    fn eval_split(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("split() expects 3 arguments (tensor, split_size, dim), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let split_size_val = self.eval_expr(&args[1])?;
        let dim_val = self.eval_expr(&args[2])?;

        let split_size = match split_size_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "split() split_size must be a scalar".to_string()
            )),
        };

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => t.to_vec_f32()[0] as usize,
            _ => return Err(RuntimeError::TypeError(
                "split() dim must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.split(split_size, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF16(t))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF16(tensor)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.split(split_size, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF32(t))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF32(tensor)))
            }
            _ => Err(RuntimeError::TypeError(
                "split() expects a tensor".to_string()
            ))
        }
    }
}
