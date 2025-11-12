//! Basic tensor operations for TensorLogic interpreter

use super::*;
use crate::interpreter::value::ToValue;
use crate::tensor::Tensor;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::tensor::TensorConvert;
use crate::error::TensorError;
use crate::device::Device;
use half::f16;
use std::sync::Arc;

/// Helper macro for extracting shape from tensor value using GPU
macro_rules! extract_shape {
    ($self:expr, $value:expr) => {
        match $value {
            Value::IntArray(ref arr) => {
                // Direct conversion from IntArray to shape
                arr.iter().map(|&v| v as usize).collect()
            }
            Value::FloatArray(ref arr) => {
                // Direct conversion from FloatArray to shape
                arr.iter().map(|&v| v as usize).collect()
            }
            Value::TensorF32(ref t) => {
                // eprintln!("[DEBUG] extract_shape!: TensorF32 branch");
                // Transfer entire tensor from GPU to CPU at once (faster than per-element reads)
                // eprintln!("[DEBUG] extract_shape!: Calling to_cpu_vec()...");
                let data = t.buffer().to_cpu_vec();
                // eprintln!("[DEBUG] extract_shape!: to_cpu_vec() completed, data.len={}", data.len());
                // Convert f32 values to usize shape dimensions
                let shape: Vec<usize> = data.iter().map(|&v| v as usize).collect();
                // eprintln!("[DEBUG] extract_shape!: shape={:?}", shape);
                shape
            }
            Value::TensorF16(ref t) => {
                // eprintln!("[DEBUG] extract_shape!: TensorF16 branch");
                // Transfer entire tensor from GPU to CPU at once (faster than per-element reads)
                // eprintln!("[DEBUG] extract_shape!: Calling to_cpu_vec()...");
                let data = t.buffer().to_cpu_vec();
                // eprintln!("[DEBUG] extract_shape!: to_cpu_vec() completed, data.len={}", data.len());
                // Convert f16 values to usize shape dimensions
                let shape: Vec<usize> = data.iter().map(|&v| v.to_f32() as usize).collect();
                // eprintln!("[DEBUG] extract_shape!: shape={:?}", shape);
                shape
            }
            _ => return Err(RuntimeError::TypeError(
                format!("Expected shape as tensor or array")
            ))
        }
    };
}

/// Helper macro for creating typed zeros/ones tensors
macro_rules! impl_tensor_fill {
    ($fn_name:ident, $type:ty, $value_variant:ident, $fill_value:expr, $op_name:expr) => {
        pub(super) fn $fn_name(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value>
        {
            if args.len() != 1 {
                return Err(RuntimeError::TypeError(
                    format!("{}() expects 1 argument (shape array), got {}", $op_name, args.len())
                ));
            }

            let shape_val = self.eval_expr(&args[0])?;
            let device = self.env.metal_device();
            let shape = extract_shape!(self, shape_val);

            let numel: usize = shape.iter().product();
            let data = vec![$fill_value; numel];
            let tensor = Tensor::<$type>::from_vec_gpu(device, data, shape)
                .map_err(|e| RuntimeError::TensorError(e))?;

            Ok(Value::$value_variant(Arc::new(tensor)))
        }
    };
}

/// Helper macro for creating typed arange tensors
macro_rules! impl_arange {
    ($fn_name:ident, $type:ty, $value_variant:ident, $convert_fn:expr) => {
        pub(super) fn $fn_name(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value>
        {
            let (start, end) = match args.len() {
                1 => {
                    // arange(n) form
                    let end_val = self.eval_expr(&args[0])?;
                    let end = match end_val {
                        Value::Float(f) => f as i32,
                        Value::Integer(i) => i as i32,
                        Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i32,
                        Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i32,
                        _ => return Err(RuntimeError::TypeError(
                            "arange() expects scalar end value".to_string()
                        )),
                    };
                    (0, end)
                }
                2 => {
                    // arange(start, end) form
                    let start_val = self.eval_expr(&args[0])?;
                    let end_val = self.eval_expr(&args[1])?;

                    let start = match start_val {
                        Value::Float(f) => f as i32,
                        Value::Integer(i) => i as i32,
                        Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i32,
                        Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i32,
                        _ => return Err(RuntimeError::TypeError(
                            "arange() expects scalar start value".to_string()
                        )),
                    };

                    let end = match end_val {
                        Value::Float(f) => f as i32,
                        Value::Integer(i) => i as i32,
                        Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i32,
                        Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i32,
                        _ => return Err(RuntimeError::TypeError(
                            "arange() expects scalar end value".to_string()
                        )),
                    };

                    (start, end)
                }
                _ => return Err(RuntimeError::TypeError(
                    format!("arange() expects 1 or 2 arguments, got {}", args.len())
                )),
            };

            if end <= start {
                return Err(RuntimeError::TypeError(
                    format!("arange() requires end ({}) > start ({})", end, start)
                ));
            }

            // Create sequence [start, start+1, ..., end-1]
            let count = (end - start) as usize;
            let data: Vec<$type> = (start..end).map(|i| $convert_fn(i)).collect();

            // Create 1D tensor on Metal device
            let device = self.env.metal_device();
            let tensor = Tensor::<$type>::from_vec_gpu(device, data, vec![count])
                .map_err(|e| RuntimeError::TensorError(e))?;

            Ok(Value::$value_variant(Arc::new(tensor)))
        }
    };
}

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
            "precompute_rope_cos" => Some(self.eval_precompute_rope_cos(args)),
            "precompute_rope_sin" => Some(self.eval_precompute_rope_sin(args)),
            "slice" => Some(self.eval_slice(args)),
            "slice_last" => Some(self.eval_slice_last(args)),

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
            "arange" => Some(self.eval_range(args)),
            "range_i" => Some(self.eval_range_i(args)),
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

    /// shape(tensor) -> ShapeDims
    /// Returns shape dimensions as CPU-side vector for instant access
    /// No GPU allocation or synchronization required
    fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("shape() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        Ok(match val {
            Value::TensorF16(tensor) => {
                let dims = tensor.dims().to_vec();
                Value::ShapeDims(dims)
            }
            Value::TensorF32(tensor) => {
                let dims = tensor.dims().to_vec();
                Value::ShapeDims(dims)
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
    }

    /// ones(shape) -> tensor
    /// Creates a tensor filled with ones
    /// Returns f32 tensor if shape is f32 array, f16 tensor if shape is f16 array
    fn eval_ones(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("ones() expects 1 argument (shape array), got {}", args.len())
            ));
        }

        // Evaluate shape argument - should be an array literal like [4, 2048]
        let shape_val = self.eval_expr(&args[0])?;

        // Extract shape and determine precision from array literal
        let device = self.env.metal_device();
        match shape_val {
            Value::TensorF32(ref t) => {
                // f32 shape array -> create f32 ones tensor
                // OPTIMIZATION: Transfer entire shape tensor once instead of element-by-element
                let num_dims = t.dims()[0];
                let shape_cpu = t.buffer().to_cpu_vec();  // Single GPU sync
                let mut shape = Vec::with_capacity(num_dims);
                for i in 0..num_dims {
                    shape.push(shape_cpu[i] as usize);
                }
                let numel: usize = shape.iter().product();
                let ones_data = vec![1.0f32; numel];
                let tensor = Tensor::from_vec_gpu(device, ones_data, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(tensor)))
            }
            Value::TensorF16(ref t) => {
                // f16 shape array -> create f16 ones tensor
                // OPTIMIZATION: Transfer entire shape tensor once instead of element-by-element
                let num_dims = t.dims()[0];
                let shape_cpu = t.buffer().to_cpu_vec();  // Single GPU sync
                let mut shape = Vec::with_capacity(num_dims);
                for i in 0..num_dims {
                    shape.push(shape_cpu[i].to_f32() as usize);
                }
                let numel: usize = shape.iter().product();
                let ones_data = vec![f16::ONE; numel];
                let tensor = Tensor::from_vec_gpu(device, ones_data, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(tensor)))
            }
            _ => Err(RuntimeError::TypeError(
                format!("ones() expects shape as array, got {:?}", shape_val)
            ))
        }
    }

    /// reshape(tensor, new_shape) -> tensor
    /// Reshapes a tensor to a new shape
    fn eval_reshape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;
        let _fn_start = std::time::Instant::now();

        // eprintln!("[DEBUG] eval_reshape: Entry, args.len={}", args.len());

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("reshape() expects 2 arguments (tensor, new_shape), got {}", args.len())
            ));
        }

        // eprintln!("[DEBUG] eval_reshape: Evaluating arg[0] (tensor)...");
        let arg0_start = std::time::Instant::now();
        let tensor_val = self.eval_expr(&args[0])?;
        // eprintln!("[DEBUG] eval_reshape: arg[0] evaluated in {:.3}ms", arg0_start.elapsed().as_secs_f64() * 1000.0);

        // eprintln!("[DEBUG] eval_reshape: Evaluating arg[1] (shape)...");
        let arg1_start = std::time::Instant::now();
        let shape_val = self.eval_expr(&args[1])?;
        // eprintln!("[DEBUG] eval_reshape: arg[1] evaluated in {:.3}ms, extracting shape...", arg1_start.elapsed().as_secs_f64() * 1000.0);

        let extract_start = std::time::Instant::now();
        let new_shape = extract_shape!(self, shape_val);
        // eprintln!("[DEBUG] eval_reshape: Shape extracted in {:.3}ms: {:?}", extract_start.elapsed().as_secs_f64() * 1000.0, new_shape);

        let reshape_start = std::time::Instant::now();
        let result = match tensor_val {
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
        };
        // eprintln!("[DEBUG] eval_reshape: reshape() call completed in {:.3}ms", reshape_start.elapsed().as_secs_f64() * 1000.0);
        // eprintln!("[DEBUG] eval_reshape: TOTAL function time: {:.3}ms", _fn_start.elapsed().as_secs_f64() * 1000.0);

        Ok(result)
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
            Value::IntArray(ref arr) => {
                arr.iter().map(|&v| v as usize).collect()
            }
            Value::FloatArray(ref arr) => {
                arr.iter().map(|&v| v as usize).collect()
            }
            Value::TensorF16(ref t) => {
                let mut dims = Vec::with_capacity(t.dims()[0]);
                for i in 0..t.dims()[0] {
                    dims.push(self.read_element_f16(t, i)? as usize);
                }
                dims
            }
            Value::TensorF32(ref t) => {
                let mut dims = Vec::with_capacity(t.dims()[0]);
                for i in 0..t.dims()[0] {
                    dims.push(self.read_element_f32(t, i)? as usize);
                }
                dims
            }
            _ => return Err(RuntimeError::TypeError(
                format!("broadcast_to() expects target_shape as tensor or array, got {:?}", shape_val)
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as usize,
            Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as usize,
            _ => return Err(RuntimeError::TypeError(
                format!("concat() expects dim as scalar, got {:?}", dim_val)
            )),
        };

        // Process based on tensor types
        match (val1, val2) {
            (Value::TensorF16(tensor1), Value::TensorF16(tensor2)) => {
                // Dereference Arc to get &Tensor for concat
                let tensors = vec![tensor1.as_ref(), tensor2.as_ref()];
                let output = crate::tensor::Tensor::concat(&tensors[..], dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(output.to_value())
            }
            (Value::TensorF32(tensor1), Value::TensorF32(tensor2)) => {
                // Dereference Arc to get &Tensor for concat
                let tensors = vec![tensor1.as_ref(), tensor2.as_ref()];
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
    /// // Old style (自動計算):
    /// let Q_rope = rope(Q_heads)  // Apply rotary position embedding with position_offset=0
    /// let Q_rope = rope(Q_heads, 29)  // Apply RoPE starting at position 29 (for KV cache)
    ///
    /// // Candle style (事前計算されたcos/sin使用):
    /// let (cos, sin) = precompute_rope_freqs(64.0, 2048.0, 10000.0)
    /// let Q_rope = rope(Q_heads, cos, sin)
    /// ```
    fn eval_rope(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;
        let _start = std::time::Instant::now();

        if args.len() < 1 || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("rope() expects 1-3 arguments (tensor, [position_offset] OR tensor, cos, sin), got {}", args.len())
            ));
        }

        // Check if this is Candle-style (3 arguments: tensor, cos, sin)
        if args.len() == 3 {
            return self.eval_rope_candle_style(args);
        }

        let tensor_val = self.eval_expr(&args[0])?;

        // Get position offset (default 0 for backward compatibility)
        let position_offset = if args.len() == 2 {
            let offset_val = self.eval_expr(&args[1])?;
            match offset_val {
                Value::Float(f) => f as usize,
                Value::Integer(i) => i as usize,
                Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
                Value::TensorF32(ref t) if t.numel() == 1 => self.read_element_f32(t, 0)? as usize,
                Value::TokenIdArray(ref arr) if arr.len() == 1 => arr.get(0).unwrap() as usize,
                _ => return Err(RuntimeError::TypeError(
                    "rope() position_offset must be a scalar integer".to_string()
                )),
            }
        } else {
            0  // Default to 0 if not provided
        };

        let result = match tensor_val {
            Value::TensorF16(tensor) => {
                Ok(tensor.rope(position_offset).map_err(|e| RuntimeError::TensorError(e))?.to_value())
            }
            Value::TensorF32(tensor) => {
                Ok(tensor.rope(position_offset).map_err(|e| RuntimeError::TensorError(e))?.to_value())
            }
            _ => Err(RuntimeError::TypeError("Expected tensor".to_string()))
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] rope({}, pos={}): {:.3}ms", dtype, position_offset, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    /// Candle-style RoPE with precomputed cos/sin arrays
    fn eval_rope_candle_style(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        // args[0]: input tensor [seq_len, n_heads, head_dim]
        // args[1]: cos array [max_seq_len, head_dim/2]
        // args[2]: sin array [max_seq_len, head_dim/2]

        let tensor_val = self.eval_expr(&args[0])?;
        let cos_val = self.eval_expr(&args[1])?;
        let sin_val = self.eval_expr(&args[2])?;

        match (tensor_val, cos_val, sin_val) {
            (Value::TensorF16(tensor), Value::TensorF16(cos), Value::TensorF16(sin)) => {
                let result = tensor.rope_candle(&cos, &sin)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            (Value::TensorF32(tensor), Value::TensorF32(cos), Value::TensorF32(sin)) => {
                let result = tensor.rope_candle(&cos, &sin)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            (Value::TensorF16(tensor), Value::TensorF32(cos), Value::TensorF32(sin)) => {
                // Convert f32 cos/sin to f16
                let cos_f16 = cos.to_f16().map_err(|e| RuntimeError::TensorError(e))?;
                let sin_f16 = sin.to_f16().map_err(|e| RuntimeError::TensorError(e))?;
                let result = tensor.rope_candle(&cos_f16, &sin_f16)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            _ => Err(RuntimeError::TypeError(
                "rope() with cos/sin requires tensors of compatible types".to_string()
            ))
        }
    }

    /// precompute_rope_cos(head_dim, max_seq_len, rope_base) -> cos_tensor
    /// Precompute RoPE cosine embeddings (Candle-style)
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension (must be even, e.g., 64)
    /// * `max_seq_len` - Maximum sequence length (e.g., 2048)
    /// * `rope_base` - RoPE base frequency (typically 10000.0)
    ///
    /// # Returns
    /// cos_tensor with shape [max_seq_len, head_dim/2]
    ///
    /// # Example
    /// ```ignore
    /// let cos = precompute_rope_cos(64.0, 2048.0, 10000.0)
    /// let sin = precompute_rope_sin(64.0, 2048.0, 10000.0)
    /// let Q_rope = rope(Q_heads, cos, sin)
    /// ```
    fn eval_precompute_rope_cos(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        self.eval_precompute_rope_impl(args, true)
    }

    /// precompute_rope_sin(head_dim, max_seq_len, rope_base) -> sin_tensor
    /// Precompute RoPE sine embeddings (Candle-style)
    fn eval_precompute_rope_sin(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        self.eval_precompute_rope_impl(args, false)
    }

    /// Internal implementation for precomputing RoPE cos/sin
    fn eval_precompute_rope_impl(&mut self, args: &[TensorExpr], compute_cos: bool) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("precompute_rope_freqs() expects 3 arguments (head_dim, max_seq_len, rope_base), got {}", args.len())
            ));
        }

        // Parse arguments
        let head_dim_val = self.eval_expr(&args[0])?;
        let head_dim = match head_dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("head_dim must be a number".to_string())),
        };

        let max_seq_len_val = self.eval_expr(&args[1])?;
        let max_seq_len = match max_seq_len_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("max_seq_len must be a number".to_string())),
        };

        let rope_base_val = self.eval_expr(&args[2])?;
        let rope_base = match rope_base_val {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            _ => return Err(RuntimeError::TypeError("rope_base must be a number".to_string())),
        };

        if head_dim % 2 != 0 {
            return Err(RuntimeError::TypeError(
                format!("head_dim must be even, got {}", head_dim)
            ));
        }

        // Compute frequency for each dimension pair: freq[i] = 1 / rope_base^(i/head_dim)
        // Following Candle's implementation
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / rope_base.powf(i as f32 / head_dim as f32))
            .collect();

        // Compute cos/sin for each (position, dimension) combination
        // Following Candle: duplicate each frequency value for the pair
        // cos/sin arrays have shape [max_seq_len, head_dim]
        let half_dim = head_dim / 2;
        let mut cos_data = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for dim_pair_idx in 0..half_dim {
                let angle = pos as f32 * theta[dim_pair_idx];
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                // Duplicate for both elements of the pair
                cos_data.push(cos_val);
                cos_data.push(cos_val);
                sin_data.push(sin_val);
                sin_data.push(sin_val);
            }
        }

        // Create tensor on Metal device
        let device = self.env.metal_device();

        if compute_cos {
            let cos_f32 = Tensor::<f32>::from_vec_gpu(
                device,
                cos_data,
                vec![max_seq_len, head_dim]
            ).map_err(|e| RuntimeError::TensorError(e))?;

            // Convert to f16 for consistency with model weights
            let cos_f16 = cos_f32.to_f16().map_err(|e| RuntimeError::TensorError(e))?;
            Ok(cos_f16.to_value())
        } else {
            let sin_f32 = Tensor::<f32>::from_vec_gpu(
                device,
                sin_data,
                vec![max_seq_len, head_dim]
            ).map_err(|e| RuntimeError::TensorError(e))?;

            // Convert to f16 for consistency with model weights
            let sin_f16 = sin_f32.to_f16().map_err(|e| RuntimeError::TensorError(e))?;
            Ok(sin_f16.to_value())
        }
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
        let tensor = tensor_val.as_tensor_f16()?;

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
                self.read_element_f16(t, 0)? as usize
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
                self.read_element_f16(t, 0)? as usize
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
                self.read_element_f16(t, 0)? as usize
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
        let data = tensor.sync_and_read();

        // Extract the slice
        let offset = row * cols + col_start;
        let length = col_end - col_start;
        let slice_data: Vec<f16> = data[offset..offset + length].to_vec();

        // Create 1D tensor with the slice
        let device = tensor.device().clone();
        let result_tensor = match &device {
            crate::device::Device::Metal(metal_device) => {
                Tensor::from_vec_gpu(metal_device, slice_data, vec![length])
            }
            crate::device::Device::CPU => {
                Tensor::from_vec(slice_data, vec![length])
            }
            crate::device::Device::NeuralEngine => {
                // Fallback to CPU for NeuralEngine
                Tensor::from_vec(slice_data, vec![length])
            }
        }.map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(Arc::new(result_tensor)))
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
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
        let tensor = array_val.as_tensor_f16()?;
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
        let data = tensor.sync_and_read();
        let slice_data: Vec<_> = data[start..end].to_vec();
        let length = end - start;

        // Create result tensor on same device
        let result_tensor = match tensor.device() {
            crate::device::Device::Metal(metal_device) => {
                Tensor::from_vec_gpu(metal_device, slice_data, vec![length])
            }
            crate::device::Device::CPU => {
                Tensor::from_vec(slice_data, vec![length])
            }
            crate::device::Device::NeuralEngine => {
                Tensor::from_vec(slice_data, vec![length])
            }
        }.map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(Arc::new(result_tensor)))
    }

    /// add(tensor1, tensor2) -> tensor
    /// Adds two tensors element-wise (supports method chaining)
    fn eval_add_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let _start = std::time::Instant::now();
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("add() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        let result = match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.add(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.add(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "add() requires two tensors of the same type".to_string()
            ))
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] add({}): {:.3}ms", dtype, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
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
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.sub(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "sub() requires two tensors of the same type".to_string()
            ))
        }
    }

    /// mul(tensor1, tensor2_or_scalar) -> tensor
    /// Multiplies tensors element-wise or by scalar (supports method chaining)
    fn eval_mul_method(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let _start = std::time::Instant::now();
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("mul() expects 2 arguments, got {}", args.len())
            ));
        }

        let left = self.eval_expr(&args[0])?;
        let right = self.eval_expr(&args[1])?;

        let result = match (left, right) {
            (Value::TensorF16(t1), Value::TensorF16(t2)) => {
                let result = t1.mul(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.mul(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            (Value::TensorF16(t), Value::Float(scalar)) | (Value::Float(scalar), Value::TensorF16(t)) => {
                let result = t.mul_scalar(f16::from_f32(scalar as f32))
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
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
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] mul({}): {:.3}ms", dtype, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
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
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(t1), Value::TensorF32(t2)) => {
                let result = t1.div(&t2).map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
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
    /// Returns f32 tensor if shape is f32 array, f16 tensor if shape is f16 array
    fn eval_zeros(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("zeros() expects 1 argument (shape array), got {}", args.len())
            ));
        }

        // Evaluate shape argument
        let shape_val = self.eval_expr(&args[0])?;

        // Extract shape and determine precision from array literal
        let device = self.env.metal_device();
        match shape_val {
            Value::TensorF32(ref t) => {
                // f32 shape array -> create f32 zeros tensor
                let data = t.sync_and_read();
                let shape = data.iter().map(|&v| v as usize).collect::<Vec<_>>();
                let numel: usize = shape.iter().product();
                let zeros_data = vec![0.0f32; numel];
                let tensor = Tensor::from_vec_gpu(device, zeros_data, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(tensor)))
            }
            Value::TensorF16(ref t) => {
                // f16 shape array -> create f16 zeros tensor
                let data = t.sync_and_read_f32();
                let shape = data.iter().map(|&v| v as usize).collect::<Vec<_>>();
                let numel: usize = shape.iter().product();
                let zeros_data = vec![f16::ZERO; numel];
                let tensor = Tensor::from_vec_gpu(device, zeros_data, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(tensor)))
            }
            _ => Err(RuntimeError::TypeError(
                format!("zeros() expects shape as array, got {:?}", shape_val)
            ))
        }
    }

    // Use macro to implement zeros/ones for f32 and f16
    impl_tensor_fill!(eval_zeros_f32, f32, TensorF32, 0.0f32, "f32::zeros");
    impl_tensor_fill!(eval_zeros_f16, f16, TensorF16, f16::ZERO, "f16::zeros");
    impl_tensor_fill!(eval_ones_f32, f32, TensorF32, 1.0f32, "f32::ones");
    impl_tensor_fill!(eval_ones_f16, f16, TensorF16, f16::ONE, "f16::ones");

    // Use macro to implement arange for f32 and f16
    impl_arange!(eval_range_f32, f32, TensorF32, |i: i32| i as f32);
    impl_arange!(eval_range_f16, f16, TensorF16, |i: i32| f16::from_f32(i as f32));

    /// Legacy arange function (calls f32 version for backward compatibility)
    pub(super) fn eval_range(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        self.eval_range_f32(args)
    }

    /// range_i(n) -> IntArray
    /// range_i(start, end) -> IntArray
    /// Creates an integer array [start, start+1, ..., end-1]
    /// This is useful for loop indices where integer values are required
    pub(super) fn eval_range_i(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let (start, end) = match args.len() {
            1 => {
                // range_i(n) form
                let end_val = self.eval_expr(&args[0])?;
                let end = match end_val {
                    Value::Float(f) => f as i64,
                    Value::Integer(i) => i,
                    Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i64,
                    Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i64,
                    _ => return Err(RuntimeError::TypeError(
                        "range_i() expects scalar end value".to_string()
                    )),
                };
                (0, end)
            }
            2 => {
                // range_i(start, end) form
                let start_val = self.eval_expr(&args[0])?;
                let end_val = self.eval_expr(&args[1])?;

                let start = match start_val {
                    Value::Float(f) => f as i64,
                    Value::Integer(i) => i,
                    Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i64,
                    Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i64,
                    _ => return Err(RuntimeError::TypeError(
                        "range_i() expects scalar start value".to_string()
                    )),
                };

                let end = match end_val {
                    Value::Float(f) => f as i64,
                    Value::Integer(i) => i,
                    Value::TensorF16(ref t) if t.numel() == 1 => self.tensor_f16_to_scalar(t)? as i64,
                    Value::TensorF32(ref t) if t.numel() == 1 => self.tensor_f32_to_scalar(t)? as i64,
                    _ => return Err(RuntimeError::TypeError(
                        "range_i() expects scalar end value".to_string()
                    )),
                };

                (start, end)
            }
            _ => return Err(RuntimeError::TypeError(
                format!("range_i() expects 1 or 2 arguments, got {}", args.len())
            ))
        };

        // Generate integer sequence
        let values: Vec<i64> = (start..end).collect();
        Ok(Value::IntArray(values))
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
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let numel = tensor.numel();
                let result = tensor.reshape(vec![numel])
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
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
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.squeeze(dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
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
                Ok(Value::TensorF16(Arc::new(result)))
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
                Ok(Value::TensorF32(Arc::new(result)))
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
                let data = tensor.sync_and_read();
                let max_val = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(max_val.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let data = tensor.sync_and_read();
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
                let data = tensor.sync_and_read();
                let min_val = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RuntimeError::TensorError(TensorError::InvalidOperation("Empty tensor".to_string())))?;
                Ok(Value::Float(min_val.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
                let data = tensor.sync_and_read();
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
                    let data = result.sync_and_read();
                    Ok(Value::Integer(data[0].to_f32() as i64))
                } else {
                    Ok(Value::TensorF16(Arc::new(result)))
                }
            }
            Value::TensorF32(tensor) => {
                let result = tensor.argmax(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.sync_and_read_f32();
                    Ok(Value::Integer(data[0] as i64))
                } else {
                    Ok(Value::TensorF32(Arc::new(result)))
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
                    let data = result.sync_and_read();
                    Ok(Value::Integer(data[0].to_f32() as i64))
                } else {
                    Ok(Value::TensorF16(Arc::new(result)))
                }
            }
            Value::TensorF32(tensor) => {
                let result = tensor.argmin(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // If no dim specified, return as integer (backward compatibility)
                if dim.is_none() && !keepdim {
                    let data = result.sync_and_read_f32();
                    Ok(Value::Integer(data[0] as i64))
                } else {
                    Ok(Value::TensorF32(Arc::new(result)))
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
                let data = t.sync_and_read_f32();
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
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.permute(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "gather() dim must be a scalar".to_string()
            )),
        };

        match (val, index_val) {
            (Value::TensorF16(tensor), Value::TensorF16(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF16(tensor), Value::TensorF32(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(tensor), Value::TensorF16(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            (Value::TensorF32(tensor), Value::TensorF32(index)) => {
                let result = tensor.gather(dim, &index)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "gather() requires tensor and index as tensors (f16 or f32)".to_string()
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "scatter() dim must be a scalar".to_string()
            )),
        };

        // scatter now supports both f16 and f32 indices
        match (val, src_val, index_val) {
            (Value::TensorF16(tensor), Value::TensorF16(src), Value::TensorF16(index)) => {
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF16(tensor), Value::TensorF16(src), Value::TensorF32(index)) => {
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(tensor), Value::TensorF32(src), Value::TensorF16(index)) => {
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            (Value::TensorF32(tensor), Value::TensorF32(src), Value::TensorF32(index)) => {
                let result = tensor.scatter(dim, &index, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "scatter() requires tensor, src, and index all as tensors (f16 or f32)".to_string()
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "chunk() chunks must be a scalar".to_string()
            )),
        };

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "chunk() dim must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.as_ref().chunk(chunks, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                // Return as array of tensors
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF16(Arc::new(t)))
                    .collect();
                // For now, return the first chunk as we don't have array support
                // TODO: Implement proper array return type
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF16(Arc::clone(&tensor))))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.as_ref().chunk(chunks, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF32(Arc::new(t)))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF32(Arc::clone(&tensor))))
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
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "split() split_size must be a scalar".to_string()
            )),
        };

        let dim = match dim_val {
            Value::Float(f) => f as usize,
            Value::Integer(i) => i as usize,
            Value::TensorF16(ref t) if t.numel() == 1 => self.read_element_f16(t, 0)? as usize,
            _ => return Err(RuntimeError::TypeError(
                "split() dim must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.as_ref().split(split_size, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF16(Arc::new(t)))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF16(Arc::clone(&tensor))))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.as_ref().split(split_size, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let tensor_values: Vec<Value> = result.into_iter()
                    .map(|t| Value::TensorF32(Arc::new(t)))
                    .collect();
                Ok(tensor_values.into_iter().next().unwrap_or(Value::TensorF32(Arc::clone(&tensor))))
            }
            _ => Err(RuntimeError::TypeError(
                "split() expects a tensor".to_string()
            ))
        }
    }

    /// Helper: Extract scalar f32 value from 1-element f32 tensor using GPU
    pub(super) fn tensor_f32_to_scalar(&self, tensor: &Tensor<f32>) -> RuntimeResult<f32> {
        if tensor.numel() != 1 {
            return Err(RuntimeError::InvalidOperation(
                format!("Expected 1-element tensor, got {} elements", tensor.numel())
            ));
        }
        self.read_element_f32(tensor, 0)
    }

    /// Helper: Extract scalar f32 value from 1-element f16 tensor using GPU
    pub(super) fn tensor_f16_to_scalar(&self, tensor: &Tensor<half::f16>) -> RuntimeResult<f32> {
        if tensor.numel() != 1 {
            return Err(RuntimeError::InvalidOperation(
                format!("Expected 1-element tensor, got {} elements", tensor.numel())
            ));
        }
        self.read_element_f16(tensor, 0)
    }

    /// create_cache_tensor(max_length, shape) -> tensor
    /// Create pre-allocated cache tensor for efficient KV cache updates
    ///
    /// # Arguments
    /// * max_length - Maximum capacity along dimension 0
    /// * shape - Remaining dimensions as tensor [num_heads, head_dim]
    ///
    /// # Returns
    /// Pre-allocated tensor with shape [0, ...shape], but buffer capacity for max_length
    ///
    /// # Example
    /// ```tl
    /// let kv_cache = create_cache_tensor(512, shape(4, 128))  # [0, 4, 128] with capacity 512
    /// ```
    pub(super) fn eval_create_cache_tensor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("create_cache_tensor() expects 2 arguments (max_length, shape), got {}", args.len())
            ));
        }

        // Evaluate max_length argument
        let max_length_val = self.eval_expr(&args[0])?;
        let max_length = match max_length_val {
            Value::Integer(n) => n as usize,
            Value::Float(f) => f as usize,
            Value::TensorF32(ref t) if t.numel() == 1 => {
                self.tensor_f32_to_scalar(t)? as usize
            }
            Value::TensorF16(ref t) if t.numel() == 1 => {
                self.tensor_f16_to_scalar(t)? as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("create_cache_tensor() expects max_length as scalar, got {:?}", max_length_val.type_name())
            ))
        };

        // Evaluate shape argument
        let shape_val = self.eval_expr(&args[1])?;
        let shape_rest = extract_shape!(self, shape_val);

        // Get device
        let device = Device::Metal(self.env.metal_device().clone());

        // Create cache tensor (use f32 by default for now, could be parameterized)
        let result = Tensor::<f32>::create_cache_tensor(max_length, &shape_rest, &device)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(result.to_value())
    }

    /// append_cache(cache, new_data) -> tensor
    /// Append data to pre-allocated cache tensor (zero-copy, in-place)
    ///
    /// # Arguments
    /// * cache - Pre-allocated cache tensor from create_cache_tensor
    /// * new_data - Data to append, shape [new_len, ...same_dims]
    ///
    /// # Returns
    /// New view with updated shape [current_len + new_len, ...], sharing same buffer
    ///
    /// # Example
    /// ```tl
    /// let kv_cache = create_cache_tensor(512, shape(4, 128))
    /// let new_kv = zeros(shape(1, 4, 128))
    /// kv_cache = append_cache(kv_cache, new_kv)  # Now [1, 4, 128]
    /// ```
    pub(super) fn eval_append_cache(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("append_cache() expects 2 arguments (cache, new_data), got {}", args.len())
            ));
        }

        // Evaluate arguments
        let cache_val = self.eval_expr(&args[0])?;
        let new_data_val = self.eval_expr(&args[1])?;

        // Type-based dispatch
        match (cache_val, new_data_val) {
            (Value::TensorF32(cache), Value::TensorF32(new_data)) => {
                let result = Tensor::append_cache(&cache, &new_data)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            (Value::TensorF16(cache), Value::TensorF16(new_data)) => {
                let result = Tensor::append_cache(&cache, &new_data)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            _ => Err(RuntimeError::TypeError(
                "append_cache() requires both tensors to be the same type (both f16 or both f32)".to_string()
            ))
        }
    }

    /// slice_last(tensor, axis) -> tensor
    /// Extract the last slice along the specified axis
    ///
    /// Uses GPU implementation when available to avoid CPU sync bottleneck.
    ///
    /// # Arguments
    /// * tensor - Input tensor
    /// * axis - Axis along which to take the last slice (0 for rows, 1 for columns, etc.)
    ///
    /// # Returns
    /// Tensor with one fewer dimension
    ///
    /// # Example
    /// ```tl
    /// let x = zeros(shape(35, 2048))
    /// let last_row = slice_last(x, 0)  # [35, 2048] → [2048]
    /// ```
    pub(super) fn eval_slice_last(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("slice_last() expects 2 arguments (tensor, axis), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;

        // Evaluate axis argument
        let axis_val = self.eval_expr(&args[1])?;
        let axis = match axis_val {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            Value::TensorF32(ref t) if t.numel() == 1 => {
                self.tensor_f32_to_scalar(t)? as usize
            }
            Value::TensorF16(ref t) if t.numel() == 1 => {
                self.tensor_f16_to_scalar(t)? as usize
            }
            _ => return Err(RuntimeError::TypeError(
                format!("slice_last() expects axis as scalar, got {:?}", axis_val.type_name())
            ))
        };

        // Type-based dispatch - use GPU implementation via Tensor method
        match tensor_val {
            Value::TensorF32(ref tensor) => {
                let result = tensor.slice_last(axis)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            Value::TensorF16(ref tensor) => {
                let result = tensor.slice_last(axis)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(result.to_value())
            }
            _ => Err(RuntimeError::TypeError(
                format!("slice_last() expects tensor, got {:?}", tensor_val.type_name())
            ))
        }
    }
}
