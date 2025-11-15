//! Math operations for TensorLogic interpreter

use super::*;
use std::sync::Arc;

impl Interpreter {
    pub(super) fn eval_math_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "matmul" => Some(self.eval_matmul(args)),
            "linear" => Some(self.eval_linear(args)),
            "sigmoid" => Some(self.eval_sigmoid(args)),

            // Activation functions (for method chaining)
            "relu" => Some(self.eval_relu(args)),
            "gelu" => Some(self.eval_gelu(args)),
            "tanh" => Some(self.eval_tanh(args)),
            "clamp" => Some(self.eval_clamp(args)),

            // Basic math operations (for method chaining)
            "exp" => Some(self.eval_exp(args)),
            "log" => Some(self.eval_log(args)),
            "sqrt" => Some(self.eval_sqrt(args)),
            "pow" => Some(self.eval_pow(args)),
            "sin" => Some(self.eval_sin(args)),
            "cos" => Some(self.eval_cos(args)),
            "tan" => Some(self.eval_tan(args)),

            // Not yet implemented
            "mean" | "max" | "min" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Math function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// matmul(a, b) -> tensor
    /// Matrix multiplication: a @ b
    fn eval_matmul(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("matmul() expects 2 arguments (a, b), got {}", args.len())
            ));
        }

        // Evaluate both tensor arguments
        let a_val = self.eval_expr(&args[0])?;
        let b_val = self.eval_expr(&args[1])?;

        // Process based on input type (f16 or f32)
        match (a_val, b_val) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let result = a.matmul(&b)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let result = a.matmul(&b)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "matmul() requires both tensors to be same type (both f16 or both f32)".to_string()
            ))
        }
    }

    /// linear(x, weight, bias) -> tensor
    /// Linear transformation: y = x @ weight.T + bias
    /// Automatically transposes weight matrix like PyTorch/Candle
    ///
    /// Args:
    ///   x: input tensor [batch, in_features] or [in_features]
    ///   weight: weight matrix [out_features, in_features] (GGUF format after reverse)
    ///   bias: optional bias vector [out_features]
    fn eval_linear(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] eval_linear: ENTRY");
        }
        let _start = std::time::Instant::now();
        if args.len() < 2 || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("linear() expects 2-3 arguments (x, weight, [bias]), got {}", args.len())
            ));
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] eval_linear: Evaluating arg[0]...");
        }
        // Evaluate input tensor
        let x_val = self.eval_expr(&args[0])?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] eval_linear: arg[0] done, evaluating arg[1]...");
        }
        let weight_val = self.eval_expr(&args[1])?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] eval_linear: Both args evaluated");
        }

        // Process based on input type (f16 or f32)
        let result = match (x_val, weight_val) {
            (Value::TensorF16(x), Value::TensorF16(weight)) => {
                // Use fused transpose-matmul: x @ weight.T
                // This is 20-30% faster than separate transpose + matmul
                let mut result = x.matmul_transposed_b(&weight)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // Add bias if provided
                if args.len() == 3 {
                    let bias_val = self.eval_expr(&args[2])?;
                    let bias = match bias_val {
                        Value::TensorF16(ref t) => t,
                        _ => return Err(RuntimeError::TypeError("linear() bias must be f16 tensor".to_string()))
                    };
                    result = result.add(&bias)
                        .map_err(|e| RuntimeError::TensorError(e))?;
                }

                // No clamping applied - matches Candle/PyTorch convention
                // Transformers rely on full dynamic range for Q/K/V projections,
                // attention scores, and FFN intermediate values
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(x), Value::TensorF32(weight)) => {
                // DEBUG: Check for final output layer (vocab_size x hidden_dim)
                if weight.dims()[0] > 30000 && std::env::var("TL_DEBUG").is_ok() {
                    use crate::tensor::TensorIO;
                    eprintln!("[DEBUG linear] Large weight matrix detected: {:?}", weight.dims());
                    eprintln!("[DEBUG linear] Input x dims: {:?}", x.dims());

                    let x_data = x.sync_and_read();
                    let weight_data = weight.sync_and_read();

                    let x_mean = x_data.iter().sum::<f32>() / x_data.len() as f32;
                    let x_abs_max = x_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let w_mean = weight_data.iter().sum::<f32>() / weight_data.len() as f32;
                    let w_abs_max = weight_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

                    eprintln!("[DEBUG linear] Input: mean={:.6}, abs_max={:.6}, first_10={:?}",
                        x_mean, x_abs_max, &x_data[..10.min(x_data.len())]);
                    eprintln!("[DEBUG linear] Weight: mean={:.6}, abs_max={:.6}, first_10={:?}",
                        w_mean, w_abs_max, &weight_data[..10.min(weight_data.len())]);
                }

                // Use fused transpose-matmul: x @ weight.T
                // This is 20-30% faster than separate transpose + matmul
                let mut result = x.matmul_transposed_b(&weight)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // DEBUG: Check result
                if weight.dims()[0] > 30000 && std::env::var("TL_DEBUG").is_ok() {
                    use crate::tensor::TensorIO;
                    let result_data = result.sync_and_read();
                    let r_mean = result_data.iter().sum::<f32>() / result_data.len() as f32;
                    let r_abs_max = result_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    eprintln!("[DEBUG linear] Result: mean={:.6}, abs_max={:.6}, first_10={:?}",
                        r_mean, r_abs_max, &result_data[..10.min(result_data.len())]);
                }

                // Add bias if provided
                if args.len() == 3 {
                    let bias_val = self.eval_expr(&args[2])?;
                    let bias = match bias_val {
                        Value::TensorF32(ref t) => t,
                        _ => return Err(RuntimeError::TypeError("linear() bias must be f32 tensor".to_string()))
                    };
                    result = result.add(&bias)
                        .map_err(|e| RuntimeError::TensorError(e))?;
                }

                // DEBUG: Log buffer pointer BEFORE returning
                if weight.dims()[0] > 30000 && std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    use crate::tensor::TensorAccessors;
                    if let Ok(metal_buf) = result.buffer().as_metal() {
                        let ptr = metal_buf.buffer.contents() as *const f32;
                        if !ptr.is_null() {
                            let slice = unsafe { std::slice::from_raw_parts(ptr, 10) };
                            eprintln!(
                                "[DEBUG linear] BEFORE RETURN: buffer_ptr={:p}, first_10={:?}",
                                metal_buf.buffer.as_ref(), slice
                            );
                        }
                    }
                }

                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "linear() requires x and weight to be same type (both f16 or both f32)".to_string()
            )),
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] linear({}): {:.3}ms", dtype, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    /// sigmoid(x) -> tensor
    /// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    fn eval_sigmoid(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;
        let _start = std::time::Instant::now();

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("sigmoid() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        let result = match tensor_val {
            Value::TensorF16(t) => Ok(t.sigmoid().map_err(|e| RuntimeError::TensorError(e))?.to_value()),
            Value::TensorF32(t) => Ok(t.sigmoid().map_err(|e| RuntimeError::TensorError(e))?.to_value()),
            _ => Err(RuntimeError::TypeError("Expected tensor".to_string()))
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] sigmoid({}): {:.3}ms", dtype, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    /// relu(tensor) -> tensor
    /// Applies ReLU activation function (for method chaining)
    fn eval_relu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("relu() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.relu()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.relu()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "relu() expects a tensor".to_string()
            ))
        }
    }

    /// gelu(tensor) -> tensor
    /// Applies GELU activation function (for method chaining)
    fn eval_gelu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let _start = std::time::Instant::now();
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("gelu() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        let result = match val {
            Value::TensorF16(tensor) => {
                let result = tensor.gelu()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.gelu()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "gelu() expects a tensor".to_string()
            ))
        };

        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!("[PERF] gelu({}): {:.3}ms", dtype, _start.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    /// tanh(tensor) -> tensor
    /// Applies Tanh activation function (for method chaining)
    fn eval_tanh(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("tanh() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.tanh()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.tanh()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "tanh() expects a tensor".to_string()
            ))
        }
    }

    /// exp(tensor) -> tensor
    /// Applies exponential function element-wise: e^x
    fn eval_exp(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("exp() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.exp()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.exp()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "exp() expects a tensor".to_string()
            ))
        }
    }

    /// log(tensor) -> tensor
    /// Applies natural logarithm element-wise: ln(x)
    fn eval_log(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("log() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.log()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.log()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "log() expects a tensor".to_string()
            ))
        }
    }

    /// sqrt(tensor) -> tensor
    /// Applies square root element-wise: √x
    fn eval_sqrt(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("sqrt() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.sqrt()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.sqrt()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "sqrt() expects a tensor".to_string()
            ))
        }
    }

    /// pow(tensor, exponent) -> tensor
    /// Raises each element to the power of exponent: x^p
    fn eval_pow(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("pow() expects 2 arguments (tensor, exponent), got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let exp_val = self.eval_expr(&args[1])?;

        let exponent = match exp_val {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            Value::TensorF16(ref t) if t.numel() == 1 => t.sync_and_read_f32()[0],
            _ => return Err(RuntimeError::TypeError(
                "pow() exponent must be a scalar".to_string()
            )),
        };

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.pow(exponent)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.pow(exponent)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "pow() expects a tensor".to_string()
            ))
        }
    }

    /// sin(tensor) -> tensor
    /// Applies sine function element-wise
    fn eval_sin(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("sin() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.sin()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.sin()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "sin() expects a tensor".to_string()
            ))
        }
    }

    /// cos(tensor) -> tensor
    /// Applies cosine function element-wise
    fn eval_cos(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cos() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.cos()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.cos()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "cos() expects a tensor".to_string()
            ))
        }
    }

    /// tan(tensor) -> tensor
    /// Applies tangent function element-wise
    fn eval_tan(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("tan() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;

        match val {
            Value::TensorF16(tensor) => {
                let result = tensor.tan()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.tan()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "tan() expects a tensor".to_string()
            ))
        }
    }

    /// clamp(tensor, min_val, max_val) -> tensor
    /// Restricts tensor values to [min_val, max_val] range
    /// Useful for preventing overflow in attention scores
    fn eval_clamp(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("clamp() expects 3 arguments (tensor, min_val, max_val), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        // Evaluate min_val (can be float or int)
        let min_val = match self.eval_expr(&args[1])? {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            _ => return Err(RuntimeError::TypeError(
                "clamp() min_val must be a number".to_string()
            ))
        };

        // Evaluate max_val (can be float or int)
        let max_val = match self.eval_expr(&args[2])? {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            _ => return Err(RuntimeError::TypeError(
                "clamp() max_val must be a number".to_string()
            ))
        };

        match tensor_val {
            Value::TensorF16(tensor) => {
                let result = tensor.clamp(min_val, max_val)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            Value::TensorF32(tensor) => {
                let result = tensor.clamp(min_val, max_val)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "clamp() expects a tensor as first argument".to_string()
            ))
        }
    }
}
