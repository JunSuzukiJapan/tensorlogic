//! Utility operations for TensorLogic interpreter

use super::*;

impl Interpreter {
    /// Evaluate utility builtin functions
    /// Returns Some(result) if function is handled, None if not in this category
    pub(super) fn eval_util_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "len" => Some(self.eval_len(args)),
            "get" => Some(self.eval_get(args)),
            "append" => Some(self.eval_append(args)),
            "to_int" => Some(self.eval_to_int(args)),
            "str" => Some(self.eval_str(args)),
            "input" => Some(self.eval_input(args)),
            "env" => Some(self.eval_env(args)),
            "cleanup" => Some(self.eval_cleanup(args)),
            "new_tensor_buffer" => Some(self.eval_new_tensor_buffer(args)),
            // Scalar math functions
            "abs" => Some(self.eval_abs(args)),
            "floor" => Some(self.eval_floor(args)),
            "ceil" => Some(self.eval_ceil(args)),
            "round" => Some(self.eval_round(args)),
            "min" => Some(self.eval_min(args)),
            "max" => Some(self.eval_max(args)),
            _ => None,
        }
    }

    /// len(value) -> int
    /// Get length of TokenIds, Tensor, String, Array, or Vec
    fn eval_len(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("len() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        let length = match value {
            Value::TokenIds(ref ids) => ids.len() as i64,
            Value::TensorF16(ref t) => {
                let dims = t.dims();
                dims[0] as i64
            }
            Value::String(ref s) => s.len() as i64,
            Value::IntArray(ref arr) => arr.len() as i64,
            Value::FloatArray(ref arr) => arr.len() as i64,
            Value::StringArray(ref arr) => arr.len() as i64,
            Value::BoolArray(ref arr) => arr.len() as i64,
            Value::IntVec(ref vec) => vec.lock().unwrap().len() as i64,
            Value::FloatVec(ref vec) => vec.lock().unwrap().len() as i64,
            Value::StringVec(ref vec) => vec.lock().unwrap().len() as i64,
            Value::BoolVec(ref vec) => vec.lock().unwrap().len() as i64,
            v => return Err(RuntimeError::TypeError(
                format!("len() argument must be TokenIds, Tensor, String, Array, or Vec, got {:?}", v)
            )),
        };

        Ok(Value::Integer(length))
    }

    /// get(collection, index) -> value
    /// Get element at specific index from TokenIds, Array, or Vec
    fn eval_get(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("get() expects 2 arguments (collection, index), got {}", args.len())
            ));
        }

        let collection = self.eval_expr(&args[0])?;
        let index_val = self.eval_expr(&args[1])?;

        let index = match index_val {
            Value::Integer(i) => i,
            Value::Float(f) => f as i64,
            v => return Err(RuntimeError::TypeError(
                format!("get() index must be a number, got {:?}", v)
            )),
        };

        match collection {
            Value::TokenIds(ids) => {
                let idx = if index < 0 {
                    (ids.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= ids.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: ids.len(),
                    });
                }
                Ok(Value::Integer(ids[idx] as i64))
            }
            Value::IntArray(arr) => {
                let idx = if index < 0 {
                    (arr.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= arr.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: arr.len(),
                    });
                }
                Ok(Value::Integer(arr[idx]))
            }
            Value::FloatArray(arr) => {
                let idx = if index < 0 {
                    (arr.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= arr.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: arr.len(),
                    });
                }
                Ok(Value::Float(arr[idx]))
            }
            Value::StringArray(arr) => {
                let idx = if index < 0 {
                    (arr.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= arr.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: arr.len(),
                    });
                }
                Ok(Value::String(arr[idx].clone()))
            }
            Value::BoolArray(arr) => {
                let idx = if index < 0 {
                    (arr.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= arr.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: arr.len(),
                    });
                }
                Ok(Value::Boolean(arr[idx]))
            }
            Value::IntVec(vec) => {
                let v = vec.lock().unwrap();
                let idx = if index < 0 {
                    (v.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: v.len(),
                    });
                }
                Ok(Value::Integer(v[idx]))
            }
            Value::FloatVec(vec) => {
                let v = vec.lock().unwrap();
                let idx = if index < 0 {
                    (v.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: v.len(),
                    });
                }
                Ok(Value::Float(v[idx]))
            }
            Value::StringVec(vec) => {
                let v = vec.lock().unwrap();
                let idx = if index < 0 {
                    (v.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: v.len(),
                    });
                }
                Ok(Value::String(v[idx].clone()))
            }
            Value::BoolVec(vec) => {
                let v = vec.lock().unwrap();
                let idx = if index < 0 {
                    (v.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index: idx,
                        length: v.len(),
                    });
                }
                Ok(Value::Boolean(v[idx]))
            }
            v => Err(RuntimeError::TypeError(
                format!("get() first argument must be TokenIds, Array, or Vec, got {:?}", v)
            )),
        }
    }

    /// append(token_ids, token_id) -> TokenIds
    /// Append a token ID to the sequence
    fn eval_append(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("append() expects 2 arguments (token_ids, token_id), got {}", args.len())
            ));
        }

        let mut token_ids = match self.eval_expr(&args[0])? {
            Value::TokenIds(ids) => ids,
            v => return Err(RuntimeError::TypeError(
                format!("append() first argument must be TokenIds, got {:?}", v)
            )),
        };

        let new_token = match self.eval_expr(&args[1])? {
            Value::Integer(i) => i as u32,
            Value::Float(f) => f as u32,
            Value::TensorF16(ref t) => {
                // Convert scalar tensor to integer
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("append() token_id tensor must be scalar, got shape {:?}", t.dims())
                    ));
                }
                self.read_element_f16(t, 0)? as u32
            }
            v => return Err(RuntimeError::TypeError(
                format!("append() token_id must be a number or scalar tensor, got {:?}", v)
            )),
        };

        token_ids.push(new_token);
        Ok(Value::TokenIds(token_ids))
    }

    /// to_int(value) -> int
    /// Convert tensor or number to integer
    fn eval_to_int(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("to_int() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        let int_val = match value {
            Value::Integer(i) => i,
            Value::Float(f) => f as i64,
            Value::TensorF16(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("to_int() tensor must be scalar, got shape {:?}", t.dims())
                    ));
                }
                self.read_element_f16(t, 0)? as i64
            }
            Value::TensorF32(ref t) => {
                if t.numel() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("to_int() tensor must be scalar, got shape {:?}", t.dims())
                    ));
                }
                self.read_element_f32(t, 0)? as i64
            }
            v => return Err(RuntimeError::TypeError(
                format!("to_int() argument must be a number or scalar tensor, got {:?}", v)
            )),
        };

        Ok(Value::Integer(int_val))
    }

    /// str(value) -> String
    /// Convert any value to string representation
    fn eval_str(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("str() expects 1 argument, got {}", args.len())
            ));
        }

        let val = self.eval_expr(&args[0])?;
        let result = match val {
            Value::String(s) => s,
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Void => "void".to_string(),
            _ => return Err(RuntimeError::TypeError(
                "str() can only convert primitives (int, float, bool, string)".to_string()
            )),
        };

        Ok(Value::String(result))
    }

    /// input() or input("prompt")
    /// Read line from stdin
    fn eval_input(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use std::io::{self, Write};

        if args.len() > 1 {
            return Err(RuntimeError::TypeError(
                format!("input() expects 0 or 1 argument (optional prompt), got {}", args.len())
            ));
        }

        // Print prompt if provided
        if args.len() == 1 {
            let prompt_val = self.eval_expr(&args[0])?;
            if let Value::String(prompt) = prompt_val {
                print!("{}", prompt);
                io::stdout().flush().unwrap();
            }
        }

        // Read line from stdin
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)
            .map_err(|e| RuntimeError::IoError(e))?;

        // Remove trailing newline
        let input = buffer.trim_end().to_string();
        Ok(Value::String(input))
    }

    /// env("VAR_NAME")
    /// Get environment variable value
    fn eval_env(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("env() expects 1 argument (var_name), got {}", args.len())
            ));
        }

        let var_name_val = self.eval_expr(&args[0])?;
        let var_name = match var_name_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "env() argument must be a string (variable name)".to_string()
            )),
        };

        let value = std::env::var(&var_name)
            .map_err(|_| RuntimeError::InvalidOperation(
                format!("Environment variable '{}' not found", var_name)
            ))?;

        Ok(Value::String(value))
    }

    /// cleanup(var1, var2, ...) -> void
    /// Clear all variables except the ones specified
    /// This allows GPU buffers to be recycled when tensors are dropped
    /// Usage: cleanup("K_cache_0", "V_cache_0", "K_cache_1", ...)
    fn eval_cleanup(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // Parse all arguments as variable names to keep
        let mut keep_vars = Vec::new();
        for arg in args {
            let val = self.eval_expr(arg)?;
            match val {
                Value::String(s) => keep_vars.push(s),
                _ => return Err(RuntimeError::TypeError(
                    "cleanup() arguments must be strings (variable names)".to_string()
                )),
            }
        }

        // TODO: Implement clear_except functionality with scope stack
        // For now, cleanup() is not fully functional with the new scope stack architecture
        // This would require iterating through scopes and selectively removing variables

        Ok(Value::Void)
    }

    /// new_tensor_buffer(capacity_bytes: int) -> TensorBuffer
    /// Create a new TensorBuffer with specified capacity for pre-allocated GPU memory
    fn eval_new_tensor_buffer(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::device::MetalDevice;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("new_tensor_buffer() expects 1 argument (capacity in bytes), got {}", args.len())
            ));
        }

        let capacity = match self.eval_expr(&args[0])? {
            Value::Integer(n) => n as usize,
            v => return Err(RuntimeError::TypeError(
                format!("new_tensor_buffer() expects Integer as capacity argument, got {}", v.type_name())
            )),
        };

        let device = MetalDevice::new().map_err(|e| RuntimeError::TensorError(e))?;
        let buffer = device.new_tensor_buffer(capacity);
        Ok(Value::TensorBuffer(std::sync::Arc::new(buffer)))
    }

    // ============================================================================
    // Scalar Math Functions
    // ============================================================================

    /// abs(x) -> number
    /// Absolute value of a number
    fn eval_abs(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("abs() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        match value {
            Value::Integer(i) => Ok(Value::Integer(i.abs())),
            Value::Float(f) => Ok(Value::Float(f.abs())),
            v => Err(RuntimeError::TypeError(
                format!("abs() expects Integer or Float, got {}", v.type_name())
            )),
        }
    }

    /// floor(x) -> int
    /// Floor of a number (round down to nearest integer)
    fn eval_floor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("floor() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        match value {
            Value::Integer(i) => Ok(Value::Integer(i)),
            Value::Float(f) => Ok(Value::Integer(f.floor() as i64)),
            v => Err(RuntimeError::TypeError(
                format!("floor() expects Integer or Float, got {}", v.type_name())
            )),
        }
    }

    /// ceil(x) -> int
    /// Ceiling of a number (round up to nearest integer)
    fn eval_ceil(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("ceil() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        match value {
            Value::Integer(i) => Ok(Value::Integer(i)),
            Value::Float(f) => Ok(Value::Integer(f.ceil() as i64)),
            v => Err(RuntimeError::TypeError(
                format!("ceil() expects Integer or Float, got {}", v.type_name())
            )),
        }
    }

    /// round(x) -> int
    /// Round number to nearest integer
    fn eval_round(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("round() expects 1 argument, got {}", args.len())
            ));
        }

        let value = self.eval_expr(&args[0])?;
        match value {
            Value::Integer(i) => Ok(Value::Integer(i)),
            Value::Float(f) => Ok(Value::Integer(f.round() as i64)),
            v => Err(RuntimeError::TypeError(
                format!("round() expects Integer or Float, got {}", v.type_name())
            )),
        }
    }

    /// min(a, b) -> number
    /// Minimum of two numbers
    fn eval_min(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("min() expects 2 arguments, got {}", args.len())
            ));
        }

        let a = self.eval_expr(&args[0])?;
        let b = self.eval_expr(&args[1])?;

        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x.min(y))),
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x.min(y))),
            (Value::Integer(x), Value::Float(y)) => Ok(Value::Float((x as f64).min(y))),
            (Value::Float(x), Value::Integer(y)) => Ok(Value::Float(x.min(y as f64))),
            (a, b) => Err(RuntimeError::TypeError(
                format!("min() expects numbers, got {} and {}", a.type_name(), b.type_name())
            )),
        }
    }

    /// max(a, b) -> number
    /// Maximum of two numbers
    fn eval_max(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("max() expects 2 arguments, got {}", args.len())
            ));
        }

        let a = self.eval_expr(&args[0])?;
        let b = self.eval_expr(&args[1])?;

        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x.max(y))),
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x.max(y))),
            (Value::Integer(x), Value::Float(y)) => Ok(Value::Float((x as f64).max(y))),
            (Value::Float(x), Value::Integer(y)) => Ok(Value::Float(x.max(y as f64))),
            (a, b) => Err(RuntimeError::TypeError(
                format!("max() expects numbers, got {} and {}", a.type_name(), b.type_name())
            )),
        }
    }
}
