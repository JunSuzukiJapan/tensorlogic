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
            _ => None,
        }
    }

    /// len(value) -> int
    /// Get length of TokenIds, Tensor, or String
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
            v => return Err(RuntimeError::TypeError(
                format!("len() argument must be TokenIds, Tensor, or String, got {:?}", v)
            )),
        };

        Ok(Value::Integer(length))
    }

    /// get(token_ids, index) -> int
    /// Get token ID at specific index
    fn eval_get(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("get() expects 2 arguments (token_ids, index), got {}", args.len())
            ));
        }

        let token_ids = match self.eval_expr(&args[0])? {
            Value::TokenIds(ids) => ids,
            v => return Err(RuntimeError::TypeError(
                format!("get() first argument must be TokenIds, got {:?}", v)
            )),
        };

        let index = match self.eval_expr(&args[1])? {
            Value::Integer(i) => {
                // Support negative indexing
                if i < 0 {
                    (token_ids.len() as i64 + i) as usize
                } else {
                    i as usize
                }
            }
            Value::Float(f) => f as usize,
            v => return Err(RuntimeError::TypeError(
                format!("get() index must be a number, got {:?}", v)
            )),
        };

        if index >= token_ids.len() {
            return Err(RuntimeError::IndexError {
                index,
                length: token_ids.len(),
            });
        }

        Ok(Value::Integer(token_ids[index] as i64))
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

        // Clear all variables except the ones in keep_vars
        self.env.clear_except(&keep_vars);

        Ok(Value::Void)
    }
}
