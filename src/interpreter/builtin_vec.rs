//! Vec operations for TensorLogic interpreter
//! Vec is a mutable, dynamically-sized collection (like Rust's Vec)

use super::*;

impl Interpreter {
    /// Evaluate Vec builtin functions
    /// Returns Some(result) if function is handled, None if not in this category
    pub(super) fn eval_vec_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "new_int_vec" => Some(self.eval_new_int_vec(args)),
            "new_float_vec" => Some(self.eval_new_float_vec(args)),
            "new_string_vec" => Some(self.eval_new_string_vec(args)),
            "new_bool_vec" => Some(self.eval_new_bool_vec(args)),
            "vec_push" => Some(self.eval_vec_push(args)),
            "vec_pop" => Some(self.eval_vec_pop(args)),
            "vec_set" => Some(self.eval_vec_set(args)),
            "vec_clear" => Some(self.eval_vec_clear(args)),
            "vec_is_empty" => Some(self.eval_vec_is_empty(args)),
            _ => None,
        }
    }

    /// new_int_vec() -> IntVec
    /// Create a new empty integer vector
    fn eval_new_int_vec(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 0 {
            return Err(RuntimeError::TypeError(
                format!("new_int_vec() expects 0 arguments, got {}", args.len())
            ));
        }

        Ok(Value::IntVec(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))))
    }

    /// new_float_vec() -> FloatVec
    /// Create a new empty float vector
    fn eval_new_float_vec(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 0 {
            return Err(RuntimeError::TypeError(
                format!("new_float_vec() expects 0 arguments, got {}", args.len())
            ));
        }

        Ok(Value::FloatVec(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))))
    }

    /// new_string_vec() -> StringVec
    /// Create a new empty string vector
    fn eval_new_string_vec(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 0 {
            return Err(RuntimeError::TypeError(
                format!("new_string_vec() expects 0 arguments, got {}", args.len())
            ));
        }

        Ok(Value::StringVec(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))))
    }

    /// new_bool_vec() -> BoolVec
    /// Create a new empty boolean vector
    fn eval_new_bool_vec(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 0 {
            return Err(RuntimeError::TypeError(
                format!("new_bool_vec() expects 0 arguments, got {}", args.len())
            ));
        }

        Ok(Value::BoolVec(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))))
    }

    /// vec_push(vec, value) -> void
    /// Push a value to the end of the vector
    fn eval_vec_push(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("vec_push() expects 2 arguments (vec, value), got {}", args.len())
            ));
        }

        let vec_val = self.eval_expr(&args[0])?;
        let value = self.eval_expr(&args[1])?;

        match (vec_val, value) {
            (Value::IntVec(vec), Value::Integer(val)) => {
                vec.lock().unwrap().push(val);
                Ok(Value::Void)
            }
            (Value::FloatVec(vec), Value::Float(val)) => {
                vec.lock().unwrap().push(val);
                Ok(Value::Void)
            }
            (Value::FloatVec(vec), Value::Integer(val)) => {
                // Allow int->float conversion
                vec.lock().unwrap().push(val as f64);
                Ok(Value::Void)
            }
            (Value::StringVec(vec), Value::String(val)) => {
                vec.lock().unwrap().push(val);
                Ok(Value::Void)
            }
            (Value::BoolVec(vec), Value::Boolean(val)) => {
                vec.lock().unwrap().push(val);
                Ok(Value::Void)
            }
            (vec_val, value) => Err(RuntimeError::TypeError(
                format!("vec_push() type mismatch: {} cannot hold {}",
                    vec_val.type_name(), value.type_name())
            )),
        }
    }

    /// vec_pop(vec) -> value
    /// Remove and return the last element from the vector
    fn eval_vec_pop(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("vec_pop() expects 1 argument (vec), got {}", args.len())
            ));
        }

        let vec_val = self.eval_expr(&args[0])?;

        match vec_val {
            Value::IntVec(vec) => {
                let val = vec.lock().unwrap().pop()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        "vec_pop() called on empty IntVec".to_string()
                    ))?;
                Ok(Value::Integer(val))
            }
            Value::FloatVec(vec) => {
                let val = vec.lock().unwrap().pop()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        "vec_pop() called on empty FloatVec".to_string()
                    ))?;
                Ok(Value::Float(val))
            }
            Value::StringVec(vec) => {
                let val = vec.lock().unwrap().pop()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        "vec_pop() called on empty StringVec".to_string()
                    ))?;
                Ok(Value::String(val))
            }
            Value::BoolVec(vec) => {
                let val = vec.lock().unwrap().pop()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        "vec_pop() called on empty BoolVec".to_string()
                    ))?;
                Ok(Value::Boolean(val))
            }
            v => Err(RuntimeError::TypeError(
                format!("vec_pop() expects a Vec type, got {}", v.type_name())
            )),
        }
    }

    /// vec_set(vec, index, value) -> void
    /// Set the value at the specified index
    fn eval_vec_set(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("vec_set() expects 3 arguments (vec, index, value), got {}", args.len())
            ));
        }

        let vec_val = self.eval_expr(&args[0])?;
        let index_val = self.eval_expr(&args[1])?;
        let value = self.eval_expr(&args[2])?;

        let index = match index_val {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            v => return Err(RuntimeError::TypeError(
                format!("vec_set() index must be a number, got {}", v.type_name())
            )),
        };

        match (vec_val, value) {
            (Value::IntVec(vec), Value::Integer(val)) => {
                let mut v = vec.lock().unwrap();
                if index >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: v.len(),
                    });
                }
                v[index] = val;
                Ok(Value::Void)
            }
            (Value::FloatVec(vec), Value::Float(val)) => {
                let mut v = vec.lock().unwrap();
                if index >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: v.len(),
                    });
                }
                v[index] = val;
                Ok(Value::Void)
            }
            (Value::FloatVec(vec), Value::Integer(val)) => {
                // Allow int->float conversion
                let mut v = vec.lock().unwrap();
                if index >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: v.len(),
                    });
                }
                v[index] = val as f64;
                Ok(Value::Void)
            }
            (Value::StringVec(vec), Value::String(val)) => {
                let mut v = vec.lock().unwrap();
                if index >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: v.len(),
                    });
                }
                v[index] = val;
                Ok(Value::Void)
            }
            (Value::BoolVec(vec), Value::Boolean(val)) => {
                let mut v = vec.lock().unwrap();
                if index >= v.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: v.len(),
                    });
                }
                v[index] = val;
                Ok(Value::Void)
            }
            (vec_val, value) => Err(RuntimeError::TypeError(
                format!("vec_set() type mismatch: {} cannot hold {}",
                    vec_val.type_name(), value.type_name())
            )),
        }
    }

    /// vec_clear(vec) -> void
    /// Remove all elements from the vector
    fn eval_vec_clear(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("vec_clear() expects 1 argument (vec), got {}", args.len())
            ));
        }

        let vec_val = self.eval_expr(&args[0])?;

        match vec_val {
            Value::IntVec(vec) => {
                vec.lock().unwrap().clear();
                Ok(Value::Void)
            }
            Value::FloatVec(vec) => {
                vec.lock().unwrap().clear();
                Ok(Value::Void)
            }
            Value::StringVec(vec) => {
                vec.lock().unwrap().clear();
                Ok(Value::Void)
            }
            Value::BoolVec(vec) => {
                vec.lock().unwrap().clear();
                Ok(Value::Void)
            }
            v => Err(RuntimeError::TypeError(
                format!("vec_clear() expects a Vec type, got {}", v.type_name())
            )),
        }
    }

    /// vec_is_empty(vec) -> bool
    /// Check if the vector is empty
    fn eval_vec_is_empty(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("vec_is_empty() expects 1 argument (vec), got {}", args.len())
            ));
        }

        let vec_val = self.eval_expr(&args[0])?;

        match vec_val {
            Value::IntVec(vec) => Ok(Value::Boolean(vec.lock().unwrap().is_empty())),
            Value::FloatVec(vec) => Ok(Value::Boolean(vec.lock().unwrap().is_empty())),
            Value::StringVec(vec) => Ok(Value::Boolean(vec.lock().unwrap().is_empty())),
            Value::BoolVec(vec) => Ok(Value::Boolean(vec.lock().unwrap().is_empty())),
            v => Err(RuntimeError::TypeError(
                format!("vec_is_empty() expects a Vec type, got {}", v.type_name())
            )),
        }
    }
}
