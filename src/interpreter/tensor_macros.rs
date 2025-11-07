//! Macros for handling f16/f32 generic tensor operations

/// Apply operation to tensor, preserving its type (f16 or f32)
///
/// Usage:
/// ```rust
/// let tensor_val = self.eval_expr(&args[0])?;
/// with_tensor_unary!(tensor_val, t, { t.sigmoid() })
/// ```
#[macro_export]
macro_rules! with_tensor_unary {
    ($val:expr, $tensor:ident, $op:expr) => {
        match $val {
            crate::interpreter::Value::TensorF16($tensor) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF16(result))
            }
            crate::interpreter::Value::TensorF32($tensor) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF32(result))
            }
            _ => Err(crate::interpreter::RuntimeError::TypeError(
                "Expected tensor (f16 or f32)".to_string()
            )),
        }
    };
}

/// Apply binary operation to two tensors of same type
///
/// Usage:
/// ```rust
/// let a_val = self.eval_expr(&args[0])?;
/// let b_val = self.eval_expr(&args[1])?;
/// with_tensor_binary!(a_val, b_val, a, b, { a.add(&b) })
/// ```
#[macro_export]
macro_rules! with_tensor_binary {
    ($val1:expr, $val2:expr, $t1:ident, $t2:ident, $op:expr) => {
        match ($val1, $val2) {
            (crate::interpreter::Value::TensorF16($t1), crate::interpreter::Value::TensorF16($t2)) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF16(result))
            }
            (crate::interpreter::Value::TensorF32($t1), crate::interpreter::Value::TensorF32($t2)) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF32(result))
            }
            _ => Err(crate::interpreter::RuntimeError::TypeError(
                "Binary operation requires both tensors to be same type (both f16 or both f32)".to_string()
            )),
        }
    };
}

/// Apply ternary operation to three tensors of same type
#[macro_export]
macro_rules! with_tensor_ternary {
    ($val1:expr, $val2:expr, $val3:expr, $t1:ident, $t2:ident, $t3:ident, $op:expr) => {
        match ($val1, $val2, $val3) {
            (crate::interpreter::Value::TensorF16($t1), crate::interpreter::Value::TensorF16($t2), crate::interpreter::Value::TensorF16($t3)) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF16(result))
            }
            (crate::interpreter::Value::TensorF32($t1), crate::interpreter::Value::TensorF32($t2), crate::interpreter::Value::TensorF32($t3)) => {
                let result = $op.map_err(|e| crate::interpreter::RuntimeError::TensorError(e))?;
                Ok(crate::interpreter::Value::TensorF32(result))
            }
            _ => Err(crate::interpreter::RuntimeError::TypeError(
                "Ternary operation requires all tensors to be same type (all f16 or all f32)".to_string()
            )),
        }
    };
}
