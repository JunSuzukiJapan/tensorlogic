//! Macros for handling f16/f32 generic builtin functions

/// Macro for single tensor input -> tensor output operations
/// Usage: tensor_unary_op!(input_val, |t| t.operation(args))
#[macro_export]
macro_rules! tensor_unary_op {
    ($input:expr, $op:expr) => {
        match $input {
            Value::TensorF16(t) => {
                let result = $op(&t)?;
                Ok(Value::TensorF16(result))
            }
            Value::TensorF32(t) => {
                let result = $op(&t)?;
                Ok(Value::TensorF32(result))
            }
            v => Err(RuntimeError::TypeError(
                format!("Expected tensor (f16 or f32), got {:?}", v)
            )),
        }
    };
}

/// Macro for two tensor inputs -> tensor output operations
/// Usage: tensor_binary_op!(input1_val, input2_val, |a, b| a.operation(b))
#[macro_export]
macro_rules! tensor_binary_op {
    ($input1:expr, $input2:expr, $op:expr) => {
        match ($input1, $input2) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let result = $op(&a, &b)?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let result = $op(&a, &b)?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "Binary operation requires both tensors to be same type (both f16 or both f32)".to_string()
            )),
        }
    };
}

/// Macro for three tensor inputs -> tensor output operations
/// Usage: tensor_ternary_op!(t1, t2, t3, |a, b, c| a.operation(b, c))
#[macro_export]
macro_rules! tensor_ternary_op {
    ($input1:expr, $input2:expr, $input3:expr, $op:expr) => {
        match ($input1, $input2, $input3) {
            (Value::TensorF16(a), Value::TensorF16(b), Value::TensorF16(c)) => {
                let result = $op(&a, &b, &c)?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(a), Value::TensorF32(b), Value::TensorF32(c)) => {
                let result = $op(&a, &b, &c)?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "Ternary operation requires all tensors to be same type (all f16 or all f32)".to_string()
            )),
        }
    };
}
