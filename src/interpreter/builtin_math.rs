//! Math operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_math_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "matmul" | "sum" | "mean" | "max" | "min" | "pow" |
            "sigmoid" | "relu" | "gelu" | "tanh" | "exp" | "log" | "sqrt" |
            "sin" | "cos" | "tan" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Math function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
