//! Utility operations for TensorLogic interpreter

use super::*;

impl Interpreter {
    pub(super) fn eval_util_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "len" | "get" | "append" | "to_int" | "str" |
            "input" | "env" |
            "sgd" | "adam" | "adamw" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Utility function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
