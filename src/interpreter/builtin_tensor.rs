//! Basic tensor operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;
use half::f16;

impl Interpreter {
    pub(super) fn eval_tensor_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "zeros" | "ones" | "reshape" | "flatten" | "shape" | "transpose" | "permute" |
            "concat" | "gather" | "scatter" | "broadcast_to" | "chunk" | "split" |
            "squeeze" | "unsqueeze" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Tensor function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
