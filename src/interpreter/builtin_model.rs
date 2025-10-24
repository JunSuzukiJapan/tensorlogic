//! Model and I/O operations for TensorLogic interpreter

use super::*;

impl Interpreter {
    pub(super) fn eval_model_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "load_model" | "load_tokenizer" | "get_tensor" |
            "tokenize" | "detokenize" | "generate" |
            "save" | "load" |
            "print" | "print_top_k" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Model/IO function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
