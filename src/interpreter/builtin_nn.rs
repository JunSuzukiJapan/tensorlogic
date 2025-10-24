//! Neural network operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_nn_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "layer_norm" | "rms_norm" | "batch_norm" | "dropout" |
            "embedding" | "positional_encoding" |
            "apply_mask" | "causal_mask" | "padding_mask" | "combine_masks" | "apply_attention_mask" |
            "fused_add_relu" | "fused_mul_relu" | "fused_affine" | "fused_gelu_linear" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("NN function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
