//! Neural network operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_nn_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "rms_norm" => Some(self.eval_rms_norm(args)),
            "layer_norm" | "batch_norm" | "dropout" |
            "positional_encoding" |
            "apply_mask" | "causal_mask" | "padding_mask" | "combine_masks" | "apply_attention_mask" |
            "fused_add_relu" | "fused_mul_relu" | "fused_affine" | "fused_gelu_linear" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("NN function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// rms_norm(x, weight) -> tensor
    /// RMS Normalization (used in LLaMA, TinyLlama)
    fn eval_rms_norm(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("rms_norm() expects 2 arguments (tensor, weight), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Evaluate weight argument
        let weight_val = self.eval_expr(&args[1])?;
        let weight = weight_val.as_tensor()?;

        // Get the normalized shape from weight dimensions
        let normalized_shape = weight.dims().to_vec();

        // Use default epsilon value (1e-6 for LLaMA/TinyLlama)
        let eps = 1e-6;

        // Apply RMS normalization
        let result = tensor.rms_norm(normalized_shape, &weight, eps)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(result))
    }
}
