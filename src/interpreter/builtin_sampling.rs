//! Sampling and generation operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_sampling_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "softmax" => Some(self.eval_softmax(args)),
            "temperature_sample" => Some(self.eval_temperature_sample(args)),
            "sample" | "top_k" | "top_p" | "top_p_sample" |
            "temperature" | "argmax" | "argmin" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Sampling function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// softmax(x) or softmax(x, dim) -> tensor
    /// Softmax activation along the last dimension (or specified dimension)
    fn eval_softmax(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.is_empty() || args.len() > 2 {
            return Err(RuntimeError::TypeError(
                format!("softmax() expects 1 or 2 arguments (tensor, [dim]), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // For now, ignore the dimension argument and apply along last dimension
        // TODO: Implement softmax with custom dimension support
        if args.len() == 2 {
            let _dim_val = self.eval_expr(&args[1])?;
            // Dimension parameter is ignored for now
        }

        // Apply softmax
        let result = tensor.softmax()
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }

    /// temperature_sample(logits, temperature) -> int
    /// Sample a token from logits with temperature scaling
    /// Returns the sampled token ID as an integer
    ///
    /// For 2D logits [seq_len, vocab_size], uses only the last sequence position
    fn eval_temperature_sample(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use half::f16;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("temperature_sample() expects 2 arguments (logits, temperature), got {}", args.len())
            ));
        }

        // Evaluate logits tensor
        let logits_val = self.eval_expr(&args[0])?;
        let logits = logits_val.as_tensor()?;

        // Evaluate temperature
        let temp_val = self.eval_expr(&args[1])?;
        let temperature = match temp_val {
            Value::Float(f) => f,
            Value::Integer(i) => i as f64,
            _ => return Err(RuntimeError::TypeError(
                "temperature_sample() temperature must be a number".to_string()
            )),
        };

        // Get logits as vector
        let logits_data = logits.to_vec();
        let dims = logits.dims();

        // For 2D tensors [seq_len, vocab_size], use only the last sequence position
        let (start_idx, vocab_size) = if dims.len() == 2 {
            let seq_len = dims[0];
            let vocab_size = dims[1];
            let start_idx = (seq_len - 1) * vocab_size;
            (start_idx, vocab_size)
        } else if dims.len() == 1 {
            (0, dims[0])
        } else {
            return Err(RuntimeError::TypeError(
                "temperature_sample() expects 1D or 2D logits tensor".to_string()
            ));
        };

        // For now, implement greedy sampling (argmax)
        // TODO: Implement proper temperature-scaled sampling with random selection

        // Find argmax in the last sequence position
        let mut max_idx = 0;
        let mut max_val = f16::NEG_INFINITY;

        for idx in 0..vocab_size {
            let val = logits_data[start_idx + idx];
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        // Return token ID as integer
        Ok(Value::Integer(max_idx as i64))
    }
}
