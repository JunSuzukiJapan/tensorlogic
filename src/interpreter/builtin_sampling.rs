//! Sampling and generation operations for TensorLogic interpreter

use super::*;
use crate::tensor::TensorIO;
use crate::tensor::Tensor;
use rand::Rng;

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

        Ok(Value::TensorF16(result))
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

        // Apply temperature scaling and sample
        // Extract logits for last position
        let mut logits_last: Vec<f32> = Vec::with_capacity(vocab_size);
        for idx in 0..vocab_size {
            logits_last.push(logits_data[start_idx + idx].to_f32());
        }

        // Apply temperature scaling
        for logit in &mut logits_last {
            *logit /= temperature as f32;
        }

        // Compute softmax (convert to probabilities)
        let max_logit = logits_last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<f32> = Vec::with_capacity(vocab_size);

        for &logit in &logits_last {
            let exp_val = (logit - max_logit).exp();
            probs.push(exp_val);
            exp_sum += exp_val;
        }

        // Normalize to get probabilities
        for prob in &mut probs {
            *prob /= exp_sum;
        }

        // Sample from the probability distribution using cumulative probabilities
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();

        let mut cumulative = 0.0f32;
        let mut sampled_idx = 0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                sampled_idx = idx;
                break;
            }
        }

        // Debug: Print sampling statistics
        if std::env::var("TL_DEBUG_SAMPLING").is_ok() {
            eprintln!("\n=== Temperature Sample Debug ===");
            eprintln!("  Logits shape: {:?}", dims);
            eprintln!("  Vocab size: {}", vocab_size);
            eprintln!("  Temperature: {}", temperature);
            eprintln!("  Sampled token: {}", sampled_idx);

            // Find top 5 tokens by probability
            let mut top_probs: Vec<(usize, f32)> = probs.iter().enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            top_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            eprintln!("  Top 5 tokens by probability:");
            for (i, (token_id, prob)) in top_probs.iter().take(5).enumerate() {
                eprintln!("    {}: token={}, prob={:.4}", i+1, token_id, prob);
            }
            eprintln!("================================\n");
        }

        // Return token ID as integer
        Ok(Value::Integer(sampled_idx as i64))
    }
}
