//! Neural network operations for TensorLogic interpreter

use super::*;
use crate::device::EncoderProvider;
use crate::interpreter::value::ToValue;
use crate::tensor::{Tensor, TensorCreation};
use half::f16;
use std::io::Write;
use std::time::Instant;
use std::sync::Arc;

impl Interpreter {
    pub(super) fn eval_nn_function(
        &mut self,
        name: &str,
        args: &[TensorExpr],
    ) -> Option<RuntimeResult<Value>> {
        match name {
            "rms_norm" => Some(self.eval_rms_norm(args)),
            "layer_norm" => Some(self.eval_layer_norm(args)),
            "positional_encoding" => Some(self.eval_positional_encoding(args)),
            "embedding" => Some(self.eval_embedding(args)),
            "apply_attention_mask" => Some(self.eval_apply_attention_mask(args)),
            "padding_mask" => Some(self.eval_padding_mask(args)),
            "combine_masks" => Some(self.eval_combine_masks(args)),
            "fused_add_relu" => Some(self.eval_fused_add_relu(args)),
            "fused_mul_relu" => Some(self.eval_fused_mul_relu(args)),
            "fused_affine" => Some(self.eval_fused_affine(args)),
            "fused_gelu_linear" => Some(self.eval_fused_gelu_linear(args)),
            "attention_with_cache" => Some(self.eval_attention_with_cache(args)),
            "causal_mask" => Some(self.eval_causal_mask(args)),
            "batch_norm" | "dropout" | "apply_mask" => {
                Some(Err(RuntimeError::NotImplemented(format!(
                    "NN function '{}' migration in progress",
                    name
                ))))
            }
            _ => None,
        }
    }

    /// rms_norm(x, weight, [eps]) -> tensor
    /// RMS Normalization (used in LLaMA, TinyLlama)
    fn eval_rms_norm(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let _start = Instant::now();
        if args.len() < 2 || args.len() > 3 {
            return Err(RuntimeError::TypeError(format!(
                "rms_norm() expects 2-3 arguments (tensor, weight, optional eps), got {}",
                args.len()
            )));
        }

        // Evaluate tensor and weight arguments
        let tensor_val = self.eval_expr(&args[0])?;
        let weight_val = self.eval_expr(&args[1])?;
        // eprintln!("[TIMING] rms_norm: arg evaluation: {:.3}ms", _start.elapsed().as_secs_f64() * 1000.0);

        // Debug: check input and weight BEFORE moving them
        if std::env::var("TL_DEBUG_ATTN").is_ok() {
            if let (Value::TensorF32(ref input_t), Value::TensorF32(ref weight_t)) =
                (&tensor_val, &weight_val)
            {
                use crate::tensor::TensorIO;
                let input_data = input_t.sync_and_read();
                let weight_data = weight_t.sync_and_read();

                let input_mean = input_data.iter().sum::<f32>() / input_data.len() as f32;
                let input_max = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_mean = weight_data.iter().sum::<f32>() / weight_data.len() as f32;
                let weight_max = weight_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                if input_mean.abs() < 1e-6 && input_max.abs() < 1e-6 {
                    eprintln!("\n=== WARNING: RMS Norm INPUT is all zeros! ===");
                    eprintln!("Input shape: {:?}", input_t.dims());
                }
                if weight_mean.abs() < 1e-6 && weight_max.abs() < 1e-6 {
                    eprintln!("\n=== WARNING: RMS Norm WEIGHT is all zeros! ===");
                    eprintln!("Weight shape: {:?}", weight_t.dims());
                }
            }
        }

        // Use default epsilon value (1e-6 for LLaMA/TinyLlama) or custom value
        let eps = if args.len() >= 3 {
            match self.eval_expr(&args[2])? {
                Value::Float(f) => f as f32,
                Value::Integer(i) => i as f32,
                _ => 1e-6_f32,
            }
        } else {
            1e-6_f32
        };

        // Process based on input type (f16 or f32)
        let compute_start = Instant::now();
        let result = match (tensor_val, weight_val) {
            (Value::TensorF16(tensor), Value::TensorF16(weight)) => {
                let normalized_shape = weight.dims().to_vec();
                let result = tensor
                    .rms_norm(normalized_shape, &weight, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(Arc::new(result)))
            }
            (Value::TensorF32(tensor), Value::TensorF32(weight)) => {
                let normalized_shape = weight.dims().to_vec();
                let result = tensor
                    .rms_norm(normalized_shape, &weight, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(Arc::new(result)))
            }
            _ => Err(RuntimeError::TypeError(
                "rms_norm() requires tensor and weight to be same type (both f16 or both f32)"
                    .to_string(),
            )),
        };
        // eprintln!("[TIMING] rms_norm: computation: {:.3}ms, total: {:.3}ms",
        //           compute_start.elapsed().as_secs_f64() * 1000.0,
        //           _start.elapsed().as_secs_f64() * 1000.0);
        if std::env::var("TL_PERF").is_ok() {
            let dtype = match &result {
                Ok(Value::TensorF16(_)) => "f16",
                Ok(Value::TensorF32(_)) => "f32",
                _ => "unknown",
            };
            eprintln!(
                "[PERF] rms_norm({}): {:.3}ms",
                dtype,
                _start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Debug: output final norm values for logits debugging
        if std::env::var("TL_DEBUG_ATTN").is_ok() {
            if let Ok(Value::TensorF32(ref tensor)) = result {
                use crate::tensor::TensorIO;
                let data = tensor.sync_and_read();
                eprintln!("\n=== RMS Norm Output (f32) ===");
                eprintln!("Shape: {:?}", tensor.dims());
                eprintln!("First 10: {:?}", &data[..data.len().min(10)]);
                eprintln!("Last 10: {:?}", &data[data.len().saturating_sub(10)..]);
                eprintln!("Mean: {}", data.iter().sum::<f32>() / data.len() as f32);
                eprintln!("Min: {}", data.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
                eprintln!(
                    "Max: {}",
                    data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                );
            }
        }

        result
    }

    /// positional_encoding(seq_len, d_model) -> tensor
    /// Generates sinusoidal positional encoding
    fn eval_positional_encoding(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "positional_encoding() expects 2 arguments (seq_len, d_model), got {}",
                args.len()
            )));
        }

        let seq_len = match self.eval_expr(&args[0])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            v => {
                return Err(RuntimeError::TypeError(format!(
                    "positional_encoding() first argument must be a number (seq_len), got {:?}",
                    v
                )))
            }
        };

        let d_model =
            match self.eval_expr(&args[1])? {
                Value::Integer(i) => i as usize,
                Value::Float(f) => f as usize,
                v => return Err(RuntimeError::TypeError(format!(
                    "positional_encoding() second argument must be a number (d_model), got {:?}",
                    v
                ))),
            };

        // Generate sinusoidal positional encoding
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        let mut pe_data = Vec::with_capacity(seq_len * d_model);

        for pos in 0..seq_len {
            for i in 0..d_model {
                let div_term = (i as f32 / d_model as f32) * 10000_f32.ln();
                let angle = pos as f32 / div_term.exp();

                let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };

                pe_data.push(half::f16::from_f32(value));
            }
        }

        // Create output tensor
        let output = crate::tensor::Tensor::from_vec_gpu(
            self.env.metal_device(),
            pe_data,
            vec![seq_len, d_model],
        )
        .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(Arc::new(output)))
    }

    /// layer_norm(tensor, [normalized_shape], [eps]) -> tensor
    /// Layer Normalization
    fn eval_layer_norm(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        // Support two signatures:
        // 1. layer_norm(tensor, eps) - 2 args (no gamma/beta)
        // 2. layer_norm(tensor, gamma, beta, eps) - 4 args (with gamma/beta)
        if args.is_empty() || (args.len() != 2 && args.len() != 4) {
            return Err(RuntimeError::TypeError(
                format!("layer_norm() expects 2 or 4 arguments:\n  - layer_norm(tensor, eps)\n  - layer_norm(tensor, gamma, beta, eps)\ngot {} arguments", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        // Parse gamma and beta if provided (4 args)
        let (gamma_opt, beta_opt, eps) = if args.len() == 4 {
            // Extract gamma tensor
            let gamma_val = self.eval_expr(&args[1])?;
            let gamma = match gamma_val {
                Value::TensorF16(ref t) => Some(t.clone()),
                Value::TensorF32(ref t) => {
                    // Convert f32 to f16 if needed
                    use crate::tensor::TensorConvert;
                    Some(std::sync::Arc::new(t.to_f16().map_err(|e| RuntimeError::TensorError(e))?))
                }
                _ => return Err(RuntimeError::TypeError("layer_norm() gamma must be a tensor".to_string())),
            };

            // Extract beta tensor
            let beta_val = self.eval_expr(&args[2])?;
            let beta = match beta_val {
                Value::TensorF16(ref t) => Some(t.clone()),
                Value::TensorF32(ref t) => {
                    // Convert f32 to f16 if needed
                    use crate::tensor::TensorConvert;
                    Some(std::sync::Arc::new(t.to_f16().map_err(|e| RuntimeError::TensorError(e))?))
                }
                _ => return Err(RuntimeError::TypeError("layer_norm() beta must be a tensor".to_string())),
            };

            // Extract eps
            let eps_val = match self.eval_expr(&args[3])? {
                Value::Float(f) => f as f32,
                Value::Integer(i) => i as f32,
                _ => 1e-5_f32,
            };

            (gamma, beta, eps_val)
        } else {
            // 2 args: just tensor and eps
            let eps_val = match self.eval_expr(&args[1])? {
                Value::Float(f) => f as f32,
                Value::Integer(i) => i as f32,
                _ => 1e-5_f32,
            };
            (None, None, eps_val)
        };

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                let shape = tensor.shape();
                let dims = shape.dims();
                let normalized_shape = vec![dims[dims.len() - 1]];
                tensor
                    .layer_norm(normalized_shape, gamma_opt.as_deref(), beta_opt.as_deref(), eps)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            Value::TensorF32(tensor) => {
                let shape = tensor.shape();
                let dims = shape.dims();
                let normalized_shape = vec![dims[dims.len() - 1]];

                // Convert gamma/beta to f32 if provided
                let gamma_f32 = gamma_opt.as_ref().map(|g| {
                    use crate::tensor::TensorConvert;
                    g.to_f32()
                }).transpose().map_err(|e| RuntimeError::TensorError(e))?;

                let beta_f32 = beta_opt.as_ref().map(|b| {
                    use crate::tensor::TensorConvert;
                    b.to_f32()
                }).transpose().map_err(|e| RuntimeError::TensorError(e))?;

                tensor
                    .layer_norm(normalized_shape, gamma_f32.as_ref(), beta_f32.as_ref(), eps)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string())),
        })
    }

    /// apply_attention_mask(scores, mask) -> tensor
    /// Apply attention mask to scores
    /// mask: 1 = keep, 0 = mask out (set to -1e9)
    fn eval_apply_attention_mask(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "apply_attention_mask() expects 2 arguments (scores, mask), got {}",
                args.len()
            )));
        }

        let scores_val = self.eval_expr(&args[0])?;
        let mask_val = self.eval_expr(&args[1])?;

        let device = self.env.metal_device();

        Ok(match (scores_val, mask_val) {
            (Value::TensorF16(scores), Value::TensorF16(mask)) => {
                // Apply mask: scores + (1 - mask) * large_negative
                // CRITICAL: f16 range is ±65504, so -1e9 becomes -inf!
                // Use -100 instead (safe for f16, exp(-100) ≈ 3.7e-44 ≈ 0 for softmax)
                let ones = Tensor::ones(device, scores.shape().dims().to_vec())
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let inv_mask = ones.sub(&mask).map_err(|e| RuntimeError::TensorError(e))?;

                let large_neg_value = half::f16::from_f32(-100.0);  // Safe for f16, large enough for masking
                let large_neg_vec = vec![large_neg_value; inv_mask.shape().numel()];
                let large_neg = Tensor::from_vec_gpu(device, large_neg_vec, inv_mask.shape().dims().to_vec())
                    .map_err(|e| RuntimeError::TensorError(e))?;

                let mask_value = inv_mask.mul(&large_neg).map_err(|e| RuntimeError::TensorError(e))?;
                scores.add(&mask_value).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(scores), Value::TensorF32(mask)) => {
                // Apply mask: scores + (1 - mask) * (-1e9)
                let ones = Tensor::ones(device, scores.shape().dims().to_vec())
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let inv_mask = ones.sub(&mask).map_err(|e| RuntimeError::TensorError(e))?;

                let large_neg_vec = vec![-1e9_f32; inv_mask.shape().numel()];
                let large_neg = Tensor::from_vec_gpu(device, large_neg_vec, inv_mask.shape().dims().to_vec())
                    .map_err(|e| RuntimeError::TensorError(e))?;

                let mask_value = inv_mask.mul(&large_neg).map_err(|e| RuntimeError::TensorError(e))?;
                scores.add(&mask_value).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "apply_attention_mask() requires both tensors to be same type (both f16 or both f32)".to_string()
            ))
        })
    }

    /// padding_mask(lengths, max_length) -> tensor
    /// Create padding mask from sequence lengths
    /// Returns [batch_size, max_length] tensor where 1 = valid, 0 = padding
    fn eval_padding_mask(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "padding_mask() expects 2 arguments (lengths, max_length), got {}",
                args.len()
            )));
        }

        let lengths_val = self.eval_expr(&args[0])?;
        let max_length = match self.eval_expr(&args[1])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            _ => {
                return Err(RuntimeError::TypeError(
                    "max_length must be an integer".to_string(),
                ))
            }
        };

        let device = self.env.metal_device();

        Ok(match lengths_val {
            Value::TensorF16(lengths_tensor) => {
                let lengths_vec = lengths_tensor.sync_and_read();
                let lengths: Vec<usize> =
                    lengths_vec.iter().map(|&v| v.to_f32() as usize).collect();

                let batch_size = lengths.len();
                let mut mask_data = Vec::with_capacity(batch_size * max_length);

                for len in lengths {
                    for pos in 0..max_length {
                        if pos < len {
                            mask_data.push(half::f16::ONE);
                        } else {
                            mask_data.push(half::f16::ZERO);
                        }
                    }
                }

                Tensor::from_vec_gpu(device, mask_data, vec![batch_size, max_length])
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            Value::TensorF32(lengths_tensor) => {
                let lengths_vec = lengths_tensor.sync_and_read_f32();
                let lengths: Vec<usize> = lengths_vec.iter().map(|&v| v as usize).collect();

                let batch_size = lengths.len();
                let mut mask_data = Vec::with_capacity(batch_size * max_length);

                for len in lengths {
                    for pos in 0..max_length {
                        if pos < len {
                            mask_data.push(1.0_f32);
                        } else {
                            mask_data.push(0.0_f32);
                        }
                    }
                }

                Tensor::from_vec_gpu(device, mask_data, vec![batch_size, max_length])
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "padding_mask() expects a tensor for lengths".to_string(),
                ))
            }
        })
    }

    /// combine_masks(mask1, mask2) -> tensor
    /// Combine two masks using element-wise multiplication (logical AND)
    fn eval_combine_masks(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "combine_masks() expects 2 arguments (mask1, mask2), got {}",
                args.len()
            )));
        }

        let mask1_val = self.eval_expr(&args[0])?;
        let mask2_val = self.eval_expr(&args[1])?;

        Ok(match (mask1_val, mask2_val) {
            (Value::TensorF16(mask1), Value::TensorF16(mask2)) => mask1
                .mul(&mask2)
                .map_err(|e| RuntimeError::TensorError(e))?
                .to_value(),
            (Value::TensorF32(mask1), Value::TensorF32(mask2)) => mask1
                .mul(&mask2)
                .map_err(|e| RuntimeError::TensorError(e))?
                .to_value(),
            _ => {
                return Err(RuntimeError::TypeError(
                    "combine_masks() requires both masks to be same type (both f16 or both f32)"
                        .to_string(),
                ))
            }
        })
    }

    /// causal_mask(seq_len) -> tensor
    /// Create a causal mask for autoregressive attention
    /// Returns [seq_len, seq_len] lower triangular matrix where 1=allow, 0=mask
    fn eval_causal_mask(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(format!(
                "causal_mask() expects 1 argument (seq_len), got {}",
                args.len()
            )));
        }

        let seq_len = match self.eval_expr(&args[0])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            _ => {
                return Err(RuntimeError::TypeError(
                    "seq_len must be a number".to_string(),
                ))
            }
        };

        let device = self.env.metal_device();

        // Create lower triangular mask: mask[i,j] = 1 if j <= i, else 0
        let mut mask_data = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    mask_data.push(half::f16::ONE);
                } else {
                    mask_data.push(half::f16::ZERO);
                }
            }
        }

        Ok(Tensor::from_vec_gpu(device, mask_data, vec![seq_len, seq_len])
            .map_err(|e| RuntimeError::TensorError(e))?
            .to_value())
    }

    /// fused_add_relu(a, b) -> tensor
    /// Fused operation: ReLU(a + b)
    fn eval_fused_add_relu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "fused_add_relu() expects 2 arguments (a, b), got {}",
                args.len()
            )));
        }

        let a_val = self.eval_expr(&args[0])?;
        let b_val = self.eval_expr(&args[1])?;

        Ok(match (a_val, b_val) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let sum = a.add(&b).map_err(|e| RuntimeError::TensorError(e))?;
                sum.relu()
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let sum = a.add(&b).map_err(|e| RuntimeError::TensorError(e))?;
                sum.relu()
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "fused_add_relu() requires both tensors to be same type (both f16 or both f32)"
                        .to_string(),
                ))
            }
        })
    }

    /// fused_mul_relu(a, b) -> tensor
    /// Fused operation: ReLU(a * b)
    fn eval_fused_mul_relu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "fused_mul_relu() expects 2 arguments (a, b), got {}",
                args.len()
            )));
        }

        let a_val = self.eval_expr(&args[0])?;
        let b_val = self.eval_expr(&args[1])?;

        Ok(match (a_val, b_val) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let product = a.mul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                product
                    .relu()
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let product = a.mul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                product
                    .relu()
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "fused_mul_relu() requires both tensors to be same type (both f16 or both f32)"
                        .to_string(),
                ))
            }
        })
    }

    /// fused_affine(x, scale, bias) -> tensor
    /// Fused operation: x * scale + bias
    fn eval_fused_affine(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 3 {
            return Err(RuntimeError::TypeError(format!(
                "fused_affine() expects 3 arguments (x, scale, bias), got {}",
                args.len()
            )));
        }

        let x_val = self.eval_expr(&args[0])?;
        let scale_val = self.eval_expr(&args[1])?;
        let bias_val = self.eval_expr(&args[2])?;

        Ok(match (x_val, scale_val, bias_val) {
            (Value::TensorF16(x), Value::TensorF16(scale), Value::TensorF16(bias)) => {
                let scaled = x.mul(&scale).map_err(|e| RuntimeError::TensorError(e))?;
                scaled
                    .add(&bias)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            (Value::TensorF32(x), Value::TensorF32(scale), Value::TensorF32(bias)) => {
                let scaled = x.mul(&scale).map_err(|e| RuntimeError::TensorError(e))?;
                scaled
                    .add(&bias)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "fused_affine() requires all tensors to be same type (all f16 or all f32)"
                        .to_string(),
                ))
            }
        })
    }

    /// fused_gelu_linear(input, weight, bias) -> tensor
    /// Fused operation: GELU(input) @ weight + bias
    fn eval_fused_gelu_linear(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 3 {
            return Err(RuntimeError::TypeError(format!(
                "fused_gelu_linear() expects 3 arguments (input, weight, bias), got {}",
                args.len()
            )));
        }

        let input_val = self.eval_expr(&args[0])?;
        let weight_val = self.eval_expr(&args[1])?;
        let bias_val = self.eval_expr(&args[2])?;

        Ok(match (input_val, weight_val, bias_val) {
            (Value::TensorF16(input), Value::TensorF16(weight), Value::TensorF16(bias)) => {
                let activated = input.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                let linear = activated
                    .matmul(&weight)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                linear
                    .add(&bias)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            (Value::TensorF32(input), Value::TensorF32(weight), Value::TensorF32(bias)) => {
                let activated = input.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                let linear = activated
                    .matmul(&weight)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                linear
                    .add(&bias)
                    .map_err(|e| RuntimeError::TensorError(e))?
                    .to_value()
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "fused_gelu_linear() requires all tensors to be same type (all f16 or all f32)"
                        .to_string(),
                ))
            }
        })
    }

    /// embedding(embedding_table, token_ids) -> tensor
    /// Embedding lookup: retrieve embedding vectors for token IDs
    /// token_ids can be: TokenIds, TokenIdArray, or Int (single token)
    fn eval_embedding(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "embedding() expects 2 arguments (embedding_table, token_ids), got {}",
                args.len()
            )));
        }

        // Evaluate embedding table
        let table_val = self.eval_expr(&args[0])?;

        // Evaluate token IDs
        let tokens_val = self.eval_expr(&args[1])?;

        // Convert tokens to i64 vector, accepting TokenIds, TokenIdArray, or Int (single token)
        let token_data: Vec<i64> = match tokens_val {
            Value::TokenIdArray(ref ids) => {
                // TokenIdArray already has i64 data
                ids.data().to_vec()
            }
            Value::TokenIds(ref ids) => {
                // TokenIds is Vec<u32>, convert to i64
                ids.iter().map(|&id| id as i64).collect()
            }
            Value::Integer(token_id) => {
                // Single token ID
                vec![token_id]
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "embedding() second argument must be TokenIds, TokenIdArray, or Int".to_string(),
                ))
            }
        };

        // Perform embedding lookup based on table type
        // For embedding lookup, we manually select rows from the table based on token IDs
        match table_val {
            Value::TensorF16(table) => {
                use crate::tensor::TensorAccessors;

                // Get table dimensions: [vocab_size, embedding_dim]
                let table_dims = table.shape().dims();
                if table_dims.len() != 2 {
                    return Err(RuntimeError::TypeError(format!(
                        "embedding() table must be 2D, got {:?}",
                        table_dims
                    )));
                }

                let vocab_size = table_dims[0];
                let emb_dim = table_dims[1];

                // Validate token IDs
                for &token_id in &token_data {
                    if token_id < 0 || token_id as usize >= vocab_size {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Token ID {} out of bounds for vocab size {}",
                            token_id, vocab_size
                        )));
                    }
                }

                let seq_len = token_data.len();

                // Check if on Metal device - use GPU implementation if available
                let result = match table.device() {
                    crate::device::Device::Metal(metal_device) => {
                        // GPU implementation
                        if std::env::var("TL_DEBUG_F16_EMBEDDING").is_ok() {
                            eprintln!("DEBUG: Using GPU embedding_lookup_f16 kernel (seq_len={}, emb_dim={})", seq_len, emb_dim);
                        }
                        self.embedding_lookup_metal_f16(
                            &table,
                            &token_data,
                            metal_device,
                            vocab_size,
                            emb_dim,
                            seq_len,
                        )?
                    }
                    _ => {
                        // CPU fallback
                        if std::env::var("TL_DEBUG_F16_EMBEDDING").is_ok() {
                            eprintln!("DEBUG: Using CPU fallback for f16 embedding");
                        }
                        let table_data = table.sync_and_read_f32();
                        let mut output_data = Vec::with_capacity(seq_len * emb_dim);

                        for &token_id in &token_data {
                            let row_start = (token_id as usize) * emb_dim;
                            let row_end = row_start + emb_dim;
                            output_data.extend_from_slice(&table_data[row_start..row_end]);
                        }

                        // Convert back to f16
                        let output_f16: Vec<f16> =
                            output_data.iter().map(|&f| f16::from_f32(f)).collect();

                        Tensor::from_vec(output_f16, vec![seq_len, emb_dim])
                            .map_err(|e| RuntimeError::TensorError(e))?
                    }
                };

                Ok(result.to_value())
            }
            Value::TensorF32(table) => {
                use crate::tensor::TensorAccessors;

                // Get table dimensions: [vocab_size, embedding_dim]
                let table_dims = table.shape().dims();
                if table_dims.len() != 2 {
                    return Err(RuntimeError::TypeError(format!(
                        "embedding() table must be 2D, got {:?}",
                        table_dims
                    )));
                }

                let vocab_size = table_dims[0];
                let emb_dim = table_dims[1];

                // Validate token IDs
                for &token_id in &token_data {
                    if token_id < 0 || token_id as usize >= vocab_size {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Token ID {} out of bounds for vocab size {}",
                            token_id, vocab_size
                        )));
                    }
                }

                let seq_len = token_data.len();

                // Check if on Metal device - use GPU implementation if available
                let result = match table.device() {
                    crate::device::Device::Metal(metal_device) => {
                        // GPU implementation
                        self.embedding_lookup_metal_f32(
                            &table,
                            &token_data,
                            metal_device,
                            vocab_size,
                            emb_dim,
                            seq_len,
                        )?
                    }
                    _ => {
                        // CPU fallback
                        let table_data = table.sync_and_read();
                        let mut output_data = Vec::with_capacity(seq_len * emb_dim);

                        for &token_id in &token_data {
                            let row_start = (token_id as usize) * emb_dim;
                            let row_end = row_start + emb_dim;
                            output_data.extend_from_slice(&table_data[row_start..row_end]);
                        }

                        Tensor::from_vec(output_data, vec![seq_len, emb_dim])
                            .map_err(|e| RuntimeError::TensorError(e))?
                    }
                };

                // Debug: check embedding output
                if std::env::var("TL_DEBUG_ATTN").is_ok() {
                    use crate::tensor::TensorIO;
                    let data = result.sync_and_read();
                    eprintln!("\n=== Embedding Output (f32) ===");
                    eprintln!("Shape: {:?}", result.dims());
                    eprintln!("First 10: {:?}", &data[..data.len().min(10)]);
                    eprintln!("Last 10: {:?}", &data[data.len().saturating_sub(10)..]);
                    eprintln!("Mean: {}", data.iter().sum::<f32>() / data.len() as f32);
                    eprintln!("Min: {}", data.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
                    eprintln!(
                        "Max: {}",
                        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                    );
                }

                Ok(result.to_value())
            }
            _ => Err(RuntimeError::TypeError(
                "embedding() first argument must be tensor (f16 or f32)".to_string(),
            )),
        }
    }

    /// Metal GPU implementation of embedding lookup (f16)
    fn embedding_lookup_metal_f16(
        &mut self,
        table: &crate::tensor::Tensor<half::f16>,
        token_ids: &[i64],
        metal_device: &crate::device::MetalDevice,
        vocab_size: usize,
        emb_dim: usize,
        seq_len: usize,
    ) -> RuntimeResult<crate::tensor::Tensor<half::f16>> {
        use crate::device::MetalBuffer;
        use crate::tensor::{BufferHandle, TensorCreation};

        let mut device = metal_device.clone();

        // Load unified shader library with all kernels
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device
                .load_library(shader_source)
                .map_err(|e| RuntimeError::TensorError(e))?;
        }

        // Get table buffer
        let table_buf = table
            .buffer()
            .as_metal()
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Create token IDs buffer (convert i64 to i32 for Metal)
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_buf = device.metal_device().new_buffer_with_data(
            token_ids_i32.as_ptr() as *const _,
            (token_ids_i32.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let output_numel = seq_len * emb_dim;
        let output_buf = MetalBuffer::<half::f16>::new_uninit(&device, output_numel)
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Create embedding_dim buffer
        let emb_dim_u32 = emb_dim as u32;
        let emb_dim_buf = device.metal_device().new_buffer_with_data(
            &emb_dim_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Execute kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor
            .get_or_compile_pipeline("embedding_lookup_f16")
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Use batched command buffer from Commands system (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()
            .map_err(|e| RuntimeError::TensorError(e.into()))?;
        let encoder = command_buffer.inner().new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&table_buf.buffer), 0);
        encoder.set_buffer(1, Some(&token_ids_buf), 0);
        encoder.set_buffer(2, Some(&output_buf.buffer), 0);
        encoder.set_buffer(3, Some(&emb_dim_buf), 0);

        let grid_size = metal::MTLSize::new(output_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(output_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // NOTE: No commit here - the Commands batching system handles this automatically
        // Commands will commit when batch size is reached or when data is read to CPU
        // This matches Candle's lazy batching strategy

        // Create result tensor first
        let result: crate::tensor::Tensor<half::f16> = crate::tensor::Tensor::new(
            BufferHandle::Metal(unsafe { std::mem::transmute(output_buf) }),
            vec![seq_len, emb_dim].into(),
            crate::device::Device::Metal(device),
        )
        .map_err(|e| RuntimeError::TensorError(e))?;

        // Debug: Read back first few values to verify (after tensor creation)
        if std::env::var("TL_DEBUG_F16_EMBEDDING").is_ok() {
            use crate::tensor::TensorAccessors;
            let data = result.sync_and_read();
            eprintln!("DEBUG F16 embedding result (first 10 values after sync):");
            for i in 0..10.min(data.len()) {
                eprintln!("  [{}]: {}", i, data[i].to_f32());
            }

            // Check full sum
            let full_sum: f32 = data.iter().map(|&x| x.to_f32()).sum();
            eprintln!("DEBUG F16 embedding result: full sum = {}", full_sum);

            // Show values at different positions to diagnose
            if data.len() >= 2048 {
                eprintln!("  Values at positions [100], [500], [1000], [1500], [2000]:");
                for &idx in &[100, 500, 1000, 1500, 2000] {
                    eprintln!("    [{}]: {}", idx, data[idx].to_f32());
                }
            }
        }

        Ok(result)
    }

    /// Metal GPU implementation of embedding lookup (f32)
    fn embedding_lookup_metal_f32(
        &mut self,
        table: &crate::tensor::Tensor<f32>,
        token_ids: &[i64],
        metal_device: &crate::device::MetalDevice,
        vocab_size: usize,
        emb_dim: usize,
        seq_len: usize,
    ) -> RuntimeResult<crate::tensor::Tensor<f32>> {
        use crate::device::MetalBuffer;
        use crate::tensor::{BufferHandle, TensorCreation};

        let mut device = metal_device.clone();

        // Load unified shader library with all kernels
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device
                .load_library(shader_source)
                .map_err(|e| RuntimeError::TensorError(e))?;
        }

        // Get table buffer
        let table_buf = table
            .buffer()
            .as_metal()
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Create token IDs buffer (convert i64 to i32 for Metal)
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_buf = device.metal_device().new_buffer_with_data(
            token_ids_i32.as_ptr() as *const _,
            (token_ids_i32.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let output_numel = seq_len * emb_dim;
        let output_buf = MetalBuffer::<f32>::new_uninit(&device, output_numel)
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Create embedding_dim buffer
        let emb_dim_u32 = emb_dim as u32;
        let emb_dim_buf = device.metal_device().new_buffer_with_data(
            &emb_dim_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Execute kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor
            .get_or_compile_pipeline("embedding_lookup_f32")
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Use batched command buffer from Commands system (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()
            .map_err(|e| RuntimeError::TensorError(e.into()))?;
        let encoder = command_buffer.inner().new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&table_buf.buffer), 0);
        encoder.set_buffer(1, Some(&token_ids_buf), 0);
        encoder.set_buffer(2, Some(&output_buf.buffer), 0);
        encoder.set_buffer(3, Some(&emb_dim_buf), 0);

        let grid_size = metal::MTLSize::new(output_numel as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(output_numel as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // NOTE: No commit here - the Commands batching system handles this automatically
        // Commands will commit when batch size is reached or when data is read to CPU
        // This matches Candle's lazy batching strategy

        // Create result tensor first
        let result: crate::tensor::Tensor<f32> = crate::tensor::Tensor::new(
            BufferHandle::Metal(unsafe { std::mem::transmute(output_buf) }),
            vec![seq_len, emb_dim].into(),
            crate::device::Device::Metal(device),
        )
        .map_err(|e| RuntimeError::TensorError(e))?;

        // Debug: Read back first few values to verify (after tensor creation)
        if std::env::var("TL_DEBUG_F16_EMBEDDING").is_ok() {
            use crate::tensor::TensorAccessors;
            let data = result.sync_and_read();
            eprintln!("DEBUG F32 embedding result (first 10 values after sync):");
            for i in 0..10.min(data.len()) {
                eprintln!("  [{}]: {}", i, data[i]);
            }
        }

        Ok(result)
    }

    /// attention_with_cache(Q, K_cache, V_cache, W_o) -> tensor
    /// Scaled dot-product attention with KV cache (for transformer inference)
    /// Based on candle's llama2_c.rs implementation
    fn eval_attention_with_cache(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 4 {
            return Err(RuntimeError::TypeError(format!(
                "attention_with_cache() expects 4 arguments (Q, K_cache, V_cache, W_o), got {}",
                args.len()
            )));
        }

        let q_val = self.eval_expr(&args[0])?;
        let k_cache_val = self.eval_expr(&args[1])?;
        let v_cache_val = self.eval_expr(&args[2])?;
        let w_o_val = self.eval_expr(&args[3])?;

        // Handle both f32 and f16 tensors
        match (q_val, k_cache_val, v_cache_val, w_o_val) {
            (
                Value::TensorF32(q),
                Value::TensorF32(k),
                Value::TensorF32(v),
                Value::TensorF32(w_o),
            ) => Self::attention_with_cache_f32((*q).clone(), (*k).clone(), (*v).clone(), (*w_o).clone()),
            (
                Value::TensorF16(q),
                Value::TensorF16(k),
                Value::TensorF16(v),
                Value::TensorF16(w_o),
            ) => Self::attention_with_cache_f16((*q).clone(), (*k).clone(), (*v).clone(), (*w_o).clone()),
            _ => Err(RuntimeError::TypeError(
                "attention_with_cache() requires all tensors to be the same type (f32 or f16)"
                    .to_string(),
            )),
        }
    }

    /// Repeat K/V tensors for Grouped Query Attention (GQA)
    /// Repeats each embedding dimension n_rep times to match Q's dimension
    fn repeat_kv_for_gqa<T: FloatType>(
        tensor: &Tensor<T>,
        n_rep: usize,
    ) -> RuntimeResult<Tensor<T>> {
        if n_rep == 1 {
            return Ok(tensor.clone());
        }

        // tensor shape: [seq_len, k_embd]
        // output shape: [seq_len, k_embd * n_rep]
        let dims = tensor.dims();
        let seq_len = dims[0];
        let k_embd = dims[1];
        let new_embd = k_embd * n_rep;

        // Use broadcast to repeat: reshape to [seq_len, k_embd, 1] then broadcast to [seq_len, k_embd, n_rep]
        // Then reshape to [seq_len, k_embd * n_rep]
        use crate::tensor::TensorShape;

        let reshaped = tensor
            .reshape(vec![seq_len, k_embd, 1])
            .map_err(|e| RuntimeError::TensorError(e))?;

        let broadcasted = reshaped
            .broadcast_to(&TensorShape::new(vec![seq_len, k_embd, n_rep]))
            .map_err(|e| RuntimeError::TensorError(e))?;

        // CRITICAL: Make contiguous after broadcast! Broadcast creates non-contiguous tensors
        // with special strides. Must materialize the data before reshape.
        use crate::tensor::TensorTransform;
        let contiguous = broadcasted
            .contiguous()
            .map_err(|e| RuntimeError::TensorError(e))?;

        let result = contiguous
            .reshape(vec![seq_len, new_embd])
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(result)
    }

    fn attention_with_cache_f32(
        q: Tensor<f32>,
        k: Tensor<f32>,
        v: Tensor<f32>,
        w_o: Tensor<f32>,
    ) -> RuntimeResult<Value> {
        let _fn_start = Instant::now();

        // Q shape: [seq_len, n_embd]
        // K shape: [cache_len, n_embd]
        // V shape: [cache_len, n_embd]
        // W_o shape: [n_embd, n_embd]

        let q_dims = q.dims();
        let k_dims = k.dims();
        // eprintln!("[TIMING]   attention_f32: setup: {:.3}ms", _fn_start.elapsed().as_secs_f64() * 1000.0);

        if q_dims.len() != 2 || k_dims.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "Q and K must be 2D tensors, got Q: {:?}, K: {:?}",
                q_dims, k_dims
            )));
        }

        let seq_len = q_dims[0];
        let n_embd = q_dims[1];
        let cache_len = k_dims[0];

        // Debug: check input shapes
        if std::env::var("TL_DEBUG_ATTN").is_ok() {
            use crate::tensor::TensorIO;
            eprintln!("\n=== Attention Input Shapes (f32) ===");
            eprintln!("Q shape: {:?}", q_dims);
            eprintln!("K shape: {:?}", k_dims);
            eprintln!("V shape: {:?}", v.dims());
            eprintln!("W_o shape: {:?}", w_o.dims());
            eprintln!(
                "seq_len: {}, cache_len: {}, n_embd: {}",
                seq_len, cache_len, n_embd
            );

            // Show actual K cache values
            let k_data = k.sync_and_read();
            eprintln!(
                "K cache (first 5 of last row): {:?}",
                &k_data[k_data.len().saturating_sub(5)..]
            );

            // Check if this is DECODE (seq_len=1) or PREFILL (seq_len>1)
            if seq_len > 1 && seq_len == cache_len {
                eprintln!(
                    ">>> PREFILL phase detected (seq_len == cache_len == {})",
                    seq_len
                );
            } else if seq_len == 1 {
                eprintln!(
                    ">>> DECODE phase expected (seq_len=1, cache_len={})",
                    cache_len
                );
            } else {
                eprintln!(
                    ">>> WARNING: Unexpected phase (seq_len={}, cache_len={})",
                    seq_len, cache_len
                );
            }
        }

        // FIX: head_dim should be 64 (n_embd=2048 / n_heads=32 = 64) for TinyLlama
        // Using full n_embd causes incorrect attention scaling
        let n_heads = 32;
        let head_dim = n_embd / n_heads;

        // Handle Grouped Query Attention (GQA): K/V may have fewer dimensions than Q
        // If k_dims[1] < n_embd, we need to repeat K/V to match Q's dimension
        let k_embd = k_dims[1];
        let v_embd = v.dims()[1];

        // Apply RoPE to Q only
        // K/V cache should already have RoPE applied when stored
        // Reshape Q: [seq_len, n_embd] -> [seq_len, n_heads, head_dim]
        let q_heads = q
            .reshape(vec![seq_len, n_heads, head_dim])
            .map_err(|e| RuntimeError::TensorError(e))?;
        let q_rope = q_heads
            .rope(cache_len - seq_len)
            .map_err(|e| RuntimeError::TensorError(e))?; // position_offset
        let q_flat = q_rope
            .reshape(vec![seq_len, n_embd])
            .map_err(|e| RuntimeError::TensorError(e))?;

        // K cache already has RoPE applied, use as-is
        let k_flat = k.clone();

        if std::env::var("TL_DEBUG_ROPE").is_ok() {
            eprintln!("\n=== RoPE Debug (f32) ===");
            eprintln!("  seq_len: {}, cache_len: {}", seq_len, cache_len);
            eprintln!("  Q position_offset: {}", cache_len - seq_len);
            eprintln!("  K position_offset: 0");
            eprintln!("  n_heads: {}, head_dim: {}", n_heads, head_dim);
        }

        let (k_expanded, v_expanded) = if k_embd != n_embd || v_embd != n_embd {
            // GQA: repeat K/V to match Q dimension
            let n_rep = n_embd / k_embd;
            if n_embd % k_embd != 0 {
                return Err(RuntimeError::TypeError(format!(
                    "GQA dimension mismatch: n_embd={} must be divisible by k_embd={}",
                    n_embd, k_embd
                )));
            }

            // Repeat K and V along the embedding dimension using broadcast
            let k_repeated = Self::repeat_kv_for_gqa(&k_flat, n_rep)?;
            let v_repeated = Self::repeat_kv_for_gqa(&v, n_rep)?;
            (k_repeated, v_repeated)
        } else {
            (k_flat, v)
        };

        // Step 1: Attention scores = Q @ K^T / sqrt(head_dim)
        // Q: [seq_len, n_embd], K: [cache_len, n_embd] -> [seq_len, cache_len]
        // Use fused transpose-matmul to avoid creating transposed copy
        let qk_start = Instant::now();
        let scores = q_flat
            .matmul_transposed_b(&k_expanded)
            .map_err(|e| RuntimeError::TensorError(e))?;

        let scale = (head_dim as f32).sqrt();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!(
                "[DEBUG_RS] attention: About to call div_scalar(scale={})...",
                scale
            );
            std::io::stderr().flush().ok();
        }
        let scaled_scores = scores
            .div_scalar(scale)
            .map_err(|e| RuntimeError::TensorError(e))?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] attention: div_scalar returned successfully");
            std::io::stderr().flush().ok();
        }
        // eprintln!("[TIMING]   attention_f32: Q@K^T: {:.3}ms", qk_start.elapsed().as_secs_f64() * 1000.0);

        // Step 2: Apply causal mask if seq_len > 1 (prefill phase)
        let masked_scores = if seq_len > 1 {
            // During prefill, we need causal masking to prevent attending to future tokens
            // Create a causal mask: mask[i, j] = -inf if i < j (upper triangle)
            // scores shape: [seq_len, cache_len]
            // During prefill: cache_len == seq_len

            if cache_len != seq_len {
                // This shouldn't happen during prefill, but handle gracefully
                return Err(RuntimeError::InvalidOperation(format!(
                    "Prefill expects cache_len == seq_len, got cache_len={}, seq_len={}",
                    cache_len, seq_len
                )));
            }

            // Create causal mask: 1 for allowed positions, 0 for masked
            use crate::device::Device;
            use crate::tensor::TensorCreation;
            let mut mask_data = Vec::with_capacity(seq_len * seq_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i {
                        mask_data.push(1.0f32); // Allow attention
                    } else {
                        mask_data.push(0.0f32); // Mask out future positions
                    }
                }
            }

            // Create mask on GPU (same device as Q tensor)
            let mask = match q.device() {
                Device::Metal(dev) => Tensor::from_vec_gpu(dev, mask_data, vec![seq_len, seq_len])
                    .map_err(|e| RuntimeError::TensorError(e))?,
                _ => Tensor::from_vec(mask_data, vec![seq_len, seq_len])
                    .map_err(|e| RuntimeError::TensorError(e))?,
            };

            // Apply mask using GPU kernel (converts 0s to -1e9)
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] attention: About to call apply_attention_mask (mask shape: {:?})...", mask.shape());
                std::io::stderr().flush().ok();
            }
            let result = scaled_scores
                .apply_attention_mask(&mask)
                .map_err(|e| RuntimeError::TensorError(e))?;
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] attention: apply_attention_mask returned successfully");
                std::io::stderr().flush().ok();
            }
            result
        } else {
            // Decode phase (seq_len == 1): no masking needed
            // The single new token can attend to all previous tokens
            scaled_scores
        };

        // Step 3: Softmax over last dimension
        let softmax_start = Instant::now();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!(
                "[DEBUG_RS] attention: About to call softmax (masked_scores shape: {:?})...",
                masked_scores.shape()
            );
            std::io::stderr().flush().ok();
        }

        // Debug: check scaled scores before softmax
        if std::env::var("TL_DEBUG_ATTN").is_ok() {
            use crate::tensor::TensorIO;
            let scores_data = masked_scores.sync_and_read();
            eprintln!("\n=== Pre-Softmax Scores (f32) ===");
            eprintln!("Scores shape: {:?}", masked_scores.shape());
            eprintln!(
                "Scores (first 10): {:?}",
                &scores_data[..scores_data.len().min(10)]
            );
            eprintln!(
                "Scores (last 10): {:?}",
                &scores_data[scores_data.len().saturating_sub(10)..]
            );
            eprintln!(
                "Scores min: {}",
                scores_data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
            );
            eprintln!(
                "Scores max: {}",
                scores_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            );
            eprintln!(
                "Scores mean: {}",
                scores_data.iter().sum::<f32>() / scores_data.len() as f32
            );
        }

        let attn_weights = masked_scores
            .softmax()
            .map_err(|e| RuntimeError::TensorError(e))?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] attention: softmax returned successfully");
            std::io::stderr().flush().ok();
        }
        // eprintln!("[TIMING]   attention_f32: softmax: {:.3}ms", softmax_start.elapsed().as_secs_f64() * 1000.0);

        // Step 4: Weighted sum: attn_weights @ V (use expanded V for GQA)
        // attn_weights: [seq_len, cache_len], V: [cache_len, n_embd] -> [seq_len, n_embd]
        // CRITICAL: Make V contiguous before matmul for performance
        // Following candle: "matmul doesn't support strided vs for now"
        let attn_v_start = Instant::now();
        use crate::tensor::TensorTransform;
        let v_contiguous = v_expanded
            .contiguous()
            .map_err(|e| RuntimeError::TensorError(e))?;
        let attn_out = attn_weights
            .matmul(&v_contiguous)
            .map_err(|e| RuntimeError::TensorError(e))?;
        // eprintln!("[TIMING]   attention_f32: attn@V: {:.3}ms", attn_v_start.elapsed().as_secs_f64() * 1000.0);

        // Step 5: Output projection: attn_out @ W_o.T (like linear layer)
        // Use fused transpose-matmul for better performance (2.89x faster than separate transpose + matmul)
        let output_start = Instant::now();
        let result = attn_out
            .matmul_transposed_b(&w_o)
            .map_err(|e| RuntimeError::TensorError(e))?;
        // eprintln!("[TIMING]   attention_f32: output projection: {:.3}ms", output_start.elapsed().as_secs_f64() * 1000.0);
        // eprintln!("[TIMING]   attention_f32: TOTAL: {:.3}ms", _fn_start.elapsed().as_secs_f64() * 1000.0);
        if std::env::var("TL_PERF").is_ok() {
            eprintln!(
                "[PERF] attention_with_cache(f32): {:.3}ms",
                _fn_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Debug: dump intermediate values
        if std::env::var("TL_DEBUG_ATTN").is_ok() {
            use crate::tensor::TensorIO;
            let attn_w_data = attn_weights.sync_and_read();
            eprintln!("\n=== Attention Debug (f32) ===");
            eprintln!(
                "Attention weights (first 10): {:?}",
                &attn_w_data[..attn_w_data.len().min(10)]
            );
            eprintln!("Attention weights sum: {}", attn_w_data.iter().sum::<f32>());
            eprintln!(
                "Attention weights max: {}",
                attn_w_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            );

            let attn_out_data = attn_out.sync_and_read();
            eprintln!(
                "Attention output (first 10): {:?}",
                &attn_out_data[..attn_out_data.len().min(10)]
            );
            eprintln!(
                "Attention output mean: {}",
                attn_out_data.iter().sum::<f32>() / attn_out_data.len() as f32
            );

            let result_data = result.sync_and_read();
            eprintln!(
                "Final output (first 10): {:?}",
                &result_data[..result_data.len().min(10)]
            );
            eprintln!(
                "Final output mean: {}",
                result_data.iter().sum::<f32>() / result_data.len() as f32
            );
        }

        Ok(Value::TensorF32(Arc::new(result)))
    }

    /* written by ChatGPT

    /// Attention with cache.
    /// - normed: the pre-normalized input used for q/k/v projection
    /// - w_q, w_k, w_v, w_o: weight matrices
    /// - k_cache, v_cache: mutable caches (appended each forward call)
    /// Returns attention output [seq, dim]
    fn attention_with_cache<T: Float>(
        q: &Tensor<T>,
        k_new: &Tensor<T>,
        v_new: &Tensor<T>,
        k_cache: &mut Tensor<T>,
        v_cache: &mut Tensor<T>,
        w_o: &Tensor<T>,
    ) -> Tensor<T> {
        // append new keys/values to cache along axis 0 (sequence axis)
        // caches shape: [cached_seq, dim]
        *k_cache = concatenate(Axis(0), &[k_cache.view(), k_new.view()]).unwrap();
        *v_cache = concatenate(Axis(0), &[v_cache.view(), v_new.view()]).unwrap();

        // Q: [q_len, dim], K_cache: [k_len, dim]
        let dk = q.len_of(Axis(1)) as f32;
        let mut scores = q.dot(&k_cache.t()); // [q_len, k_len]
        scores.mapv_inplace(|s| s / dk.sqrt());
        let weights = softmax_rows(scores);
        let attn = weights.dot(&*v_cache); // [q_len, dim]
                                           // project output
        attn.dot(w_o)
    }
    */

    fn attention_with_cache_f16(
        q: Tensor<f16>,
        k: Tensor<f16>,
        v: Tensor<f16>,
        w_o: Tensor<f16>,
    ) -> RuntimeResult<Value> {
        let _fn_start = Instant::now();

        // Q shape: [seq_len, n_embd]
        // K shape: [cache_len, n_embd]
        // V shape: [cache_len, n_embd]
        // W_o shape: [n_embd, n_embd]

        let q_dims = q.dims();
        let k_dims = k.dims();

        if q_dims.len() != 2 || k_dims.len() != 2 {
            return Err(RuntimeError::TypeError(format!(
                "Q and K must be 2D tensors, got Q: {:?}, K: {:?}",
                q_dims, k_dims
            )));
        }

        let seq_len = q_dims[0];
        let n_embd = q_dims[1];
        let cache_len = k_dims[0];

        // FIX: head_dim should be 64 (n_embd=2048 / n_heads=32 = 64) for TinyLlama
        // Using full n_embd causes incorrect attention scaling
        let n_heads = 32;
        let head_dim = n_embd / n_heads;

        // Handle Grouped Query Attention (GQA): K/V may have fewer dimensions than Q
        // If k_dims[1] < n_embd, we need to repeat K/V to match Q's dimension
        let k_embd = k_dims[1];
        let v_embd = v.dims()[1];

        // Apply RoPE to Q only
        // K/V cache should already have RoPE applied when stored
        // Reshape Q: [seq_len, n_embd] -> [seq_len, n_heads, head_dim]
        let q_heads = q
            .reshape(vec![seq_len, n_heads, head_dim])
            .map_err(|e| RuntimeError::TensorError(e))?;
        let q_rope = q_heads
            .rope(cache_len - seq_len)
            .map_err(|e| RuntimeError::TensorError(e))?; // position_offset
        let q_flat = q_rope
            .reshape(vec![seq_len, n_embd])
            .map_err(|e| RuntimeError::TensorError(e))?;

        // K cache already has RoPE applied, use as-is
        let n_kv_heads = k_embd / head_dim;
        let k_flat = k.clone();

        if std::env::var("TL_DEBUG_ROPE").is_ok() {
            eprintln!("\n=== RoPE Debug (f16) ===");
            eprintln!("  seq_len: {}, cache_len: {}", seq_len, cache_len);
            eprintln!("  Q position_offset: {}", cache_len - seq_len);
            eprintln!("  K position_offset: 0");
            eprintln!(
                "  n_heads: {}, n_kv_heads: {}, head_dim: {}",
                n_heads, n_kv_heads, head_dim
            );
        }

        let (k_expanded, v_expanded) = if k_embd != n_embd || v_embd != n_embd {
            // GQA: repeat K/V to match Q dimension
            let n_rep = n_embd / k_embd;
            if n_embd % k_embd != 0 {
                return Err(RuntimeError::TypeError(format!(
                    "GQA dimension mismatch: n_embd={} must be divisible by k_embd={}",
                    n_embd, k_embd
                )));
            }

            // Repeat K and V along the embedding dimension using broadcast
            let k_repeated = Self::repeat_kv_for_gqa(&k_flat, n_rep)?;
            let v_repeated = Self::repeat_kv_for_gqa(&v, n_rep)?;
            (k_repeated, v_repeated)
        } else {
            (k_flat, v)
        };

        // Step 1: Attention scores = Q @ K^T / sqrt(head_dim)
        // Q: [seq_len, n_embd], K: [cache_len, n_embd] -> [seq_len, cache_len]
        // Use fused transpose-matmul to avoid creating transposed copy + contiguous call
        // This is the key optimization: matmul_transposed_b transposes K on-the-fly during matmul
        let scores = q_flat
            .matmul_transposed_b(&k_expanded)
            .map_err(|e| RuntimeError::TensorError(e))?;

        let scale = (head_dim as f32).sqrt();
        let scaled_scores = scores
            .div_scalar(f16::from_f32(scale))
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Step 2: Apply causal mask if seq_len > 1 (prefill phase)
        let masked_scores = if seq_len > 1 {
            // During prefill, we need causal masking to prevent attending to future tokens
            // Create a causal mask: mask[i, j] = -inf if i < j (upper triangle)
            // scores shape: [seq_len, cache_len]
            // During prefill: cache_len == seq_len

            if cache_len != seq_len {
                // This shouldn't happen during prefill, but handle gracefully
                return Err(RuntimeError::InvalidOperation(format!(
                    "Prefill expects cache_len == seq_len, got cache_len={}, seq_len={}",
                    cache_len, seq_len
                )));
            }

            if std::env::var("TL_DEBUG_F16_MASK").is_ok() {
                eprintln!(
                    "  [F16 MASK DEBUG] Creating causal mask: seq_len={}",
                    seq_len
                );
            }

            // Create causal mask: 1 for allowed positions, 0 for masked
            use crate::device::Device;
            use crate::tensor::TensorCreation;
            let mut mask_data = Vec::with_capacity(seq_len * seq_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i {
                        mask_data.push(f16::from_f32(1.0)); // Allow attention
                    } else {
                        mask_data.push(f16::from_f32(0.0)); // Mask out future positions
                    }
                }
            }

            if std::env::var("TL_DEBUG_F16_MASK").is_ok() {
                eprintln!("  [F16 MASK DEBUG] Uploading mask to GPU...");
            }

            // Create mask on GPU (same device as Q tensor)
            let mask = match q_flat.device() {
                Device::Metal(dev) => Tensor::from_vec_gpu(dev, mask_data, vec![seq_len, seq_len])
                    .map_err(|e| RuntimeError::TensorError(e))?,
                _ => Tensor::from_vec(mask_data, vec![seq_len, seq_len])
                    .map_err(|e| RuntimeError::TensorError(e))?,
            };

            if std::env::var("TL_DEBUG_F16_MASK").is_ok() {
                eprintln!("  [F16 MASK DEBUG] Applying mask...");
            }

            // Apply mask using GPU kernel (converts 0s to -1e4)
            let result = scaled_scores
                .apply_attention_mask(&mask)
                .map_err(|e| RuntimeError::TensorError(e))?;

            if std::env::var("TL_DEBUG_F16_MASK").is_ok() {
                eprintln!("  [F16 MASK DEBUG] Mask applied successfully");
            }

            result
        } else {
            // Decode phase (seq_len == 1): no masking needed
            // The single new token can attend to all previous tokens
            scaled_scores
        };

        // Step 3: Softmax over last dimension
        let attn_weights = masked_scores
            .softmax()
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Step 4: Weighted sum: attn_weights @ V (use expanded V for GQA)
        // attn_weights: [seq_len, cache_len], V: [cache_len, n_embd] -> [seq_len, n_embd]
        // CRITICAL: Make V contiguous before matmul for performance
        // Following candle: "matmul doesn't support strided vs for now"
        use crate::tensor::TensorTransform;
        let v_contiguous = v_expanded
            .contiguous()
            .map_err(|e| RuntimeError::TensorError(e))?;
        let attn_out = attn_weights
            .matmul(&v_contiguous)
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Step 5: Output projection: attn_out @ W_o.T (like linear layer)
        // Use fused transpose-matmul for better performance (2.89x faster than separate transpose + matmul)
        let result = attn_out
            .matmul_transposed_b(&w_o)
            .map_err(|e| RuntimeError::TensorError(e))?;

        if std::env::var("TL_DEBUG_ACTIVATION").is_ok() {
            use crate::tensor::TensorIO;
            let data: Vec<f32> = result.sync_and_read_f32();
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            eprintln!("\n=== Attention Output (f16) ===");
            eprintln!("  Shape: {:?}", result.dims());
            eprintln!("  Range: [{:.4}, {:.4}]", min, max);
            eprintln!("  Mean: {:.4}, AbsMax: {:.4}", mean, abs_max);

            // Check for NaN or Inf
            let nan_count = data.iter().filter(|x| x.is_nan()).count();
            let inf_count = data.iter().filter(|x| x.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                eprintln!("  ⚠️  WARNING: NaN={}, Inf={}", nan_count, inf_count);
            }
        }

        if std::env::var("TL_PERF").is_ok() {
            eprintln!(
                "[PERF] attention_with_cache(f16): {:.3}ms",
                _fn_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Value::TensorF16(Arc::new(result)))
    }
}
