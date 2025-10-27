//! Neural network operations for TensorLogic interpreter

use super::*;
use crate::interpreter::value::ToValue;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_nn_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "rms_norm" => Some(self.eval_rms_norm(args)),
            "layer_norm" => Some(self.eval_layer_norm(args)),
            "positional_encoding" => Some(self.eval_positional_encoding(args)),
            "apply_attention_mask" => Some(self.eval_apply_attention_mask(args)),
            "padding_mask" => Some(self.eval_padding_mask(args)),
            "combine_masks" => Some(self.eval_combine_masks(args)),
            "fused_add_relu" => Some(self.eval_fused_add_relu(args)),
            "fused_mul_relu" => Some(self.eval_fused_mul_relu(args)),
            "fused_affine" => Some(self.eval_fused_affine(args)),
            "fused_gelu_linear" => Some(self.eval_fused_gelu_linear(args)),
            "batch_norm" | "dropout" |
            "apply_mask" | "causal_mask" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("NN function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// rms_norm(x, weight, [eps]) -> tensor
    /// RMS Normalization (used in LLaMA, TinyLlama)
    fn eval_rms_norm(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() < 2 || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("rms_norm() expects 2-3 arguments (tensor, weight, optional eps), got {}", args.len())
            ));
        }

        // Evaluate tensor and weight arguments
        let tensor_val = self.eval_expr(&args[0])?;
        let weight_val = self.eval_expr(&args[1])?;

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
        match (tensor_val, weight_val) {
            (Value::TensorF16(tensor), Value::TensorF16(weight)) => {
                let normalized_shape = weight.dims().to_vec();
                let result = tensor.rms_norm(normalized_shape, &weight, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(tensor), Value::TensorF32(weight)) => {
                let normalized_shape = weight.dims().to_vec();
                let result = tensor.rms_norm(normalized_shape, &weight, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            _ => Err(RuntimeError::TypeError(
                "rms_norm() requires tensor and weight to be same type (both f16 or both f32)".to_string()
            )),
        }
    }

    /// positional_encoding(seq_len, d_model) -> tensor
    /// Generates sinusoidal positional encoding
    fn eval_positional_encoding(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("positional_encoding() expects 2 arguments (seq_len, d_model), got {}", args.len())
            ));
        }

        let seq_len = match self.eval_expr(&args[0])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            v => return Err(RuntimeError::TypeError(
                format!("positional_encoding() first argument must be a number (seq_len), got {:?}", v)
            )),
        };

        let d_model = match self.eval_expr(&args[1])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            v => return Err(RuntimeError::TypeError(
                format!("positional_encoding() second argument must be a number (d_model), got {:?}", v)
            )),
        };

        // Generate sinusoidal positional encoding
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        let mut pe_data = Vec::with_capacity(seq_len * d_model);

        for pos in 0..seq_len {
            for i in 0..d_model {
                let div_term = (i as f32 / d_model as f32) * 10000_f32.ln();
                let angle = pos as f32 / div_term.exp();

                let value = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };

                pe_data.push(half::f16::from_f32(value));
            }
        }

        // Create output tensor
        let output = crate::tensor::Tensor::from_vec_metal(
            self.env.metal_device(),
            pe_data,
            vec![seq_len, d_model]
        ).map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TensorF16(output))
    }

    /// layer_norm(tensor, [normalized_shape], [eps]) -> tensor
    /// Layer Normalization
    fn eval_layer_norm(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.is_empty() || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("layer_norm() expects 1-3 arguments (tensor, optional normalized_shape, optional eps), got {}", args.len())
            ));
        }

        let tensor_val = self.eval_expr(&args[0])?;

        let eps = if args.len() >= 3 {
            match self.eval_expr(&args[2])? {
                Value::Float(f) => f as f32,
                Value::Integer(i) => i as f32,
                _ => 1e-5_f32,
            }
        } else {
            1e-5_f32
        };

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                let shape = tensor.shape();
                let dims = shape.dims();
                let normalized_shape = vec![dims[dims.len() - 1]];
                tensor.layer_norm(normalized_shape, None, None, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(tensor) => {
                let shape = tensor.shape();
                let dims = shape.dims();
                let normalized_shape = vec![dims[dims.len() - 1]];
                tensor.layer_norm(normalized_shape, None, None, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("Expected tensor".to_string()))
        })
    }

    /// apply_attention_mask(scores, mask) -> tensor
    /// Apply attention mask to scores
    /// mask: 1 = keep, 0 = mask out (set to -1e9)
    fn eval_apply_attention_mask(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("apply_attention_mask() expects 2 arguments (scores, mask), got {}", args.len())
            ));
        }

        let scores_val = self.eval_expr(&args[0])?;
        let mask_val = self.eval_expr(&args[1])?;

        let device = self.env.metal_device();

        Ok(match (scores_val, mask_val) {
            (Value::TensorF16(scores), Value::TensorF16(mask)) => {
                // Apply mask: scores + (1 - mask) * (-1e9)
                let ones = Tensor::ones(device, scores.shape().dims().to_vec())
                    .map_err(|e| RuntimeError::TensorError(e))?;
                let inv_mask = ones.sub(&mask).map_err(|e| RuntimeError::TensorError(e))?;

                let large_neg_value = half::f16::from_f32(-1e9);
                let large_neg_vec = vec![large_neg_value; inv_mask.shape().numel()];
                let large_neg = Tensor::from_vec_metal(device, large_neg_vec, inv_mask.shape().dims().to_vec())
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
                let large_neg = Tensor::from_vec_metal(device, large_neg_vec, inv_mask.shape().dims().to_vec())
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
            return Err(RuntimeError::TypeError(
                format!("padding_mask() expects 2 arguments (lengths, max_length), got {}", args.len())
            ));
        }

        let lengths_val = self.eval_expr(&args[0])?;
        let max_length = match self.eval_expr(&args[1])? {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            _ => return Err(RuntimeError::TypeError("max_length must be an integer".to_string())),
        };

        let device = self.env.metal_device();

        Ok(match lengths_val {
            Value::TensorF16(lengths_tensor) => {
                let lengths_vec = lengths_tensor.to_vec();
                let lengths: Vec<usize> = lengths_vec.iter()
                    .map(|&v| v.to_f32() as usize)
                    .collect();

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

                Tensor::from_vec_metal(device, mask_data, vec![batch_size, max_length])
                    .map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            Value::TensorF32(lengths_tensor) => {
                let lengths_vec = lengths_tensor.to_vec_f32();
                let lengths: Vec<usize> = lengths_vec.iter()
                    .map(|&v| v as usize)
                    .collect();

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

                Tensor::from_vec_metal(device, mask_data, vec![batch_size, max_length])
                    .map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError("padding_mask() expects a tensor for lengths".to_string()))
        })
    }

    /// combine_masks(mask1, mask2) -> tensor
    /// Combine two masks using element-wise multiplication (logical AND)
    fn eval_combine_masks(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("combine_masks() expects 2 arguments (mask1, mask2), got {}", args.len())
            ));
        }

        let mask1_val = self.eval_expr(&args[0])?;
        let mask2_val = self.eval_expr(&args[1])?;

        Ok(match (mask1_val, mask2_val) {
            (Value::TensorF16(mask1), Value::TensorF16(mask2)) => {
                mask1.mul(&mask2).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(mask1), Value::TensorF32(mask2)) => {
                mask1.mul(&mask2).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "combine_masks() requires both masks to be same type (both f16 or both f32)".to_string()
            ))
        })
    }

    /// fused_add_relu(a, b) -> tensor
    /// Fused operation: ReLU(a + b)
    fn eval_fused_add_relu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("fused_add_relu() expects 2 arguments (a, b), got {}", args.len())
            ));
        }

        let a_val = self.eval_expr(&args[0])?;
        let b_val = self.eval_expr(&args[1])?;

        Ok(match (a_val, b_val) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let sum = a.add(&b).map_err(|e| RuntimeError::TensorError(e))?;
                sum.relu().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let sum = a.add(&b).map_err(|e| RuntimeError::TensorError(e))?;
                sum.relu().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "fused_add_relu() requires both tensors to be same type (both f16 or both f32)".to_string()
            ))
        })
    }

    /// fused_mul_relu(a, b) -> tensor
    /// Fused operation: ReLU(a * b)
    fn eval_fused_mul_relu(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("fused_mul_relu() expects 2 arguments (a, b), got {}", args.len())
            ));
        }

        let a_val = self.eval_expr(&args[0])?;
        let b_val = self.eval_expr(&args[1])?;

        Ok(match (a_val, b_val) {
            (Value::TensorF16(a), Value::TensorF16(b)) => {
                let product = a.mul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                product.relu().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(a), Value::TensorF32(b)) => {
                let product = a.mul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                product.relu().map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "fused_mul_relu() requires both tensors to be same type (both f16 or both f32)".to_string()
            ))
        })
    }

    /// fused_affine(x, scale, bias) -> tensor
    /// Fused operation: x * scale + bias
    fn eval_fused_affine(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("fused_affine() expects 3 arguments (x, scale, bias), got {}", args.len())
            ));
        }

        let x_val = self.eval_expr(&args[0])?;
        let scale_val = self.eval_expr(&args[1])?;
        let bias_val = self.eval_expr(&args[2])?;

        Ok(match (x_val, scale_val, bias_val) {
            (Value::TensorF16(x), Value::TensorF16(scale), Value::TensorF16(bias)) => {
                let scaled = x.mul(&scale).map_err(|e| RuntimeError::TensorError(e))?;
                scaled.add(&bias).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(x), Value::TensorF32(scale), Value::TensorF32(bias)) => {
                let scaled = x.mul(&scale).map_err(|e| RuntimeError::TensorError(e))?;
                scaled.add(&bias).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "fused_affine() requires all tensors to be same type (all f16 or all f32)".to_string()
            ))
        })
    }

    /// fused_gelu_linear(input, weight, bias) -> tensor
    /// Fused operation: GELU(input) @ weight + bias
    fn eval_fused_gelu_linear(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("fused_gelu_linear() expects 3 arguments (input, weight, bias), got {}", args.len())
            ));
        }

        let input_val = self.eval_expr(&args[0])?;
        let weight_val = self.eval_expr(&args[1])?;
        let bias_val = self.eval_expr(&args[2])?;

        Ok(match (input_val, weight_val, bias_val) {
            (Value::TensorF16(input), Value::TensorF16(weight), Value::TensorF16(bias)) => {
                let activated = input.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                let linear = activated.matmul(&weight).map_err(|e| RuntimeError::TensorError(e))?;
                linear.add(&bias).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            (Value::TensorF32(input), Value::TensorF32(weight), Value::TensorF32(bias)) => {
                let activated = input.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                let linear = activated.matmul(&weight).map_err(|e| RuntimeError::TensorError(e))?;
                linear.add(&bias).map_err(|e| RuntimeError::TensorError(e))?.to_value()
            }
            _ => return Err(RuntimeError::TypeError(
                "fused_gelu_linear() requires all tensors to be same type (all f16 or all f32)".to_string()
            ))
        })
    }
}
