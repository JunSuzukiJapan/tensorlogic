//! Candle-based operations for TensorLogic interpreter
//! All functions in this module use the Cndl type to organize Candle-based operations.

use super::*;
use candle_core::{DType, Device as CandleDevice, Tensor as CandleTensor};
use candle_nn::ops;
use std::sync::Arc;

/// Cndl: A namespace for Candle-based operations
pub struct Cndl;

impl Interpreter {
    pub(super) fn eval_candle_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            // Tensor operations
            "cndl_matmul" => Some(Cndl::matmul(self, args)),
            "cndl_transpose" => Some(Cndl::transpose(self, args)),
            "cndl_reshape" => Some(Cndl::reshape(self, args)),

            // Math operations
            "cndl_softmax" => Some(Cndl::softmax(self, args)),
            "cndl_log_softmax" => Some(Cndl::log_softmax(self, args)),

            // Activation functions
            "cndl_gelu" => Some(Cndl::gelu(self, args)),
            "cndl_silu" => Some(Cndl::silu(self, args)),
            "cndl_relu" => Some(Cndl::relu(self, args)),
            "cndl_tanh" => Some(Cndl::tanh(self, args)),

            // Neural network operations
            "cndl_layer_norm" => Some(Cndl::layer_norm(self, args)),
            "cndl_rms_norm" => Some(Cndl::rms_norm(self, args)),
            "cndl_embedding" => Some(Cndl::embedding(self, args)),

            // Position embeddings
            "cndl_rope" => Some(Cndl::rope(self, args)),

            // Model loading and saving
            "cndl_load_safetensor" => Some(Cndl::load_safetensor(self, args)),
            "cndl_save_safetensor" => Some(Cndl::save_safetensor(self, args)),
            "cndl_list_safetensors" => Some(Cndl::list_safetensors(self, args)),
            "cndl_load_gguf_tensor" => Some(Cndl::load_gguf_tensor(self, args)),
            "cndl_list_gguf_tensors" => Some(Cndl::list_gguf_tensors(self, args)),

            // Model save/load (full models with multiple tensors)
            "cndl_save_model_safetensor" => Some(Cndl::save_model_safetensor(self, args)),
            "cndl_load_model_safetensor" => Some(Cndl::load_model_safetensor(self, args)),

            _ => None,
        }
    }

}

// ============================================================================
// Cndl implementation
// ============================================================================

impl Cndl {
    // ============================================================================
    // Helper functions for converting between TensorLogic and Candle tensors
    // ============================================================================

    /// Convert TensorLogic Tensor<f32> to Candle Tensor
    fn tl_to_candle_f32(interpreter: &Interpreter, tensor: &crate::tensor::Tensor<f32>) -> RuntimeResult<CandleTensor> {
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let shape = tensor.dims();
        let data = tensor.buffer().to_cpu_vec();

        CandleTensor::from_vec(data, shape, &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle tensor: {}", e))
            ))
    }

    /// Convert TensorLogic Tensor<f16> to Candle Tensor
    fn tl_to_candle_f16(interpreter: &Interpreter, tensor: &crate::tensor::Tensor<half::f16>) -> RuntimeResult<CandleTensor> {
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let shape = tensor.dims();
        let data = tensor.buffer().to_cpu_vec();

        // Convert f16 to f32 for Candle
        let data_f32: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();

        CandleTensor::from_vec(data_f32, shape, &device)
            .and_then(|t| t.to_dtype(DType::F16))
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle tensor: {}", e))
            ))
    }

    /// Convert Candle Tensor to TensorLogic Tensor<f32>
    fn candle_to_tl_f32(interpreter: &Interpreter, tensor: CandleTensor) -> RuntimeResult<crate::tensor::Tensor<f32>> {
        use crate::tensor::TensorCreation;

        let shape = tensor.dims().to_vec();
        let data = tensor.to_vec1::<f32>()
            .or_else(|_| {
                // If not 1D, flatten and get the data
                tensor.flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
            })
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to convert Candle tensor to vec: {}", e))
            ))?;

        let device = interpreter.env.metal_device();
        crate::tensor::Tensor::<f32>::from_vec_gpu(device, data, shape)
            .map_err(|e| RuntimeError::TensorError(e))
    }

    /// Convert Candle Tensor to TensorLogic Tensor<f16>
    fn candle_to_tl_f16(interpreter: &Interpreter, tensor: CandleTensor) -> RuntimeResult<crate::tensor::Tensor<half::f16>> {
        use crate::tensor::TensorCreation;

        let shape = tensor.dims().to_vec();

        // Convert to f16 if needed
        let tensor_f16 = if tensor.dtype() != DType::F16 {
            tensor.to_dtype(DType::F16)
                .map_err(|e| RuntimeError::TensorError(
                    crate::error::TensorError::InvalidOperation(format!("Failed to convert to f16: {}", e))
                ))?
        } else {
            tensor
        };

        let data_f32 = tensor_f16.to_vec1::<f32>()
            .or_else(|_| {
                tensor_f16.flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
            })
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to convert Candle tensor to vec: {}", e))
            ))?;

        // Convert f32 to f16
        let data_f16: Vec<half::f16> = data_f32.iter().map(|x| half::f16::from_f32(*x)).collect();

        let device = interpreter.env.metal_device();
        crate::tensor::Tensor::<half::f16>::from_vec_gpu(device, data_f16, shape)
            .map_err(|e| RuntimeError::TensorError(e))
    }

    // ============================================================================
    // Candle-based operations
    // ============================================================================

    /// cndl_matmul(a, b) -> tensor
    /// Matrix multiplication using Candle: a @ b
    pub fn matmul(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_matmul() expects 2 arguments (a, b), got {}", args.len())
            ));
        }

        let a_val = interpreter.eval_expr(&args[0])?;
        let b_val = interpreter.eval_expr(&args[1])?;

        match (a_val, b_val) {
            (Value::TensorF32(ref a), Value::TensorF32(ref b)) => {
                let a_candle = Self::tl_to_candle_f32(interpreter, a)?;
                let b_candle = Self::tl_to_candle_f32(interpreter, b)?;

                let result = a_candle.matmul(&b_candle)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle matmul failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            (Value::TensorF16(ref a), Value::TensorF16(ref b)) => {
                let a_candle = Self::tl_to_candle_f16(interpreter, a)?;
                let b_candle = Self::tl_to_candle_f16(interpreter, b)?;

                let result = a_candle.matmul(&b_candle)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle matmul failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError(
                "cndl_matmul() requires both tensors to be same type (both f16 or both f32)".to_string()
            ))
        }
    }

    /// cndl_transpose(x, dim0, dim1) -> tensor
    /// Transpose two dimensions using Candle
    pub fn transpose(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("cndl_transpose() expects 3 arguments (x, dim0, dim1), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let dim0_val = interpreter.eval_expr(&args[1])?;
        let dim1_val = interpreter.eval_expr(&args[2])?;

        let dim0 = match dim0_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("dim0 must be an integer".to_string())),
        };

        let dim1 = match dim1_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("dim1 must be an integer".to_string())),
        };

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = x_candle.transpose(dim0, dim1)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle transpose failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = x_candle.transpose(dim0, dim1)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle transpose failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_transpose() requires a tensor".to_string()))
        }
    }

    /// cndl_softmax(x, dim) -> tensor
    /// Softmax operation using Candle
    pub fn softmax(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_softmax() expects 2 arguments (x, dim), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let dim_val = interpreter.eval_expr(&args[1])?;

        let dim = match dim_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("dim must be an integer".to_string())),
        };

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = candle_nn::ops::softmax(&x_candle, dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle softmax failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = candle_nn::ops::softmax(&x_candle, dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle softmax failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_softmax() requires a tensor".to_string()))
        }
    }

    /// cndl_log_softmax(x, dim) -> tensor
    /// Log softmax operation using Candle
    pub fn log_softmax(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_log_softmax() expects 2 arguments (x, dim), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let dim_val = interpreter.eval_expr(&args[1])?;

        let dim = match dim_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError("dim must be an integer".to_string())),
        };

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = candle_nn::ops::log_softmax(&x_candle, dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle log_softmax failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = candle_nn::ops::log_softmax(&x_candle, dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle log_softmax failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_log_softmax() requires a tensor".to_string()))
        }
    }

    /// cndl_gelu(x) -> tensor
    /// GELU activation using Candle
    pub fn gelu(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_gelu() expects 1 argument (x), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = x_candle.gelu()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle gelu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = x_candle.gelu()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle gelu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_gelu() requires a tensor".to_string()))
        }
    }

    /// cndl_silu(x) -> tensor
    /// SiLU (Swish) activation using Candle
    pub fn silu(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_silu() expects 1 argument (x), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = candle_nn::ops::silu(&x_candle)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle silu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = candle_nn::ops::silu(&x_candle)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle silu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_silu() requires a tensor".to_string()))
        }
    }

    /// cndl_relu(x) -> tensor
    /// ReLU activation using Candle
    pub fn relu(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_relu() expects 1 argument (x), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = x_candle.relu()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle relu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = x_candle.relu()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle relu failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_relu() requires a tensor".to_string()))
        }
    }

    /// cndl_tanh(x) -> tensor
    /// Tanh activation using Candle
    pub fn tanh(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_tanh() expects 1 argument (x), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = x_candle.tanh()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle tanh failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = x_candle.tanh()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle tanh failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_tanh() requires a tensor".to_string()))
        }
    }

    /// cndl_layer_norm(x, normalized_shape, weight, bias, eps) -> tensor
    /// Layer normalization using Candle
    pub fn layer_norm(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() < 2 || args.len() > 5 {
            return Err(RuntimeError::TypeError(
                format!("cndl_layer_norm() expects 2-5 arguments (x, normalized_shape, [weight], [bias], [eps]), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let normalized_shape_val = interpreter.eval_expr(&args[1])?;

        let normalized_shape = match normalized_shape_val {
            Value::Integer(i) => vec![i as usize],
            Value::TensorF32(ref t) => {
                let data = t.buffer().to_cpu_vec();
                data.iter().map(|&v| v as usize).collect()
            }
            Value::TensorF16(ref t) => {
                let data = t.buffer().to_cpu_vec();
                data.iter().map(|&v| v.to_f32() as usize).collect()
            }
            _ => return Err(RuntimeError::TypeError("normalized_shape must be an integer or tensor".to_string())),
        };

        let eps = if args.len() >= 5 {
            let eps_val = interpreter.eval_expr(&args[4])?;
            match eps_val {
                Value::Float(f) => f,
                Value::Integer(i) => i as f64,
                _ => 1e-5,
            }
        } else {
            1e-5
        };

        // TODO: Implement proper layer norm with weight and bias
        // For now, just normalize without weight/bias
        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;

                // Simple layer norm: (x - mean) / sqrt(var + eps)
                let dims = x_candle.dims();
                let last_dim = dims.len() - 1;

                let mean = x_candle.mean_keepdim(last_dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Mean calculation failed: {}", e))
                    ))?;

                let x_centered = x_candle.broadcast_sub(&mean)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast sub failed: {}", e))
                    ))?;

                let variance = x_centered.sqr()
                    .and_then(|v| v.mean_keepdim(last_dim))
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Variance calculation failed: {}", e))
                    ))?;

                let std = (variance + eps)
                    .and_then(|v| v.sqrt())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Std calculation failed: {}", e))
                    ))?;

                let result = x_centered.broadcast_div(&std)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast div failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;

                let dims = x_candle.dims();
                let last_dim = dims.len() - 1;

                let mean = x_candle.mean_keepdim(last_dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Mean calculation failed: {}", e))
                    ))?;

                let x_centered = x_candle.broadcast_sub(&mean)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast sub failed: {}", e))
                    ))?;

                let variance = x_centered.sqr()
                    .and_then(|v| v.mean_keepdim(last_dim))
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Variance calculation failed: {}", e))
                    ))?;

                let std = (variance + eps)
                    .and_then(|v| v.sqrt())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Std calculation failed: {}", e))
                    ))?;

                let result = x_centered.broadcast_div(&std)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast div failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_layer_norm() requires a tensor".to_string()))
        }
    }

    /// cndl_rms_norm(x, weight, eps) -> tensor
    /// RMS normalization using Candle
    pub fn rms_norm(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() < 1 || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("cndl_rms_norm() expects 1-3 arguments (x, [weight], [eps]), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;

        let eps = if args.len() >= 3 {
            let eps_val = interpreter.eval_expr(&args[2])?;
            match eps_val {
                Value::Float(f) => f,
                Value::Integer(i) => i as f64,
                _ => 1e-5,
            }
        } else {
            1e-5
        };

        // RMS norm: x / sqrt(mean(x^2) + eps)
        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;

                let dims = x_candle.dims();
                let last_dim = dims.len() - 1;

                let squared = x_candle.sqr()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Square failed: {}", e))
                    ))?;

                let mean = squared.mean_keepdim(last_dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Mean calculation failed: {}", e))
                    ))?;

                let rms = (mean + eps)
                    .and_then(|v| v.sqrt())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("RMS calculation failed: {}", e))
                    ))?;

                let result = x_candle.broadcast_div(&rms)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast div failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;

                let dims = x_candle.dims();
                let last_dim = dims.len() - 1;

                let squared = x_candle.sqr()
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Square failed: {}", e))
                    ))?;

                let mean = squared.mean_keepdim(last_dim)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Mean calculation failed: {}", e))
                    ))?;

                let rms = (mean + eps)
                    .and_then(|v| v.sqrt())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("RMS calculation failed: {}", e))
                    ))?;

                let result = x_candle.broadcast_div(&rms)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Broadcast div failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_rms_norm() requires a tensor".to_string()))
        }
    }

    /// cndl_embedding(indices, embeddings) -> tensor
    /// Embedding lookup using Candle
    pub fn embedding(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_embedding() expects 2 arguments (indices, embeddings), got {}", args.len())
            ));
        }

        let indices_val = interpreter.eval_expr(&args[0])?;
        let embeddings_val = interpreter.eval_expr(&args[1])?;

        match (indices_val, embeddings_val) {
            (Value::TensorF32(ref indices), Value::TensorF32(ref embeddings)) => {
                let indices_candle = Self::tl_to_candle_f32(interpreter, indices)?;
                let embeddings_candle = Self::tl_to_candle_f32(interpreter, embeddings)?;

                // Convert indices to u32
                let indices_u32 = indices_candle.to_dtype(DType::U32)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Failed to convert indices to u32: {}", e))
                    ))?;

                let result = embeddings_candle.embedding(&indices_u32)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle embedding failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            (Value::TensorF16(ref indices), Value::TensorF16(ref embeddings)) => {
                let indices_candle = Self::tl_to_candle_f16(interpreter, indices)?;
                let embeddings_candle = Self::tl_to_candle_f16(interpreter, embeddings)?;

                let indices_u32 = indices_candle.to_dtype(DType::U32)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Failed to convert indices to u32: {}", e))
                    ))?;

                let result = embeddings_candle.embedding(&indices_u32)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle embedding failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            (Value::Integer(idx), Value::TensorF32(ref embeddings)) => {
                let embeddings_candle = Self::tl_to_candle_f32(interpreter, embeddings)?;

                let result = embeddings_candle.get(idx as usize)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle embedding failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            (Value::Integer(idx), Value::TensorF16(ref embeddings)) => {
                let embeddings_candle = Self::tl_to_candle_f16(interpreter, embeddings)?;

                let result = embeddings_candle.get(idx as usize)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle embedding failed: {}", e))
                    ))?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError(
                "cndl_embedding() requires indices and embeddings tensors of same type".to_string()
            ))
        }
    }

    /// cndl_rope(x, position_ids, rope_theta) -> tensor
    /// Rotary Position Embedding using Candle
    pub fn rope(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() < 2 || args.len() > 3 {
            return Err(RuntimeError::TypeError(
                format!("cndl_rope() expects 2-3 arguments (x, position_ids, [rope_theta]), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let position_ids_val = interpreter.eval_expr(&args[1])?;

        let rope_theta = if args.len() >= 3 {
            let theta_val = interpreter.eval_expr(&args[2])?;
            match theta_val {
                Value::Float(f) => f as f32,
                Value::Integer(i) => i as f32,
                _ => 10000.0,
            }
        } else {
            10000.0
        };

        // Get position offset
        let position_offset = match position_ids_val {
            Value::Integer(i) => i as usize,
            Value::TensorF32(ref t) if t.numel() == 1 => {
                interpreter.tensor_f32_to_scalar(t)? as usize
            }
            Value::TensorF16(ref t) if t.numel() == 1 => {
                interpreter.tensor_f16_to_scalar(t)? as usize
            }
            _ => return Err(RuntimeError::TypeError(
                "position_ids must be a scalar integer or single-element tensor".to_string()
            )),
        };

        // Use Candle's RoPE implementation
        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;

                // Apply RoPE using Candle
                let dims = x_candle.dims();
                if dims.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("RoPE requires at least 3D tensor (got {}D)", dims.len())
                    ));
                }

                let head_dim = dims[dims.len() - 1];

                // Create cos/sin cache for RoPE
                let cos_sin = Self::create_rope_cache_candle(head_dim, rope_theta, position_offset + 1)?;

                // Apply RoPE rotation
                let result = Self::apply_rope_candle(x_candle, &cos_sin, position_offset)?;

                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;

                let dims = x_candle.dims();
                if dims.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("RoPE requires at least 3D tensor (got {}D)", dims.len())
                    ));
                }

                let head_dim = dims[dims.len() - 1];

                let cos_sin = Self::create_rope_cache_candle(head_dim, rope_theta, position_offset + 1)?;
                let result = Self::apply_rope_candle(x_candle, &cos_sin, position_offset)?;

                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_rope() requires a tensor".to_string()))
        }
    }

    /// Helper: Create RoPE cos/sin cache
    fn create_rope_cache_candle( head_dim: usize, theta: f32, max_seq_len: usize) -> RuntimeResult<(CandleTensor, CandleTensor)> {
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let half_dim = head_dim / 2;

        // Create frequency bands: theta^(-2i/d) for i in [0, d/2)
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exp = -2.0 * (i as f32) / (head_dim as f32);
                theta.powf(exp)
            })
            .collect();

        let freqs_tensor = CandleTensor::from_vec(freqs, (half_dim,), &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create freqs tensor: {}", e))
            ))?;

        // Create position indices: [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions_tensor = CandleTensor::from_vec(positions, (max_seq_len,), &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create positions tensor: {}", e))
            ))?;

        // Compute angles: outer product of positions and freqs
        let angles = positions_tensor
            .unsqueeze(1)
            .and_then(|p| p.matmul(&freqs_tensor.unsqueeze(0)?))
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to compute angles: {}", e))
            ))?;

        // Compute cos and sin
        let cos = angles.cos()
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to compute cos: {}", e))
            ))?;

        let sin = angles.sin()
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to compute sin: {}", e))
            ))?;

        Ok((cos, sin))
    }

    /// Helper: Apply RoPE rotation
    fn apply_rope_candle( x: CandleTensor, cos_sin: &(CandleTensor, CandleTensor), position_offset: usize) -> RuntimeResult<CandleTensor> {
        let (cos, sin) = cos_sin;

        let dims = x.dims();
        let seq_len = dims[dims.len() - 3];
        let head_dim = dims[dims.len() - 1];
        let half_dim = head_dim / 2;

        // Get cos/sin for the current positions
        let cos_slice = cos.narrow(0, position_offset, seq_len)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to slice cos: {}", e))
            ))?;

        let sin_slice = sin.narrow(0, position_offset, seq_len)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to slice sin: {}", e))
            ))?;

        // Split x into first and second half
        let x1 = x.narrow(dims.len() - 1, 0, half_dim)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to split x (first half): {}", e))
            ))?;

        let x2 = x.narrow(dims.len() - 1, half_dim, half_dim)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to split x (second half): {}", e))
            ))?;

        // Reshape cos/sin to broadcast properly
        let mut cos_shape = vec![1; dims.len()];
        cos_shape[dims.len() - 3] = seq_len;
        cos_shape[dims.len() - 1] = half_dim;

        let cos_reshaped = cos_slice.reshape(cos_shape.as_slice())
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to reshape cos: {}", e))
            ))?;

        let sin_reshaped = sin_slice.reshape(cos_shape.as_slice())
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to reshape sin: {}", e))
            ))?;

        // Apply rotation: x_out = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let x1_cos = x1.broadcast_mul(&cos_reshaped)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed x1 * cos: {}", e))
            ))?;

        let x2_sin = x2.broadcast_mul(&sin_reshaped)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed x2 * sin: {}", e))
            ))?;

        let x1_sin = x1.broadcast_mul(&sin_reshaped)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed x1 * sin: {}", e))
            ))?;

        let x2_cos = x2.broadcast_mul(&cos_reshaped)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed x2 * cos: {}", e))
            ))?;

        let out1 = x1_cos.sub(&x2_sin)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed out1 computation: {}", e))
            ))?;

        let out2 = x1_sin.add(&x2_cos)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed out2 computation: {}", e))
            ))?;

        // Concatenate the two halves
        CandleTensor::cat(&[out1, out2], dims.len() - 1)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to concatenate: {}", e))
            ))
    }

    /// cndl_reshape(x, shape) -> tensor
    /// Reshape tensor using Candle
    pub fn reshape(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_reshape() expects 2 arguments (x, shape), got {}", args.len())
            ));
        }

        let x_val = interpreter.eval_expr(&args[0])?;
        let shape_val = interpreter.eval_expr(&args[1])?;

        let shape: Vec<usize> = match shape_val {
            Value::TensorF32(ref t) => {
                let data = t.buffer().to_cpu_vec();
                data.iter().map(|&v| v as usize).collect()
            }
            Value::TensorF16(ref t) => {
                let data = t.buffer().to_cpu_vec();
                data.iter().map(|&v| v.to_f32() as usize).collect()
            }
            _ => return Err(RuntimeError::TypeError("shape must be a tensor".to_string())),
        };

        match x_val {
            Value::TensorF32(ref x) => {
                let x_candle = Self::tl_to_candle_f32(interpreter, x)?;
                let result = x_candle.reshape(shape.as_slice())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle reshape failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, result)?;
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            Value::TensorF16(ref x) => {
                let x_candle = Self::tl_to_candle_f16(interpreter, x)?;
                let result = x_candle.reshape(shape.as_slice())
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Candle reshape failed: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f16(interpreter, result)?;
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            _ => Err(RuntimeError::TypeError("cndl_reshape() requires a tensor".to_string()))
        }
    }

    // ============================================================================
    // Model loading and saving operations
    // ============================================================================

    /// cndl_load_safetensor(path, tensor_name) -> tensor
    /// Load a specific tensor from a Safetensors file using Candle
    pub fn load_safetensor(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_load_safetensor() expects 2 arguments (path, tensor_name), got {}", args.len())
            ));
        }

        let path_val = interpreter.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        let tensor_name_val = interpreter.eval_expr(&args[1])?;
        let tensor_name = match tensor_name_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("tensor_name must be a string".to_string())),
        };

        // Load the safetensors file
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let tensors = candle_core::safetensors::load(&path, &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to load safetensors file: {}", e))
            ))?;

        // Get the specific tensor
        let tensor = tensors.get(&tensor_name)
            .ok_or_else(|| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Tensor '{}' not found in file", tensor_name))
            ))?;

        // Convert to TensorLogic tensor based on dtype
        match tensor.dtype() {
            DType::F32 => {
                let result_tl = Self::candle_to_tl_f32(interpreter, tensor.clone())?;
                println!("Loaded tensor '{}' from {} (f32, shape: {:?})", tensor_name, path, tensor.dims());
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
            DType::F16 => {
                let result_tl = Self::candle_to_tl_f16(interpreter, tensor.clone())?;
                println!("Loaded tensor '{}' from {} (f16, shape: {:?})", tensor_name, path, tensor.dims());
                Ok(Value::TensorF16(Arc::new(result_tl)))
            }
            dtype => {
                // Try to convert to f32
                let tensor_f32 = tensor.to_dtype(DType::F32)
                    .map_err(|e| RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(format!("Failed to convert tensor to f32: {}", e))
                    ))?;
                let result_tl = Self::candle_to_tl_f32(interpreter, tensor_f32)?;
                println!("Loaded tensor '{}' from {} (converted from {:?} to f32, shape: {:?})",
                         tensor_name, path, dtype, tensor.dims());
                Ok(Value::TensorF32(Arc::new(result_tl)))
            }
        }
    }

    /// cndl_save_safetensor(tensor, path, tensor_name) -> void
    /// Save a tensor to a Safetensors file using Candle
    pub fn save_safetensor(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("cndl_save_safetensor() expects 3 arguments (tensor, path, tensor_name), got {}", args.len())
            ));
        }

        let tensor_val = interpreter.eval_expr(&args[0])?;

        let path_val = interpreter.eval_expr(&args[1])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        let tensor_name_val = interpreter.eval_expr(&args[2])?;
        let tensor_name = match tensor_name_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("tensor_name must be a string".to_string())),
        };

        // Convert TensorLogic tensor to Candle tensor
        let candle_tensor = match tensor_val {
            Value::TensorF32(ref t) => Self::tl_to_candle_f32(interpreter, t)?,
            Value::TensorF16(ref t) => Self::tl_to_candle_f16(interpreter, t)?,
            _ => return Err(RuntimeError::TypeError("First argument must be a tensor".to_string())),
        };

        // Create a HashMap with the tensor
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(tensor_name.clone(), candle_tensor);

        // Save to file
        candle_core::safetensors::save(&tensors, &path)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to save safetensors file: {}", e))
            ))?;

        println!("Saved tensor '{}' to {}", tensor_name, path);
        Ok(Value::Void)
    }

    /// cndl_list_safetensors(path) -> void
    /// List all tensor names in a Safetensors file
    pub fn list_safetensors(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_list_safetensors() expects 1 argument (path), got {}", args.len())
            ));
        }

        let path_val = interpreter.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        // Load the safetensors file
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let tensors = candle_core::safetensors::load(&path, &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to load safetensors file: {}", e))
            ))?;

        println!("Tensors in {}:", path);
        println!("  Total: {} tensors", tensors.len());
        println!("");

        let mut tensor_names: Vec<_> = tensors.keys().collect();
        tensor_names.sort();

        for name in tensor_names {
            let tensor = &tensors[name];
            println!("  - {} : {:?} {:?}", name, tensor.dtype(), tensor.dims());
        }

        Ok(Value::Void)
    }

    /// cndl_load_gguf_tensor(path, tensor_name) -> tensor
    /// Load a specific tensor from a GGUF file using Candle
    pub fn load_gguf_tensor(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_load_gguf_tensor() expects 2 arguments (path, tensor_name), got {}", args.len())
            ));
        }

        let path_val = interpreter.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        let tensor_name_val = interpreter.eval_expr(&args[1])?;
        let tensor_name = match tensor_name_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("tensor_name must be a string".to_string())),
        };

        // Load the GGUF file using Candle's VarBuilder
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        use candle_transformers::quantized_var_builder::VarBuilder;

        let vb = VarBuilder::from_gguf(&path, &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to load GGUF file: {}", e))
            ))?;

        // Try to get the tensor (returns Arc<QTensor>)
        let qtensor = vb.get_no_shape(&tensor_name)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to get tensor '{}': {}", tensor_name, e))
            ))?;

        // Dequantize the tensor to get a regular Tensor
        let tensor = qtensor.dequantize(&device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to dequantize tensor '{}': {}", tensor_name, e))
            ))?;

        // Convert to TensorLogic tensor
        // Check the dtype and convert accordingly
        let tensor_f32 = if tensor.dtype() == DType::F32 {
            tensor
        } else if tensor.dtype() == DType::F16 {
            // Try f16 first
            let result_tl = Self::candle_to_tl_f16(interpreter, tensor.clone())?;
            println!("Loaded tensor '{}' from {} (f16, shape: {:?})", tensor_name, path, tensor.dims());
            return Ok(Value::TensorF16(Arc::new(result_tl)));
        } else {
            tensor.to_dtype(DType::F32)
                .map_err(|e| RuntimeError::TensorError(
                    crate::error::TensorError::InvalidOperation(format!("Failed to convert tensor to f32: {}", e))
                ))?
        };

        let result_tl = Self::candle_to_tl_f32(interpreter, tensor_f32.clone())?;
        println!("Loaded tensor '{}' from {} (f32, shape: {:?})", tensor_name, path, tensor_f32.dims());
        Ok(Value::TensorF32(Arc::new(result_tl)))
    }

    /// cndl_list_gguf_tensors(path) -> void
    /// List all tensor names in a GGUF file
    pub fn list_gguf_tensors(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_list_gguf_tensors() expects 1 argument (path), got {}", args.len())
            ));
        }

        let path_val = interpreter.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        // Open GGUF file and read metadata
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(&path)
            .map_err(|e| RuntimeError::IoError(e))?;

        let reader = BufReader::new(file);

        // Use gguf-rs-lib to parse the file
        use gguf_rs_lib::prelude::*;

        let gguf = GGUFFileReader::new(reader)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to parse GGUF file: {}", e))
            ))?;

        println!("Tensors in {}:", path);
        println!("  GGUF version: {}", gguf.header().version);
        println!("  Total: {} tensors", gguf.tensor_count());
        println!("");

        let tensor_infos = gguf.tensor_infos();

        for info in tensor_infos {
            println!("  - {} : {:?} {:?}", info.name, info.tensor_type, info.shape);
        }

        Ok(Value::Void)
    }

    // ============================================================================
    // Full model save/load operations
    // ============================================================================

    /// cndl_save_model_safetensor(model, path) -> void
    /// Save an entire model (all tensors) to a Safetensors file using Candle
    pub fn save_model_safetensor(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("cndl_save_model_safetensor() expects 2 arguments (model, path), got {}", args.len())
            ));
        }

        let model_val = interpreter.eval_expr(&args[0])?;

        let path_val = interpreter.eval_expr(&args[1])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        // Convert all model tensors to Candle tensors
        let mut candle_tensors = std::collections::HashMap::new();

        match model_val {
            Value::ModelF16(ref model) => {
                println!("Saving model with {} tensors (f16)...", model.num_tensors());

                for (name, tensor) in &model.tensors {
                    let candle_tensor = Self::tl_to_candle_f16(interpreter, tensor)?;
                    candle_tensors.insert(name.clone(), candle_tensor);
                }
            }
            Value::ModelF32(ref model) => {
                println!("Saving model with {} tensors (f32)...", model.num_tensors());

                for (name, tensor) in &model.tensors {
                    let candle_tensor = Self::tl_to_candle_f32(interpreter, tensor)?;
                    candle_tensors.insert(name.clone(), candle_tensor);
                }
            }
            _ => {
                return Err(RuntimeError::TypeError(
                    "cndl_save_model_safetensor() requires a Model (ModelF16 or ModelF32)".to_string()
                ))
            }
        }

        // Save all tensors to file
        candle_core::safetensors::save(&candle_tensors, &path)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to save model: {}", e))
            ))?;

        println!("✓ Saved model to {} ({} tensors)", path, candle_tensors.len());
        Ok(Value::Void)
    }

    /// cndl_load_model_safetensor(path) -> model
    /// Load an entire model from a Safetensors file using Candle
    pub fn load_model_safetensor(interpreter: &mut Interpreter, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("cndl_load_model_safetensor() expects 1 argument (path), got {}", args.len())
            ));
        }

        let path_val = interpreter.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError("path must be a string".to_string())),
        };

        // Load the safetensors file
        let device = CandleDevice::new_metal(0)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to create Candle Metal device: {}", e))
            ))?;

        let candle_tensors = candle_core::safetensors::load(&path, &device)
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation(format!("Failed to load model: {}", e))
            ))?;

        println!("Loading model from {} ({} tensors)...", path, candle_tensors.len());

        // Determine dtype from first tensor
        if candle_tensors.is_empty() {
            return Err(RuntimeError::TensorError(
                crate::error::TensorError::InvalidOperation("Model file contains no tensors".to_string())
            ));
        }

        let first_dtype = candle_tensors.values().next().unwrap().dtype();

        // Convert all tensors based on first tensor's dtype
        match first_dtype {
            DType::F16 => {
                let mut tl_tensors = std::collections::HashMap::new();

                for (name, candle_tensor) in candle_tensors {
                    let tl_tensor = Self::candle_to_tl_f16(interpreter, candle_tensor)?;
                    tl_tensors.insert(name, Arc::new(tl_tensor));
                }

                // Create model with metadata
                use crate::model::{Model, ModelMetadata, ModelFormat};
                let model_name = std::path::Path::new(&path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                let metadata = ModelMetadata {
                    name: model_name,
                    format: ModelFormat::SafeTensors,
                    quantization: None,
                };
                let model = Model::from_tensors(tl_tensors, metadata);

                println!("✓ Loaded model (f16, {} tensors)", model.num_tensors());
                Ok(Value::ModelF16(model))
            }
            DType::F32 => {
                let mut tl_tensors = std::collections::HashMap::new();

                for (name, candle_tensor) in candle_tensors {
                    let tl_tensor = Self::candle_to_tl_f32(interpreter, candle_tensor)?;
                    tl_tensors.insert(name, Arc::new(tl_tensor));
                }

                // Create model with metadata
                use crate::model::{Model, ModelMetadata, ModelFormat};
                let model_name = std::path::Path::new(&path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                let metadata = ModelMetadata {
                    name: model_name,
                    format: ModelFormat::SafeTensors,
                    quantization: None,
                };
                let model = Model::from_tensors(tl_tensors, metadata);

                println!("✓ Loaded model (f32, {} tensors)", model.num_tensors());
                Ok(Value::ModelF32(model))
            }
            dtype => {
                // Convert to f32
                let mut tl_tensors = std::collections::HashMap::new();

                for (name, candle_tensor) in candle_tensors {
                    let tensor_f32 = candle_tensor.to_dtype(DType::F32)
                        .map_err(|e| RuntimeError::TensorError(
                            crate::error::TensorError::InvalidOperation(format!("Failed to convert tensor to f32: {}", e))
                        ))?;
                    let tl_tensor = Self::candle_to_tl_f32(interpreter, tensor_f32)?;
                    tl_tensors.insert(name, Arc::new(tl_tensor));
                }

                // Create model with metadata
                use crate::model::{Model, ModelMetadata, ModelFormat};
                let model_name = std::path::Path::new(&path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                let metadata = ModelMetadata {
                    name: model_name,
                    format: ModelFormat::SafeTensors,
                    quantization: None,
                };
                let model = Model::from_tensors(tl_tensors, metadata);

                println!("✓ Loaded model (converted from {:?} to f32, {} tensors)", dtype, model.num_tensors());
                Ok(Value::ModelF32(model))
            }
        }
    }
}
