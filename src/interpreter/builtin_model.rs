//! Model and I/O operations for TensorLogic interpreter

use super::*;
use crate::device::Device;

impl Interpreter {
    pub(super) fn eval_model_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "save" => Some(self.eval_save(args)),
            "load" => Some(self.eval_load(args)),
            "load_model" | "load_tokenizer" | "get_tensor" |
            "tokenize" | "detokenize" | "generate" |
            "print" | "print_top_k" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Model/IO function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// save(tensor, "filename")
    /// Save tensor to file
    fn eval_save(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("save() expects 2 arguments (tensor, filename), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = match tensor_val {
            Value::Tensor(t) => t,
            _ => return Err(RuntimeError::TypeError(
                "save() first argument must be a tensor".to_string()
            )),
        };

        // Evaluate filename argument
        let filename_val = self.eval_expr(&args[1])?;
        let filename = match filename_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "save() second argument must be a string (filename)".to_string()
            )),
        };

        // Save tensor to file
        tensor.save(&filename).map_err(|e| RuntimeError::TensorError(e))?;

        println!("Saved tensor to: {}", filename);
        Ok(Value::Void)
    }

    /// load("filename")
    /// Load tensor from file
    fn eval_load(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("load() expects 1 argument (filename), got {}", args.len())
            ));
        }

        // Evaluate filename argument
        let filename_val = self.eval_expr(&args[0])?;
        let filename = match filename_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "load() argument must be a string (filename)".to_string()
            )),
        };

        // Load tensor from file using existing Metal device
        let device = Device::Metal(self.env.metal_device().clone());
        let tensor = Tensor::load(&device, &filename).map_err(|e| RuntimeError::TensorError(e))?;

        println!("Loaded tensor from: {} (shape: {:?})", filename, tensor.dims());
        Ok(Value::Tensor(tensor))
    }
}
