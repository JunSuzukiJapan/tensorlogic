//! Model and I/O operations for TensorLogic interpreter

use super::*;
use crate::device::Device;

impl Interpreter {
    pub(super) fn eval_model_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "save" => Some(self.eval_save(args)),
            "load" => Some(self.eval_load(args)),
            "load_model" => Some(self.eval_load_model(args)),
            "load_model_f32" => Some(self.eval_load_model_f32(args)),
            "get_tensor" => Some(self.eval_get_tensor(args)),

            "print" => Some(self.eval_print(args)),
            "load_tokenizer" => Some(self.eval_load_tokenizer(args)),
            "tokenize" => Some(self.eval_tokenize(args)),
            "detokenize" => Some(self.eval_detokenize(args)),
            "detokenize_single" => Some(self.eval_detokenize_single(args)),
            "int_to_tokenids" => Some(self.eval_int_to_tokenids(args)),
            "generate" | "print_top_k" => {
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
            Value::TensorF16(t) => t,
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
        Ok(Value::TensorF16(tensor))
    }

    /// load_model("path/to/model.gguf")
    /// Load a GGUF model file
    fn eval_load_model(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::model::Model;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("load_model() expects 1 argument (path), got {}", args.len())
            ));
        }

        // Evaluate path argument
        let path_val = self.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "load_model() argument must be a string (path)".to_string()
            )),
        };

        // Load model as f16 using Metal device
        let device = self.env.metal_device();
        let model = Model::<half::f16>::load(&path, device)
            .map_err(|e| RuntimeError::TensorError(e))?;

        println!("Loaded model from: {} (f16)", path);
        Ok(Value::ModelF16(model))
    }

    /// load_model_f32("path/to/model.gguf")
    /// Load a GGUF model as f32 (no f16 conversion)
    fn eval_load_model_f32(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        
        use crate::model::formats::GGUFLoader;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("load_model_f32() expects 1 argument (path), got {}", args.len())
            ));
        }

        // Evaluate path argument
        let path_val = self.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "load_model_f32() argument must be a string (path)".to_string()
            )),
        };

        // Load model as f32 using Metal device
        let device = self.env.metal_device();
        let model = GGUFLoader::load_f32(&path, device)
            .map_err(|e| RuntimeError::TensorError(e))?;

        println!("Loaded model from: {} (f32)", path);
        Ok(Value::ModelF32(model))
    }

    

    /// print(args...)
    /// Print values to stdout
    fn eval_print(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let mut output = String::new();

        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.push(' ');
            }

            let val = self.eval_expr(arg)?;
            match val {
                Value::String(s) => output.push_str(&s),
                Value::Integer(n) => output.push_str(&n.to_string()),
                Value::Float(f) => output.push_str(&f.to_string()),
                Value::Boolean(b) => output.push_str(&b.to_string()),
                Value::TensorF16(ref t) => {
                    output.push_str(&format!("Tensor(shape={:?})", t.dims()));
                }
                Value::TensorF32(ref t) => {
                    output.push_str(&format!("Tensor(shape={:?})", t.dims()));
                }
                Value::ModelF16(_) => output.push_str("Model<f16>(...)"),
                Value::ModelF32(_) => output.push_str("Model<f32>(...)"),
                Value::ModelLayerCollectionF16(ref c) => output.push_str(&format!("ModelLayerCollection<f16>(layers={})", c.layers.len())),
                Value::ModelLayerCollectionF32(ref c) => output.push_str(&format!("ModelLayerCollection<f32>(layers={})", c.layers.len())),
                Value::ModelLayerF16(ref l) => output.push_str(&format!("ModelLayer<f16>[{}]", l.index)),
                Value::ModelLayerF32(ref l) => output.push_str(&format!("ModelLayer<f32>[{}]", l.index)),
                Value::ModelFeatureF16(ref f) => output.push_str(&format!("ModelFeature<f16>({})", f.name)),
                Value::ModelFeatureF32(ref f) => output.push_str(&format!("ModelFeature<f32>({})", f.name)),
                Value::Tokenizer(_) => output.push_str("Tokenizer(...)"),
                Value::TokenIds(ref ids) => {
                    output.push_str(&format!("TokenIds(len={})", ids.len()));
                }
                Value::TokenIdArray(ref arr) => {
                    output.push_str(&format!("{}", arr.data().iter().map(|&id| id.to_string()).collect::<Vec<_>>().join(", ")));
                }
                Value::Type(ref ty) => output.push_str(&format!("Type({})", ty)),
                Value::Void => output.push_str("void"),
            }
        }

        println!("{}", output);
        Ok(Value::Void)
    }

    /// load_tokenizer("path/to/tokenizer.json")
    /// Load a HuggingFace tokenizer
    fn eval_load_tokenizer(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::tokenizer::Tokenizer;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("load_tokenizer() expects 1 argument (path), got {}", args.len())
            ));
        }

        // Evaluate path argument
        let path_val = self.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "load_tokenizer() argument must be a string (path)".to_string()
            )),
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| RuntimeError::TensorError(e))?;

        println!("Loaded tokenizer from: {}", path);
        Ok(Value::Tokenizer(std::sync::Arc::new(tokenizer)))
    }

    /// tokenize(tokenizer, "text", add_special_tokens)
    /// Convert text to token IDs
    fn eval_tokenize(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("tokenize() expects 3 arguments (tokenizer, text, add_special_tokens), got {}", args.len())
            ));
        }

        // Get tokenizer
        let tokenizer_val = self.eval_expr(&args[0])?;
        let tokenizer = match tokenizer_val {
            Value::Tokenizer(t) => t,
            _ => return Err(RuntimeError::TypeError(
                "tokenize() first argument must be a Tokenizer".to_string()
            )),
        };

        // Get text
        let text_val = self.eval_expr(&args[1])?;
        let text = match text_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "tokenize() second argument must be a string (text)".to_string()
            )),
        };

        // Get add_special_tokens flag
        let special_val = self.eval_expr(&args[2])?;
        let add_special = match special_val {
            Value::Boolean(b) => b,
            _ => return Err(RuntimeError::TypeError(
                "tokenize() third argument must be a boolean".to_string()
            )),
        };

        // Tokenize
        let token_ids = tokenizer.encode(&text, add_special)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::TokenIds(token_ids))
    }

    /// detokenize(tokenizer, token_ids, skip_special_tokens)
    /// Convert token IDs to text
    fn eval_detokenize(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("detokenize() expects 3 arguments (tokenizer, token_ids, skip_special_tokens), got {}", args.len())
            ));
        }

        // Get tokenizer
        let tokenizer_val = self.eval_expr(&args[0])?;
        let tokenizer = match tokenizer_val {
            Value::Tokenizer(t) => t,
            _ => return Err(RuntimeError::TypeError(
                "detokenize() first argument must be a Tokenizer".to_string()
            )),
        };

        // Get token IDs
        let ids_val = self.eval_expr(&args[1])?;
        let token_ids = match ids_val {
            Value::TokenIds(ids) => ids,
            _ => return Err(RuntimeError::TypeError(
                "detokenize() second argument must be TokenIds".to_string()
            )),
        };

        // Get skip_special_tokens flag
        let skip_val = self.eval_expr(&args[2])?;
        let skip_special = match skip_val {
            Value::Boolean(b) => b,
            _ => return Err(RuntimeError::TypeError(
                "detokenize() third argument must be a boolean".to_string()
            )),
        };

        // Detokenize
        let text = tokenizer.decode(&token_ids, skip_special)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::String(text))
    }

    /// detokenize_single(tokenizer, token_id, skip_special_tokens)
    /// Convert a single token ID (Integer) to text
    fn eval_detokenize_single(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("detokenize_single() expects 3 arguments (tokenizer, token_id, skip_special_tokens), got {}", args.len())
            ));
        }

        // Get tokenizer
        let tokenizer_val = self.eval_expr(&args[0])?;
        let tokenizer = match tokenizer_val {
            Value::Tokenizer(t) => t,
            _ => return Err(RuntimeError::TypeError(
                "detokenize_single() first argument must be a Tokenizer".to_string()
            )),
        };

        // Get token ID (Integer)
        let token_id_val = self.eval_expr(&args[1])?;
        let token_id = match token_id_val {
            Value::Integer(id) => id as u32,
            _ => return Err(RuntimeError::TypeError(
                "detokenize_single() second argument must be an Integer (token ID)".to_string()
            )),
        };

        // Get skip_special_tokens flag
        let skip_val = self.eval_expr(&args[2])?;
        let skip_special = match skip_val {
            Value::Boolean(b) => b,
            _ => return Err(RuntimeError::TypeError(
                "detokenize_single() third argument must be a boolean".to_string()
            )),
        };

        // Convert single token to TokenIds array
        let token_ids = vec![token_id];

        // Detokenize
        let text = tokenizer.decode(&token_ids, skip_special)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::String(text))
    }

    /// int_to_tokenids(token_id)
    /// Convert a single token ID (Integer) to TokenIds array
    fn eval_int_to_tokenids(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("int_to_tokenids() expects 1 argument (token_id), got {}", args.len())
            ));
        }

        // Get token ID (Integer)
        let token_id_val = self.eval_expr(&args[0])?;
        let token_id = match token_id_val {
            Value::Integer(id) => id as u32,
            _ => return Err(RuntimeError::TypeError(
                "int_to_tokenids() argument must be an Integer (token ID)".to_string()
            )),
        };

        // Create single-element TokenIds vector
        let token_ids = vec![token_id];

        Ok(Value::TokenIds(token_ids))
    }

    /// get_tensor(model, "tensor_name")
    /// Get a tensor from a model by name
    fn eval_get_tensor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("get_tensor() expects 2 arguments (model, tensor_name), got {}", args.len())
            ));
        }

        // Evaluate model argument
        let model_val = self.eval_expr(&args[0])?;

        // Evaluate tensor name argument
        let name_val = self.eval_expr(&args[1])?;
        let tensor_name = match name_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "get_tensor() second argument must be a string (tensor name)".to_string()
            )),
        };

        // Get tensor from model (f16 or f32)
        match model_val {
            Value::ModelF16(model) => {
                let tensor = model.get_tensor(&tensor_name)
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Tensor '{}' not found in model", tensor_name)
                    ))?;
                Ok(Value::TensorF16(tensor.clone()))
            }
            Value::ModelF32(model) => {
                let tensor = model.get_tensor(&tensor_name)
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Tensor '{}' not found in model", tensor_name)
                    ))?;
                Ok(Value::TensorF32(tensor.clone()))
            }
            _ => Err(RuntimeError::TypeError(
                "get_tensor() first argument must be a Model".to_string()
            )),
        }
    }
}
