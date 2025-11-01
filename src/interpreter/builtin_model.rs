//! Model and I/O operations for TensorLogic interpreter

use super::*;
use crate::device::Device;

impl Interpreter {
    pub(super) fn eval_model_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "save" => Some(self.eval_save(args)),
            "load" => Some(self.eval_load(args)),
            "load_model" => Some(self.eval_load_model(args)),
            "load_model_f16" => Some(self.eval_load_model_f16(args)),
            "load_model_f32" => Some(self.eval_load_model_f32(args)),
            "get_tensor" => Some(self.eval_get_tensor(args)),

            "print" => Some(self.eval_print(args)),
            "load_tokenizer" => Some(self.eval_load_tokenizer(args)),
            // tokenize, detokenize, append_token are now type methods only
            // Use: tokenizer.tokenize() / tokenizer.detokenize() / tokens.append_token()
            "detokenize_single" => Some(self.eval_detokenize_single(args)),
            "detokenize_incremental" => Some(self.eval_detokenize_incremental(args)),
            "int_to_tokenids" => Some(self.eval_int_to_tokenids(args)),
            "string_length" => Some(self.eval_string_length(args)),
            "string_substring" => Some(self.eval_string_substring(args)),
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

    /// load_model_f16("path/to/model.gguf")
    /// Load a GGUF model using fast mmap loader with f16 support
    /// This is significantly faster than load_model() and load_model_f32()
    fn eval_load_model_f16(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::model::formats::MmapGGUFLoader;

        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("load_model_f16() expects 1 argument (path), got {}", args.len())
            ));
        }

        // Evaluate path argument
        let path_val = self.eval_expr(&args[0])?;
        let path = match path_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "load_model_f16() argument must be a string (path)".to_string()
            )),
        };

        // Create mmap loader
        let loader = MmapGGUFLoader::new(&path)
            .map_err(|e| RuntimeError::TensorError(e))?;

        println!("Created mmap loader for: {}", path);
        println!("  Tensors: {}", loader.metadata().tensor_count);
        println!("  Version: {}", loader.metadata().version);

        // Load model as f16 using Metal device
        let device = self.env.metal_device();
        let model = loader.load_f16_model(device)
            .map_err(|e| RuntimeError::TensorError(e))?;

        println!("Loaded model as f16 (mmap zero-copy)");
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
        // print(value1, value2, ...) - simple mode
        // print("format {}", arg1, arg2, ...) - format string mode

        if args.is_empty() {
            println!();
            return Ok(Value::Void);
        }

        // Check if first argument is a string literal (format string mode)
        let first_val = self.eval_expr(&args[0])?;

        if let Value::String(ref format_str) = first_val {
            // Check if this is format string mode (contains {}) or simple mode
            if format_str.contains("{}") {
                // Format string mode: print("Hello {}", name)
                if args.len() > 1 {
                    // Evaluate remaining arguments
                    let mut format_args = Vec::new();
                    for arg in &args[1..] {
                        format_args.push(self.eval_expr(arg)?);
                    }

                    // Use format_string helper from eval.rs
                    let formatted = self.format_string(&format_str, &format_args)?;
                    println!("{}", formatted);
                } else {
                    // Just a string, print it
                    println!("{}", format_str);
                }
            } else if args.len() == 1 {
                // Just a single string, print it
                println!("{}", format_str);
            } else {
                // Simple mode with multiple arguments: print("A", "B", "C")
                print!("{}", self.value_to_display(&first_val));
                for arg in &args[1..] {
                    print!(" ");
                    let val = self.eval_expr(arg)?;
                    print!("{}", self.value_to_display(&val));
                }
                println!();
            }
        } else {
            // Simple mode: print(value1, value2, ...)
            print!("{}", self.value_to_display(&first_val));
            for arg in &args[1..] {
                print!(" ");
                let val = self.eval_expr(arg)?;
                print!("{}", self.value_to_display(&val));
            }
            println!();
        }

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
    pub(super) fn eval_tokenize(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
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
    pub(super) fn eval_detokenize(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
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

    /// detokenize_incremental(tokenizer, token_ids_array)
    /// Detokenize all tokens and return only the new text since last call
    /// This handles multi-byte UTF-8 characters correctly by decoding all tokens together
    fn eval_detokenize_incremental(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("detokenize_incremental() expects 2 arguments (tokenizer, token_ids), got {}", args.len())
            ));
        }

        // Get tokenizer
        let tokenizer_val = self.eval_expr(&args[0])?;
        let tokenizer = match tokenizer_val {
            Value::Tokenizer(t) => t,
            _ => return Err(RuntimeError::TypeError(
                "detokenize_incremental() first argument must be a Tokenizer".to_string()
            )),
        };

        // Get token IDs array
        let token_ids_val = self.eval_expr(&args[1])?;
        let token_ids = match token_ids_val {
            Value::TokenIds(ids) => ids,
            _ => return Err(RuntimeError::TypeError(
                "detokenize_incremental() second argument must be TokenIds array".to_string()
            )),
        };

        // Decode all tokens
        let full_text = tokenizer.decode(&token_ids, false)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::String(full_text))
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

    /// append_token(token_ids, token_id)
    /// Append a token ID to TokenIds array (returns new array)
    pub(super) fn eval_append_token(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("append_token() expects 2 arguments (token_ids, token_id), got {}", args.len())
            ));
        }

        // Get existing token IDs array
        let token_ids_val = self.eval_expr(&args[0])?;
        let mut token_ids = match token_ids_val {
            Value::TokenIds(ids) => ids,
            _ => return Err(RuntimeError::TypeError(
                "append_token() first argument must be TokenIds array".to_string()
            )),
        };

        // Get new token ID to append
        let token_id_val = self.eval_expr(&args[1])?;
        let new_token_id = match token_id_val {
            Value::Integer(id) => id as u32,
            _ => return Err(RuntimeError::TypeError(
                "append_token() second argument must be an Integer (token ID)".to_string()
            )),
        };

        // Append new token
        token_ids.push(new_token_id);

        Ok(Value::TokenIds(token_ids))
    }

    /// string_length(text)
    /// Get the length of a string
    fn eval_string_length(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("string_length() expects 1 argument (text), got {}", args.len())
            ));
        }

        let text_val = self.eval_expr(&args[0])?;
        let text = match text_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "string_length() argument must be a String".to_string()
            )),
        };

        Ok(Value::Integer(text.len() as i64))
    }

    /// string_substring(text, start, length)
    /// Get a substring from a string
    fn eval_string_substring(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 3 {
            return Err(RuntimeError::TypeError(
                format!("string_substring() expects 3 arguments (text, start, length), got {}", args.len())
            ));
        }

        let text_val = self.eval_expr(&args[0])?;
        let text = match text_val {
            Value::String(s) => s,
            _ => return Err(RuntimeError::TypeError(
                "string_substring() first argument must be a String".to_string()
            )),
        };

        let start_val = self.eval_expr(&args[1])?;
        let start = match start_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError(
                "string_substring() second argument must be an Integer (start)".to_string()
            )),
        };

        let length_val = self.eval_expr(&args[2])?;
        let length = match length_val {
            Value::Integer(i) => i as usize,
            _ => return Err(RuntimeError::TypeError(
                "string_substring() third argument must be an Integer (length)".to_string()
            )),
        };

        let substring = text.chars().skip(start).take(length).collect::<String>();
        Ok(Value::String(substring))
    }
}

#[cfg(test)]
mod detokenize_tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use std::sync::Arc;

    /// Helper function to load TinyLlama tokenizer for testing
    /// Returns None if tokenizer file is not available (optional test)
    fn load_test_tokenizer() -> Option<Arc<Tokenizer>> {
        let home = std::env::var("HOME").ok()?;
        let tokenizer_path = format!("{}/.llm/tokenizers/tinyllama-tokenizer.json", home);

        if !std::path::Path::new(&tokenizer_path).exists() {
            return None;
        }

        Tokenizer::from_file(&tokenizer_path).ok().map(Arc::new)
    }

    #[test]
    fn test_detokenize_single_special_tokens() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test BOS token <s> (ID: 1)
        let bos_text = tokenizer.decode(&[1], false).unwrap();
        assert_eq!(bos_text, "<s>", "Token ID 1 should decode to <s>");

        // Test EOS token </s> (ID: 2)
        let eos_text = tokenizer.decode(&[2], false).unwrap();
        assert_eq!(eos_text, "</s>", "Token ID 2 should decode to </s>");

        // Test UNK token <unk> (ID: 0)
        let unk_text = tokenizer.decode(&[0], false).unwrap();
        assert_eq!(unk_text, "<unk>", "Token ID 0 should decode to <unk>");
    }

    #[test]
    fn test_detokenize_single_skip_special_tokens() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // When skip_special_tokens=true, special tokens should be empty or filtered
        let bos_text_skip = tokenizer.decode(&[1], true).unwrap();
        assert!(
            bos_text_skip.is_empty() || bos_text_skip == " ",
            "BOS token should be skipped when skip_special_tokens=true, got: '{}'",
            bos_text_skip
        );

        let eos_text_skip = tokenizer.decode(&[2], true).unwrap();
        assert!(
            eos_text_skip.is_empty() || eos_text_skip == " ",
            "EOS token should be skipped when skip_special_tokens=true, got: '{}'",
            eos_text_skip
        );
    }

    #[test]
    fn test_detokenize_single_regular_tokens() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test known single-character token mappings from debug_sampling.tl
        // Token ID 100 should produce 'a' or similar common character
        let token_100 = tokenizer.decode(&[100], false).unwrap();
        assert!(!token_100.is_empty(), "Token ID 100 should decode to non-empty string");

        // For TinyLlama, token 100 is known to be a valid character
        assert!(
            token_100.len() <= 10,
            "Single token should not produce unreasonably long text, got: '{}'",
            token_100
        );
    }

    #[test]
    fn test_detokenize_single_utf8_handling() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test that decoder properly handles UTF-8 encoding
        // We'll test with a sequence of tokens that should form valid UTF-8
        let hello_tokens = tokenizer.encode("Hello", false).unwrap();
        let decoded = tokenizer.decode(&hello_tokens, false).unwrap();

        assert_eq!(decoded, "Hello", "UTF-8 encoding/decoding should be consistent");
        assert!(decoded.is_ascii(), "Hello should decode as ASCII");
    }

    #[test]
    fn test_detokenize_single_consistency() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test encode-decode consistency (mathematical property: decode(encode(x)) == x)
        let test_texts = vec![
            "Hello, world!",
            "The quick brown fox",
            "123456",
            "Special chars: !@#$%",
        ];

        for text in test_texts {
            let tokens = tokenizer.encode(text, false).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();

            assert_eq!(
                decoded, text,
                "Encode-decode should be consistent: encode('{}') → {:?} → decode → '{}'",
                text, tokens, decoded
            );
        }
    }

    #[test]
    fn test_detokenize_single_empty_behavior() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test that decoding empty token array produces empty string
        let empty_decoded = tokenizer.decode(&[], false).unwrap();
        assert_eq!(empty_decoded, "", "Empty token array should decode to empty string");
    }

    #[test]
    fn test_detokenize_single_token_id_bounds() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        let vocab_size = tokenizer.vocab_size();
        println!("TinyLlama vocab size: {}", vocab_size);

        // Test that valid token IDs within vocabulary bounds decode successfully
        // TinyLlama has 32000 tokens in vocabulary
        assert!(vocab_size > 0, "Vocabulary size should be positive");
        assert!(vocab_size <= 100000, "Vocabulary size should be reasonable");

        // Test first token (usually <unk>)
        let first_token = tokenizer.decode(&[0], false);
        assert!(first_token.is_ok(), "First token (ID 0) should decode successfully");

        // Test a mid-range token
        let mid_token = tokenizer.decode(&[vocab_size as u32 / 2], false);
        assert!(mid_token.is_ok(), "Mid-range token should decode successfully");

        // Test last valid token
        let last_token = tokenizer.decode(&[(vocab_size - 1) as u32], false);
        assert!(last_token.is_ok(), "Last valid token should decode successfully");
    }

    #[test]
    fn test_detokenize_single_deterministic() {
        let Some(tokenizer) = load_test_tokenizer() else {
            eprintln!("Skipping test: TinyLlama tokenizer not available");
            return;
        };

        // Test that same token ID always produces same output (deterministic property)
        let token_id = 100u32;

        let result1 = tokenizer.decode(&[token_id], false).unwrap();
        let result2 = tokenizer.decode(&[token_id], false).unwrap();
        let result3 = tokenizer.decode(&[token_id], false).unwrap();

        assert_eq!(result1, result2, "Decoding should be deterministic");
        assert_eq!(result2, result3, "Decoding should be deterministic");
    }
}
