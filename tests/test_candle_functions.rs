#![allow(unused_variables)]
/// Tests for Candle-based operations (cndl_* functions)
///
/// These tests verify that the Candle-based wrappers work correctly
/// and produce expected results for various mathematical operations.

use tensorlogic::prelude::*;
use tensorlogic::interpreter::Interpreter;
use tensorlogic::parser::TensorLogicParser;
use serial_test::serial;

/// Helper to run TensorLogic code and return result
#[allow(dead_code)]
fn run_tl_code(code: &str) -> Result<String, Box<dyn std::error::Error>> {
    let program = TensorLogicParser::parse_program(code)?;
    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)?;

    let result = interpreter.get_variable("result")?;
    Ok(format!("{:?}", result))
}

#[test]
#[serial]
fn test_cndl_matmul_f32() -> TensorResult<()> {
    let code = r#"
        main {
            a := f32::ones([2, 3])
            b := f32::ones([3, 4])
            result := cndl_matmul(a, b)
            print("cndl_matmul shape:", shape(result))
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    // Check result shape should be [2, 4]
    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[2, 4]);
            println!("✓ cndl_matmul f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_softmax_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::ones([2, 4])
            result := cndl_softmax(x, -1)
            print("cndl_softmax result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[2, 4]);

            // Check that softmax output sums to 1 along last dimension
            let data = t.buffer().to_cpu_vec();

            // Sum along last dimension (dim=1) for each row
            for i in 0..2 {
                let row_sum: f32 = (0..4).map(|j| data[i * 4 + j]).sum();
                assert!((row_sum - 1.0).abs() < 1e-3, "Softmax should sum to 1, got {}", row_sum);
            }

            println!("✓ cndl_softmax f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_gelu_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::from_array([0.0, 1.0, -1.0, 2.0])
            result := cndl_gelu(x)
            print("cndl_gelu result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[4]);

            let data = t.buffer().to_cpu_vec();

            // GELU(0) should be close to 0
            assert!(data[0].abs() < 0.01, "GELU(0) should be ~0, got {}", data[0]);

            // GELU(1) should be close to 0.841
            assert!((data[1] - 0.841).abs() < 0.1, "GELU(1) should be ~0.841, got {}", data[1]);

            println!("✓ cndl_gelu f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_silu_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::from_array([0.0, 1.0, -1.0, 2.0])
            result := cndl_silu(x)
            print("cndl_silu result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[4]);

            let data = t.buffer().to_cpu_vec();

            // SiLU(0) should be close to 0
            assert!(data[0].abs() < 0.01, "SiLU(0) should be ~0, got {}", data[0]);

            // SiLU(x) = x * sigmoid(x), so SiLU(1) ≈ 1 * 0.731 = 0.731
            assert!((data[1] - 0.731).abs() < 0.1, "SiLU(1) should be ~0.731, got {}", data[1]);

            println!("✓ cndl_silu f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_relu_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::from_array([-2.0, -1.0, 0.0, 1.0, 2.0])
            result := cndl_relu(x)
            print("cndl_relu result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[5]);

            let data = t.buffer().to_cpu_vec();
            let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

            for (i, (&result, &expected)) in data.iter().zip(expected.iter()).enumerate() {
                assert!((result - expected).abs() < 1e-5,
                    "ReLU mismatch at index {}: got {}, expected {}", i, result, expected);
            }

            println!("✓ cndl_relu f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_tanh_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::from_array([0.0, 1.0, -1.0])
            result := cndl_tanh(x)
            print("cndl_tanh result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[3]);

            let data = t.buffer().to_cpu_vec();

            // tanh(0) should be 0
            assert!(data[0].abs() < 1e-5, "tanh(0) should be 0, got {}", data[0]);

            // tanh(1) should be ~0.762
            assert!((data[1] - 0.762).abs() < 0.01, "tanh(1) should be ~0.762, got {}", data[1]);

            // tanh(-1) should be ~-0.762
            assert!((data[2] + 0.762).abs() < 0.01, "tanh(-1) should be ~-0.762, got {}", data[2]);

            println!("✓ cndl_tanh f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_transpose_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::ones([2, 3, 4])
            result := cndl_transpose(x, 0, 2)
            print("cndl_transpose shape:", shape(result))
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            // Shape should be [4, 3, 2] after transposing dims 0 and 2
            assert_eq!(t.dims(), &[4, 3, 2]);
            println!("✓ cndl_transpose f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
fn test_cndl_rms_norm_f32() -> TensorResult<()> {
    let code = r#"
        main {
            x := f32::ones([2, 4])
            result := cndl_rms_norm(x)
            print("cndl_rms_norm result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[2, 4]);

            let data = t.buffer().to_cpu_vec();

            // For ones tensor, RMS norm should give 1.0 for each element
            for (i, &val) in data.iter().enumerate() {
                assert!((val - 1.0).abs() < 0.01,
                    "RMS norm of ones should be ~1.0, got {} at index {}", val, i);
            }

            println!("✓ cndl_rms_norm f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
#[ignore] // This test requires proper implementation of embedding lookup
fn test_cndl_embedding_f32() -> TensorResult<()> {
    let code = r#"
        main {
            // Create embedding table: 5 words, 3 dims each
            embeddings := f32::from_array([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0],
                                          [10.0, 11.0, 12.0],
                                          [13.0, 14.0, 15.0]])
            // Get embedding for word 2
            result := cndl_embedding(2, embeddings)
            print("cndl_embedding result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[3]);

            let data = t.buffer().to_cpu_vec();
            let expected = vec![7.0, 8.0, 9.0];

            for (i, (&result, &expected)) in data.iter().zip(expected.iter()).enumerate() {
                assert!((result - expected).abs() < 1e-5,
                    "Embedding mismatch at index {}: got {}, expected {}", i, result, expected);
            }

            println!("✓ cndl_embedding f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

#[test]
#[serial]
#[ignore] // This test requires proper RoPE implementation with Candle
fn test_cndl_rope_f32() -> TensorResult<()> {
    let code = r#"
        main {
            // Create input: [seq_len=2, n_heads=1, head_dim=4]
            x := f32::ones([2, 1, 4])
            result := cndl_rope(x, 0, 10000.0)
            print("cndl_rope result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[2, 1, 4]);
            println!("✓ cndl_rope f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    Ok(())
}

// ============================================================================
// Model loading and saving tests
// ============================================================================

#[test]
#[serial]
fn test_cndl_save_load_safetensor_f32() -> TensorResult<()> {
    let code = r#"
        main {
            // Create a test tensor
            x := f32::from_array([1.0, 2.0, 3.0, 4.0])

            // Save it
            cndl_save_safetensor(x, "/tmp/test_candle_tensor.safetensors", "test_tensor")

            // Load it back
            result := cndl_load_safetensor("/tmp/test_candle_tensor.safetensors", "test_tensor")

            print("cndl_save_load_safetensor result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(t) => {
            assert_eq!(t.dims(), &[4]);

            let data = t.buffer().to_cpu_vec();
            let expected = vec![1.0, 2.0, 3.0, 4.0];

            for (i, (&result, &expected)) in data.iter().zip(expected.iter()).enumerate() {
                assert!((result - expected).abs() < 1e-5,
                    "Tensor value mismatch at index {}: got {}, expected {}", i, result, expected);
            }

            println!("✓ cndl_save_load_safetensor f32 test passed");
        }
        _ => panic!("Expected TensorF32"),
    }

    // Clean up
    let _ = std::fs::remove_file("/tmp/test_candle_tensor.safetensors");

    Ok(())
}

#[test]
#[serial]
fn test_cndl_save_load_safetensor_f16() -> TensorResult<()> {
    let code = r#"
        main {
            // Create a test tensor (f16)
            x := f16::from_array([1.0, 2.0, 3.0, 4.0])

            // Save it
            cndl_save_safetensor(x, "/tmp/test_candle_tensor_f16.safetensors", "test_tensor")

            // Load it back
            result := cndl_load_safetensor("/tmp/test_candle_tensor_f16.safetensors", "test_tensor")

            print("cndl_save_load_safetensor f16 result:", result)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF16(t) => {
            assert_eq!(t.dims(), &[4]);

            let data = t.buffer().to_cpu_vec();
            let expected = vec![1.0, 2.0, 3.0, 4.0];

            for (i, (&result, &expected)) in data.iter().zip(expected.iter()).enumerate() {
                assert!((result.to_f32() - expected).abs() < 1e-2,
                    "Tensor value mismatch at index {}: got {}, expected {}", i, result.to_f32(), expected);
            }

            println!("✓ cndl_save_load_safetensor f16 test passed");
        }
        _ => panic!("Expected TensorF16"),
    }

    // Clean up
    let _ = std::fs::remove_file("/tmp/test_candle_tensor_f16.safetensors");

    Ok(())
}

#[test]
#[serial]
fn test_cndl_list_safetensors() -> TensorResult<()> {
    // First create a test file with a tensor
    let code_save = r#"
        main {
            x := f32::from_array([1.0, 2.0, 3.0, 4.0])
            cndl_save_safetensor(x, "/tmp/test_list_safetensors.safetensors", "my_tensor")
        }
    "#;

    let program = TensorLogicParser::parse_program(code_save)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    // Now list the tensors
    let code_list = r#"
        main {
            cndl_list_safetensors("/tmp/test_list_safetensors.safetensors")
        }
    "#;

    let program = TensorLogicParser::parse_program(code_list)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    println!("✓ cndl_list_safetensors test passed");

    // Clean up
    let _ = std::fs::remove_file("/tmp/test_list_safetensors.safetensors");

    Ok(())
}

#[test]
#[serial]
#[ignore] // This test requires a GGUF file to exist
fn test_cndl_load_gguf_tensor() -> TensorResult<()> {
    use std::env;

    // This test assumes a GGUF model file exists
    let model_path = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
        + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";

    let code = format!(r#"
        main {{
            // Load a specific tensor from GGUF
            result := cndl_load_gguf_tensor("{}", "token_embd.weight")
            print("Loaded tensor shape:", shape(result))
        }}
    "#, model_path);

    let program = TensorLogicParser::parse_program(&code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    let result = interpreter.get_variable("result")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    match result {
        tensorlogic::interpreter::Value::TensorF32(_) | tensorlogic::interpreter::Value::TensorF16(_) => {
            println!("✓ cndl_load_gguf_tensor test passed");
        }
        _ => panic!("Expected tensor"),
    }

    Ok(())
}

#[test]
#[serial]
#[ignore] // This test requires a GGUF file to exist
fn test_cndl_list_gguf_tensors() -> TensorResult<()> {
    use std::env;

    let model_path = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
        + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";

    let code = format!(r#"
        main {{
            cndl_list_gguf_tensors("{}")
        }}
    "#, model_path);

    let program = TensorLogicParser::parse_program(&code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    println!("✓ cndl_list_gguf_tensors test passed");

    Ok(())
}

// ============================================================================
// Full model save/load tests
// ============================================================================

#[test]
#[serial]
fn test_cndl_save_load_model_safetensor_f32() -> TensorResult<()> {
    let code = r#"
        main {
            // Create a simple model with multiple tensors
            layer1_weight := f32::from_array([[1.0, 2.0], [3.0, 4.0]])
            layer1_bias := f32::from_array([0.5, 0.5])
            layer2_weight := f32::from_array([[5.0, 6.0], [7.0, 8.0]])

            // Save each tensor individually first
            cndl_save_safetensor(layer1_weight, "/tmp/test_model.safetensors", "layer1.weight")

            // Then create a full model
            model := load_model_f32("/tmp/test_model.safetensors")

            // Note: For now we test the basic save/load flow
            // Full model creation from scratch will be tested separately
            print("Model loaded:", model)
        }
    "#;

    let program = TensorLogicParser::parse_program(code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Execution error: {}", e)))?;

    println!("✓ cndl_save_load_model_safetensor f32 test passed");

    // Clean up
    let _ = std::fs::remove_file("/tmp/test_model.safetensors");

    Ok(())
}

#[test]
#[serial]
fn test_cndl_model_save_load_round_trip() -> TensorResult<()> {
    use tensorlogic::model::{Model, ModelMetadata, ModelFormat};
    use tensorlogic::device::MetalDevice;
    use tensorlogic::tensor::{Tensor, TensorCreation};
    use std::collections::HashMap;

    let device = MetalDevice::new()?;

    // Create a model with multiple tensors
    let mut tensors = HashMap::new();
    tensors.insert("layer1.weight".to_string(),
                   Tensor::<f32>::from_vec_gpu(&device, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?);
    tensors.insert("layer1.bias".to_string(),
                   Tensor::<f32>::from_vec_gpu(&device, vec![0.5, 0.5], vec![2])?);
    tensors.insert("layer2.weight".to_string(),
                   Tensor::<f32>::from_vec_gpu(&device, vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?);

    let metadata = ModelMetadata {
        name: "test_model".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    };
    let model = Model::from_tensors(tensors, metadata);

    // Test via TensorLogic interpreter
    let code = format!(r#"
        main {{
            // This will be tested when we can construct models in TL
            print("Model round-trip test")
        }}
    "#);

    let program = TensorLogicParser::parse_program(&code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();

    // Manually set the model in the environment for testing
    interpreter.set_variable("test_model".to_string(),
        tensorlogic::interpreter::Value::ModelF32(model));

    // Execute save
    let save_code = r#"
        main {
            cndl_save_model_safetensor(test_model, "/tmp/test_full_model.safetensors")
        }
    "#;

    let program = TensorLogicParser::parse_program(save_code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Save execution error: {}", e)))?;

    // Execute load
    let load_code = r#"
        main {
            loaded_model := cndl_load_model_safetensor("/tmp/test_full_model.safetensors")
            print("Loaded model:", loaded_model)
        }
    "#;

    let program = TensorLogicParser::parse_program(load_code)
        .map_err(|e| TensorError::InvalidOperation(format!("Parse error: {}", e)))?;

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| TensorError::InvalidOperation(format!("Load execution error: {}", e)))?;

    let loaded_model = interpreter.get_variable("loaded_model")
        .map_err(|e| TensorError::InvalidOperation(format!("Get variable error: {}", e)))?;

    // Verify it's a model
    match loaded_model {
        tensorlogic::interpreter::Value::ModelF32(model) => {
            assert_eq!(model.num_tensors(), 3);
            assert!(model.get_tensor("layer1.weight").is_some());
            assert!(model.get_tensor("layer1.bias").is_some());
            assert!(model.get_tensor("layer2.weight").is_some());

            // Verify tensor shapes
            let layer1_weight = model.get_tensor("layer1.weight").unwrap();
            assert_eq!(layer1_weight.dims(), &[2, 2]);

            println!("✓ cndl_model_save_load_round_trip test passed");
        }
        _ => panic!("Expected ModelF32"),
    }

    // Clean up
    let _ = std::fs::remove_file("/tmp/test_full_model.safetensors");

    Ok(())
}
