/// Comprehensive tests for Model Loading and Management
///
/// Model loading is critical for production ML systems.
/// Tests cover:
/// - Model creation and initialization
/// - Tensor insertion and retrieval
/// - Model metadata management
/// - Format detection (SafeTensors, GGUF, CoreML)
/// - Error handling (file not found, invalid format, etc.)
/// - Model structure validation
///
/// Note: Actual file loading tests require model files to exist.
/// These tests focus on the API and structure.

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};
use tensorlogic::model::{Model, ModelMetadata, ModelFormat, QuantizationType};
use std::collections::HashMap;
use half::f16;
use serial_test::serial;

// Model Creation and Basic Operations

#[test]
#[serial]
fn test_model_create_empty() -> TensorResult<()> {
    let metadata = ModelMetadata {
        name: "test_model".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    };

    let model = Model::new(metadata);

    assert_eq!(model.num_tensors(), 0);
    assert_eq!(model.tensor_names().len(), 0);
    assert_eq!(model.metadata.name, "test_model");
    assert!(matches!(model.metadata.format, ModelFormat::SafeTensors));

    println!("✓ Model create empty test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_insert_single_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let metadata = ModelMetadata {
        name: "test".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    };

    let mut model = Model::new(metadata);

    let tensor = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(1.0); 6],
        vec![2, 3]
    )?;

    model.insert_tensor("layer.0.weight".to_string(), tensor);

    assert_eq!(model.num_tensors(), 1);
    assert!(model.get_tensor("layer.0.weight").is_some());

    println!("✓ Model insert single tensor test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_insert_multiple_tensors() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let metadata = ModelMetadata {
        name: "multi_layer".to_string(),
        format: ModelFormat::GGUF,
        quantization: Some(QuantizationType::Q4_0),
    };

    let mut model = Model::new(metadata);

    // Insert multiple tensors
    for i in 0..5 {
        let tensor = Tensor::from_vec_gpu(
            &device,
            vec![f16::from_f32(i as f32); 10],
            vec![2, 5]
        )?;
        model.insert_tensor(format!("layer.{}.weight", i), tensor);
    }

    assert_eq!(model.num_tensors(), 5);

    // Verify all tensors exist
    for i in 0..5 {
        let name = format!("layer.{}.weight", i);
        assert!(
            model.get_tensor(&name).is_some(),
            "Tensor {} should exist",
            name
        );
    }

    println!("✓ Model insert multiple tensors test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_get_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "test".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    let tensor_data = vec![f16::from_f32(42.0); 4];
    let tensor = Tensor::from_vec_gpu(&device, tensor_data.clone(), vec![2, 2])?;

    model.insert_tensor("test_tensor".to_string(), tensor);

    // Get tensor
    let retrieved = model.get_tensor("test_tensor");
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.shape(), vec![2, 2]);

    let data = retrieved.sync_and_read();
    for &val in &data {
        assert_eq!(val, f16::from_f32(42.0));
    }

    println!("✓ Model get tensor test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_get_nonexistent_tensor() -> TensorResult<()> {
    let model = Model::new(ModelMetadata {
        name: "empty".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    let result = model.get_tensor("nonexistent");
    assert!(result.is_none(), "Getting nonexistent tensor should return None");

    println!("✓ Model get nonexistent tensor test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_tensor_names() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "named_model".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    let names = vec!["layer.0.weight", "layer.0.bias", "layer.1.weight", "layer.1.bias"];

    for name in &names {
        let tensor = Tensor::from_vec_gpu(&device, vec![f16::ONE; 4], vec![2, 2])?;
        model.insert_tensor(name.to_string(), tensor);
    }

    let retrieved_names = model.tensor_names();
    assert_eq!(retrieved_names.len(), names.len());

    // Check all names are present (order may vary due to HashMap)
    for name in &names {
        assert!(
            retrieved_names.iter().any(|n| n.as_str() == *name),
            "Name {} should be present",
            name
        );
    }

    println!("✓ Model tensor names test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_from_tensors() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut tensors = HashMap::new();

    tensors.insert(
        "weight".to_string(),
        Tensor::from_vec_gpu(&device, vec![f16::ONE; 4], vec![2, 2])?
    );
    tensors.insert(
        "bias".to_string(),
        Tensor::from_vec_gpu(&device, vec![f16::ZERO; 2], vec![2])?
    );

    let metadata = ModelMetadata {
        name: "from_tensors".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    };

    let model = Model::from_tensors(tensors, metadata);

    assert_eq!(model.num_tensors(), 2);
    assert!(model.get_tensor("weight").is_some());
    assert!(model.get_tensor("bias").is_some());

    println!("✓ Model from tensors test passed");
    Ok(())
}

// Metadata Tests

#[test]
#[serial]
fn test_model_metadata_safetensors() -> TensorResult<()> {
    let metadata = ModelMetadata {
        name: "safetensors_model".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    };

    let model = Model::new(metadata);

    assert_eq!(model.metadata.name, "safetensors_model");
    assert!(matches!(model.metadata.format, ModelFormat::SafeTensors));
    assert!(model.metadata.quantization.is_none());

    println!("✓ Model metadata SafeTensors test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_metadata_gguf_quantized() -> TensorResult<()> {
    let metadata = ModelMetadata {
        name: "gguf_model".to_string(),
        format: ModelFormat::GGUF,
        quantization: Some(QuantizationType::Q4_0),
    };

    let model = Model::new(metadata);

    assert_eq!(model.metadata.name, "gguf_model");
    assert!(matches!(model.metadata.format, ModelFormat::GGUF));
    assert!(matches!(
        model.metadata.quantization,
        Some(QuantizationType::Q4_0)
    ));

    println!("✓ Model metadata GGUF quantized test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_metadata_coreml() -> TensorResult<()> {
    let metadata = ModelMetadata {
        name: "coreml_model".to_string(),
        format: ModelFormat::CoreML,
        quantization: None,
    };

    let model = Model::new(metadata);

    assert_eq!(model.metadata.name, "coreml_model");
    assert!(matches!(model.metadata.format, ModelFormat::CoreML));

    println!("✓ Model metadata CoreML test passed");
    Ok(())
}

// Model Structure Tests

#[test]
#[serial]
fn test_model_typical_llm_structure() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "tiny_llm".to_string(),
        format: ModelFormat::GGUF,
        quantization: Some(QuantizationType::Q4_0),
    });

    // Simulate typical LLM structure
    let vocab_size = 1000;
    let d_model = 64;
    let num_layers = 2;

    // Token embedding
    let embed = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(0.01); vocab_size * d_model],
        vec![vocab_size, d_model]
    )?;
    model.insert_tensor("token_embd.weight".to_string(), embed);

    // Layer weights
    for layer in 0..num_layers {
        // Attention weights
        model.insert_tensor(
            format!("blk.{}.attn_q.weight", layer),
            Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * d_model], vec![d_model, d_model])?
        );
        model.insert_tensor(
            format!("blk.{}.attn_k.weight", layer),
            Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * d_model], vec![d_model, d_model])?
        );
        model.insert_tensor(
            format!("blk.{}.attn_v.weight", layer),
            Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * d_model], vec![d_model, d_model])?
        );

        // FFN weights
        model.insert_tensor(
            format!("blk.{}.ffn_up.weight", layer),
            Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * d_model * 4], vec![d_model, d_model * 4])?
        );
        model.insert_tensor(
            format!("blk.{}.ffn_down.weight", layer),
            Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * 4 * d_model], vec![d_model * 4, d_model])?
        );
    }

    // Output layer
    model.insert_tensor(
        "output.weight".to_string(),
        Tensor::from_vec_gpu(&device, vec![f16::ONE; d_model * vocab_size], vec![d_model, vocab_size])?
    );

    // Verify structure
    let expected_tensors = 1 + (num_layers * 5) + 1; // embed + layers + output
    assert_eq!(model.num_tensors(), expected_tensors);

    // Verify specific tensors exist
    assert!(model.get_tensor("token_embd.weight").is_some());
    assert!(model.get_tensor("blk.0.attn_q.weight").is_some());
    assert!(model.get_tensor("blk.1.ffn_down.weight").is_some());
    assert!(model.get_tensor("output.weight").is_some());

    println!("✓ Model typical LLM structure test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_update_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "updatable".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Insert initial tensor
    let tensor1 = Tensor::from_vec_gpu(&device, vec![f16::ONE; 4], vec![2, 2])?;
    model.insert_tensor("weight".to_string(), tensor1);

    // Update tensor (same name, new value)
    let tensor2 = Tensor::from_vec_gpu(&device, vec![f16::from_f32(2.0); 4], vec![2, 2])?;
    model.insert_tensor("weight".to_string(), tensor2);

    // Should still have 1 tensor
    assert_eq!(model.num_tensors(), 1);

    // Verify new value
    let retrieved = model.get_tensor("weight").unwrap();
    let data = retrieved.sync_and_read();
    for &val in &data {
        assert_eq!(val, f16::from_f32(2.0));
    }

    println!("✓ Model update tensor test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_large_number_of_tensors() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "large_model".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Insert 100 tensors
    let num_tensors = 100;
    for i in 0..num_tensors {
        let tensor = Tensor::from_vec_gpu(&device, vec![f16::from_f32(i as f32); 4], vec![2, 2])?;
        model.insert_tensor(format!("tensor_{}", i), tensor);
    }

    assert_eq!(model.num_tensors(), num_tensors);

    // Verify random tensors exist
    for i in [0, 25, 50, 75, 99] {
        let name = format!("tensor_{}", i);
        assert!(model.get_tensor(&name).is_some(), "Tensor {} should exist", name);
    }

    println!("✓ Model large number of tensors test passed");
    Ok(())
}

// Model Loading Error Cases

#[test]
#[serial]
#[should_panic(expected = "File has no extension")]
fn test_model_load_no_extension() {
    let device = MetalDevice::new().unwrap();
    let _ = Model::load("model_without_extension", &device).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "Unsupported file extension")]
fn test_model_load_unsupported_extension() {
    let device = MetalDevice::new().unwrap();
    let _ = Model::load("model.txt", &device).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "Unsupported file extension")]
fn test_model_load_unknown_format() {
    let device = MetalDevice::new().unwrap();
    let _ = Model::load("model.xyz", &device).unwrap();
}

#[test]
#[serial]
fn test_model_load_nonexistent_file() {
    let device = MetalDevice::new();
    if device.is_err() {
        println!("✓ Metal not available, skipping test");
        return;
    }
    let device = device.unwrap();

    // Try to load a file that doesn't exist
    let result = Model::load("nonexistent_model.safetensors", &device);

    // Should fail (either file not found or parse error)
    assert!(result.is_err(), "Loading nonexistent file should fail");

    println!("✓ Model load nonexistent file test passed");
}

// Tensor Shape Validation

#[test]
#[serial]
fn test_model_various_tensor_shapes() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "shapes".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Various shapes
    let shapes = vec![
        vec![10],               // 1D
        vec![5, 5],             // 2D square
        vec![3, 7],             // 2D rectangular
        vec![2, 3, 4],          // 3D
        vec![2, 2, 3, 4],       // 4D
        vec![1, 1, 1, 1, 10],   // 5D
    ];

    for (i, shape) in shapes.iter().enumerate() {
        let size: usize = shape.iter().product();
        let tensor = Tensor::from_vec_gpu(&device, vec![f16::ONE; size], shape.clone())?;
        model.insert_tensor(format!("tensor_{}", i), tensor);
    }

    assert_eq!(model.num_tensors(), shapes.len());

    // Verify shapes
    for (i, expected_shape) in shapes.iter().enumerate() {
        let tensor = model.get_tensor(&format!("tensor_{}", i)).unwrap();
        assert_eq!(tensor.shape(), *expected_shape);
    }

    println!("✓ Model various tensor shapes test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_empty_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "empty_tensors".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Empty tensor (0 elements)
    let tensor = Tensor::from_vec_gpu(&device, vec![], vec![0, 5])?;
    model.insert_tensor("empty".to_string(), tensor);

    assert_eq!(model.num_tensors(), 1);

    let retrieved = model.get_tensor("empty").unwrap();
    assert_eq!(retrieved.shape(), vec![0, 5]);

    println!("✓ Model empty tensor test passed");
    Ok(())
}

// Model Name Conventions

#[test]
#[serial]
fn test_model_nested_tensor_names() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "nested".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Nested naming conventions (common in PyTorch/HuggingFace)
    let names = vec![
        "model.encoder.layer.0.weight",
        "model.encoder.layer.0.bias",
        "model.encoder.layer.1.weight",
        "model.decoder.attention.qkv.weight",
        "model.decoder.mlp.fc1.weight",
        "output.layernorm.weight",
    ];

    for name in &names {
        let tensor = Tensor::from_vec_gpu(&device, vec![f16::ONE; 4], vec![2, 2])?;
        model.insert_tensor(name.to_string(), tensor);
    }

    assert_eq!(model.num_tensors(), names.len());

    // Verify all names
    for name in &names {
        assert!(model.get_tensor(name).is_some(), "Tensor {} should exist", name);
    }

    println!("✓ Model nested tensor names test passed");
    Ok(())
}

#[test]
#[serial]
fn test_model_special_characters_in_names() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "special_chars".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Names with special characters
    let names = vec![
        "layer-0",
        "layer_1",
        "layer.2",
        "layer/3",
        "layer:4",
    ];

    for name in &names {
        let tensor = Tensor::from_vec_gpu(&device, vec![f16::ONE; 2], vec![2])?;
        model.insert_tensor(name.to_string(), tensor);
    }

    assert_eq!(model.num_tensors(), names.len());

    for name in &names {
        assert!(model.get_tensor(name).is_some(), "Tensor {} should exist", name);
    }

    println!("✓ Model special characters in names test passed");
    Ok(())
}

// Quantization Tests

#[test]
#[serial]
fn test_model_quantization_types() -> TensorResult<()> {
    let quantization_types = vec![
        None,
        Some(QuantizationType::Q4_0),
        Some(QuantizationType::Q4_1),
        Some(QuantizationType::Q5_0),
        Some(QuantizationType::Q5_1),
        Some(QuantizationType::Q8_0),
    ];

    for quant in quantization_types {
        let metadata = ModelMetadata {
            name: format!("model_{:?}", quant),
            format: ModelFormat::GGUF,
            quantization: quant.clone(),
        };

        let model = Model::new(metadata);

        match quant {
            None => assert!(model.metadata.quantization.is_none()),
            Some(q) => assert!(matches!(model.metadata.quantization, Some(_))),
        }
    }

    println!("✓ Model quantization types test passed");
    Ok(())
}

// Model Cloning

#[test]
#[serial]
fn test_model_clone() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let mut model = Model::new(ModelMetadata {
        name: "original".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    let tensor = Tensor::from_vec_gpu(&device, vec![f16::from_f32(42.0); 4], vec![2, 2])?;
    model.insert_tensor("weight".to_string(), tensor);

    // Clone model
    let cloned = model.clone();

    assert_eq!(cloned.num_tensors(), model.num_tensors());
    assert_eq!(cloned.metadata.name, model.metadata.name);

    // Verify tensor was cloned
    let original_tensor = model.get_tensor("weight").unwrap();
    let cloned_tensor = cloned.get_tensor("weight").unwrap();

    assert_eq!(original_tensor.shape(), cloned_tensor.shape());

    println!("✓ Model clone test passed");
    Ok(())
}

// Integration Test: Simulate Model Loading Workflow

#[test]
#[serial]
fn test_model_workflow_simulation() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Step 1: Create model structure
    let mut model = Model::new(ModelMetadata {
        name: "workflow_test".to_string(),
        format: ModelFormat::SafeTensors,
        quantization: None,
    });

    // Step 2: Load tensors (simulated)
    let vocab_size = 100;
    let d_model = 32;

    let embedding = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(0.01); vocab_size * d_model],
        vec![vocab_size, d_model]
    )?;
    model.insert_tensor("embedding.weight".to_string(), embedding);

    // Step 3: Verify model is ready for inference
    assert_eq!(model.num_tensors(), 1);

    let emb = model.get_tensor("embedding.weight");
    assert!(emb.is_some());
    assert_eq!(emb.unwrap().shape(), vec![vocab_size, d_model]);

    // Step 4: Simulate tensor access during inference
    for _ in 0..10 {
        let _tensor = model.get_tensor("embedding.weight").unwrap();
        // Would perform inference operations here
    }

    println!("✓ Model workflow simulation test passed");
    Ok(())
}
