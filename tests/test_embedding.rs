/// Comprehensive tests for Embedding (Token Lookup) operations
///
/// Embedding is fundamental for NLP tasks, especially language models.
/// Tests cover:
/// - Basic embedding lookup
/// - 1D and 2D token_ids (single sequence and batched)
/// - Different vocab sizes and embedding dimensions
/// - TokenIdArray precision (no f16 loss)
/// - Boundary conditions and error cases
/// - Numerical correctness

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors, TokenIdArray};
use half::f16;
use serial_test::serial;

// Helper function to assert tensors are close
fn assert_tensor_close_f32(result: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(result.len(), expected.len(), "Length mismatch");
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r, e, (r - e).abs()
        );
    }
}

fn assert_tensor_close_f16(result: &[f16], expected: &[f16], epsilon: f32) {
    assert_eq!(result.len(), expected.len(), "Length mismatch");
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r.to_f32() - e.to_f32()).abs();
        assert!(
            diff < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r.to_f32(), e.to_f32(), diff
        );
    }
}

#[test]
#[serial]
fn test_embedding_basic() -> TensorResult<()> {
    // Test basic embedding lookup with small vocabulary
    let vocab_size = 5;
    let d_model = 3;

    // Create embedding weight matrix [vocab_size, d_model]
    let weight = Tensor::<f32>::from_vec(
        vec![
            1.0, 2.0, 3.0,    // token 0
            4.0, 5.0, 6.0,    // token 1
            7.0, 8.0, 9.0,    // token 2
            10.0, 11.0, 12.0, // token 3
            13.0, 14.0, 15.0, // token 4
        ],
        vec![vocab_size, d_model]
    )?;

    // Create token IDs [2]
    let token_ids = Tensor::<f32>::from_vec(vec![2.0], vec![1])?;

    // Lookup embeddings
    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Should retrieve row 2: [7.0, 8.0, 9.0]
    let expected = vec![7.0, 8.0, 9.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    // Check output shape: [1, d_model]
    assert_eq!(embeddings.shape(), vec![1, d_model]);

    println!("✓ Embedding basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_multiple_tokens() -> TensorResult<()> {
    // Test embedding lookup for multiple tokens (sequence)
    let vocab_size = 4;
    let d_model = 2;

    let weight = Tensor::<f32>::from_vec(
        vec![
            1.0, 2.0,  // token 0
            3.0, 4.0,  // token 1
            5.0, 6.0,  // token 2
            7.0, 8.0,  // token 3
        ],
        vec![vocab_size, d_model]
    )?;

    // Token sequence: [0, 2, 1, 3]
    let token_ids = Tensor::<f32>::from_vec(vec![0.0, 2.0, 1.0, 3.0], vec![4])?;

    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Expected: [[1,2], [5,6], [3,4], [7,8]] flattened
    let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    // Check shape: [4, 2]
    assert_eq!(embeddings.shape(), vec![4, d_model]);

    println!("✓ Embedding multiple tokens test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_batch() -> TensorResult<()> {
    // Test batched embedding lookup [batch_size, seq_len]
    let vocab_size = 3;
    let d_model = 2;

    let weight = Tensor::<f32>::from_vec(
        vec![
            1.0, 2.0,  // token 0
            3.0, 4.0,  // token 1
            5.0, 6.0,  // token 2
        ],
        vec![vocab_size, d_model]
    )?;

    // Batch of 2 sequences, each with 3 tokens
    // [[0, 1, 2], [2, 1, 0]]
    let token_ids = Tensor::<f32>::from_vec(
        vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
        vec![2, 3]
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Expected:
    // Batch 0: [[1,2], [3,4], [5,6]]
    // Batch 1: [[5,6], [3,4], [1,2]]
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        5.0, 6.0, 3.0, 4.0, 1.0, 2.0,
    ];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    // Check shape: [2, 3, 2]
    assert_eq!(embeddings.shape(), vec![2, 3, d_model]);

    println!("✓ Embedding batch test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_f16() -> TensorResult<()> {
    // Test embedding with f16 precision
    let vocab_size = 4;
    let d_model = 3;

    let weight = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
            f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0),
            f16::from_f32(7.0), f16::from_f32(8.0), f16::from_f32(9.0),
            f16::from_f32(10.0), f16::from_f32(11.0), f16::from_f32(12.0),
        ],
        vec![vocab_size, d_model]
    )?;

    let token_ids = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(3.0)],
        vec![2]
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Expected: rows 1 and 3
    let expected = vec![
        f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0),
        f16::from_f32(10.0), f16::from_f32(11.0), f16::from_f32(12.0),
    ];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    assert_eq!(embeddings.shape(), vec![2, d_model]);

    println!("✓ Embedding f16 test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_from_token_ids() -> TensorResult<()> {
    // Test embedding_from_token_ids (no f16 precision loss)
    let vocab_size = 5;
    let d_model = 4;

    let weight = Tensor::<f32>::from_vec(
        (0..vocab_size * d_model).map(|i| i as f32).collect(),
        vec![vocab_size, d_model]
    )?;

    // Create TokenIdArray
    let token_ids = TokenIdArray::new(vec![0, 2, 4]);

    let embeddings = weight.embedding_from_token_ids(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Expected: rows 0, 2, 4
    let expected = vec![
        0.0, 1.0, 2.0, 3.0,      // row 0
        8.0, 9.0, 10.0, 11.0,    // row 2
        16.0, 17.0, 18.0, 19.0,  // row 4
    ];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    assert_eq!(embeddings.shape(), vec![3, d_model]);

    println!("✓ Embedding from TokenIdArray test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_large_vocabulary() -> TensorResult<()> {
    // Test with realistic vocabulary size
    let vocab_size = 1000;
    let d_model = 64;

    // Create embedding matrix with sequential values for easy verification
    let weight_data: Vec<f32> = (0..vocab_size * d_model)
        .map(|i| (i % 100) as f32)
        .collect();
    let weight = Tensor::<f32>::from_vec(weight_data, vec![vocab_size, d_model])?;

    // Lookup tokens
    let token_ids = Tensor::<f32>::from_vec(vec![0.0, 100.0, 500.0, 999.0], vec![4])?;

    let embeddings = weight.embedding(&token_ids)?;

    // Check shape
    assert_eq!(embeddings.shape(), vec![4, d_model]);

    // Verify first embedding (token 0)
    let result = embeddings.sync_and_read();
    for i in 0..d_model {
        assert!((result[i] - (i % 100) as f32).abs() < 1e-5);
    }

    println!("✓ Embedding large vocabulary test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_identity() -> TensorResult<()> {
    // Test with identity-like embedding (each token maps to one-hot-like vector)
    let vocab_size = 4;
    let d_model = 4;

    // Identity matrix
    let weight = Tensor::<f32>::from_vec(
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        vec![vocab_size, d_model]
    )?;

    let token_ids = Tensor::<f32>::from_vec(vec![2.0], vec![1])?;
    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Should be [0, 0, 1, 0]
    let expected = vec![0.0, 0.0, 1.0, 0.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ Embedding identity test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_repeated_tokens() -> TensorResult<()> {
    // Test embedding lookup with repeated tokens
    let vocab_size = 3;
    let d_model = 2;

    let weight = Tensor::<f32>::from_vec(
        vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ],
        vec![vocab_size, d_model]
    )?;

    // Repeated tokens: [0, 0, 1, 0, 2, 2]
    let token_ids = Tensor::<f32>::from_vec(
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 2.0],
        vec![6]
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Expected: [1,2], [1,2], [3,4], [1,2], [5,6], [5,6]
    let expected = vec![
        1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0, 5.0, 6.0,
    ];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ Embedding repeated tokens test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_single_token() -> TensorResult<()> {
    // Test embedding for single token (common in autoregressive generation)
    let vocab_size = 10;
    let d_model = 8;

    let weight = Tensor::<f32>::ones(vec![vocab_size, d_model])?;
    let token_ids = Tensor::<f32>::from_vec(vec![5.0], vec![1])?;

    let embeddings = weight.embedding(&token_ids)?;

    assert_eq!(embeddings.shape(), vec![1, d_model]);

    let result = embeddings.sync_and_read();
    for &val in &result {
        assert!((val - 1.0).abs() < 1e-5);
    }

    println!("✓ Embedding single token test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_sequential_tokens() -> TensorResult<()> {
    // Test embedding with sequential token IDs
    let vocab_size = 10;
    let d_model = 4;

    let weight_data: Vec<f32> = (0..vocab_size * d_model)
        .map(|i| i as f32)
        .collect();
    let weight = Tensor::<f32>::from_vec(weight_data, vec![vocab_size, d_model])?;

    // Sequential tokens: [0, 1, 2, 3, 4]
    let token_ids = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5])?;

    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Verify each embedding
    for i in 0..5 {
        for j in 0..d_model {
            let expected = (i * d_model + j) as f32;
            let actual = result[i * d_model + j];
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    println!("✓ Embedding sequential tokens test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_different_dimensions() -> TensorResult<()> {
    // Test various embedding dimensions
    let test_configs = vec![
        (10, 8),      // Small
        (100, 64),    // Medium
        (1000, 128),  // Large
        (5000, 256),  // Very large (like small LLMs)
    ];

    for (vocab_size, d_model) in test_configs {
        let weight = Tensor::<f32>::zeros(vec![vocab_size, d_model])?;
        let token_ids = Tensor::<f32>::from_vec(vec![0.0, 1.0], vec![2])?;

        let embeddings = weight.embedding(&token_ids)?;
        assert_eq!(
            embeddings.shape(),
            vec![2, d_model],
            "Failed for vocab_size={}, d_model={}",
            vocab_size, d_model
        );
    }

    println!("✓ Embedding different dimensions test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_batch_different_sizes() -> TensorResult<()> {
    // Test different batch and sequence length combinations
    let vocab_size = 100;
    let d_model = 32;

    let weight = Tensor::<f32>::zeros(vec![vocab_size, d_model])?;

    let test_shapes = vec![
        vec![1, 5],   // Single sequence, length 5
        vec![2, 10],  // 2 sequences, length 10 each
        vec![4, 8],   // 4 sequences, length 8 each
        vec![8, 16],  // 8 sequences, length 16 each
    ];

    for shape in test_shapes {
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_tokens = batch_size * seq_len;

        let token_ids = Tensor::<f32>::zeros(shape.clone())?;
        let embeddings = weight.embedding(&token_ids)?;

        let expected_shape = vec![batch_size, seq_len, d_model];
        assert_eq!(
            embeddings.shape(),
            expected_shape,
            "Failed for shape {:?}",
            shape
        );
    }

    println!("✓ Embedding batch different sizes test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_token_id_array_large() -> TensorResult<()> {
    // Test TokenIdArray with large token IDs (beyond f16 precision)
    let vocab_size = 32000;  // TinyLlama vocab size
    let d_model = 128;

    // Create sparse embedding matrix (mostly zeros, some non-zero values)
    let mut weight_data = vec![0.0f32; vocab_size * d_model];
    // Set specific rows to identifiable values
    for i in &[0, 100, 1000, 10000, 20000, 31999] {
        for j in 0..d_model {
            weight_data[i * d_model + j] = (*i as f32) + (j as f32) / 1000.0;
        }
    }

    let weight = Tensor::<f32>::from_vec(weight_data, vec![vocab_size, d_model])?;

    // Large token IDs that would lose precision in f16
    let token_ids = TokenIdArray::new(vec![0, 10000, 20000, 31999]);

    let embeddings = weight.embedding_from_token_ids(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Verify token 31999 embedding is correct
    let idx_31999 = 3 * d_model;
    for j in 0..d_model {
        let expected = 31999.0 + (j as f32) / 1000.0;
        let actual = result[idx_31999 + j];
        assert!(
            (actual - expected).abs() < 1e-3,
            "Token 31999 embedding mismatch at dim {}: expected {}, got {}",
            j, expected, actual
        );
    }

    println!("✓ Embedding TokenIdArray large tokens test passed");
    Ok(())
}

// Error handling tests

#[test]
#[serial]
#[should_panic(expected = "out of range")]
fn test_embedding_token_out_of_range() {
    // Test that out-of-range token ID causes error
    let vocab_size = 5;
    let d_model = 3;

    let weight = Tensor::<f32>::ones(vec![vocab_size, d_model]).unwrap();
    let token_ids = Tensor::<f32>::from_vec(vec![5.0], vec![1]).unwrap(); // 5 >= vocab_size

    let _ = weight.embedding(&token_ids).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "out of range")]
fn test_embedding_negative_token_id() {
    // Test that negative token ID causes error (if converted to large positive)
    let vocab_size = 10;
    let d_model = 4;

    let weight = Tensor::<f32>::ones(vec![vocab_size, d_model]).unwrap();
    let token_ids = Tensor::<f32>::from_vec(vec![-1.0], vec![1]).unwrap();

    let _ = weight.embedding(&token_ids).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "must be 2D")]
fn test_embedding_wrong_weight_dimensions_1d() {
    // Test that 1D weight causes error
    let weight = Tensor::<f32>::ones(vec![10]).unwrap();
    let token_ids = Tensor::<f32>::from_vec(vec![0.0], vec![1]).unwrap();

    let _ = weight.embedding(&token_ids).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "must be 2D")]
fn test_embedding_wrong_weight_dimensions_3d() {
    // Test that 3D weight causes error
    let weight = Tensor::<f32>::ones(vec![5, 3, 2]).unwrap();
    let token_ids = Tensor::<f32>::from_vec(vec![0.0], vec![1]).unwrap();

    let _ = weight.embedding(&token_ids).unwrap();
}

#[test]
#[serial]
fn test_embedding_edge_case_token_zero() -> TensorResult<()> {
    // Test that token ID 0 works correctly
    let vocab_size = 10;
    let d_model = 4;

    let weight = Tensor::<f32>::from_vec(
        (0..vocab_size * d_model).map(|i| i as f32).collect(),
        vec![vocab_size, d_model]
    )?;

    let token_ids = Tensor::<f32>::from_vec(vec![0.0], vec![1])?;
    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Should be first row: [0, 1, 2, 3]
    let expected = vec![0.0, 1.0, 2.0, 3.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ Embedding edge case token zero test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_edge_case_last_token() -> TensorResult<()> {
    // Test that last token ID (vocab_size - 1) works correctly
    let vocab_size = 10;
    let d_model = 4;

    let weight = Tensor::<f32>::from_vec(
        (0..vocab_size * d_model).map(|i| i as f32).collect(),
        vec![vocab_size, d_model]
    )?;

    let token_ids = Tensor::<f32>::from_vec(vec![9.0], vec![1])?; // Last valid token
    let embeddings = weight.embedding(&token_ids)?;
    let result = embeddings.sync_and_read();

    // Should be last row: [36, 37, 38, 39]
    let expected = vec![36.0, 37.0, 38.0, 39.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ Embedding edge case last token test passed");
    Ok(())
}

#[test]
#[serial]
fn test_embedding_preserves_device() -> TensorResult<()> {
    // Test that embedding preserves tensor device
    let device = MetalDevice::new()?;

    let vocab_size = 10;
    let d_model = 4;

    let weight = Tensor::<f32>::from_vec_gpu(
        &device,
        (0..vocab_size * d_model).map(|i| i as f32).collect(),
        vec![vocab_size, d_model]
    )?;

    let token_ids = Tensor::<f32>::from_vec_gpu(
        &device,
        vec![0.0, 1.0, 2.0],
        vec![3]
    )?;

    let embeddings = weight.embedding(&token_ids)?;

    // Result should also be on Metal device
    assert!(embeddings.device().is_metal(), "Embedding should preserve Metal device");

    println!("✓ Embedding preserves device test passed");
    Ok(())
}
