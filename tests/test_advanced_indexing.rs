#![allow(unused_variables)]
//! Comprehensive tests for advanced indexing operations
//! Tests gather, scatter, and embedding lookup operations

use half::f16;
use tensorlogic::prelude::*;

// ============================================================================
// Gather Operations Tests
// ============================================================================

#[test]
fn test_gather_1d_basic() -> TensorResult<()> {
    let x = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
            f16::from_f32(40.0),
            f16::from_f32(50.0),
        ],
        vec![5],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(3.0),
            f16::from_f32(1.0),
        ],
        vec![3],
    )?;

    let result = x.gather(0, &indices)?;
    let values = result.sync_and_read();

    assert_eq!(values.len(), 3);
    assert!((values[0].to_f32() - 10.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 40.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 20.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_gather_1d_reverse() -> TensorResult<()> {
    let x = Tensor::from_vec(
        (0..10).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(9.0),
            f16::from_f32(8.0),
            f16::from_f32(7.0),
            f16::from_f32(6.0),
            f16::from_f32(5.0),
        ],
        vec![5],
    )?;

    let result = x.gather(0, &indices)?;
    let values = result.sync_and_read();

    for i in 0..5 {
        assert!((values[i].to_f32() - (9 - i) as f32).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_gather_2d_dim0() -> TensorResult<()> {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let x = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    // Gather row 1, row 0
    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ],
        vec![2, 3],
    )?;

    let result = x.gather(0, &indices)?;
    let values = result.sync_and_read();

    assert_eq!(result.shape().dims(), &[2, 3]);
    assert!((values[0].to_f32() - 4.0).abs() < 1e-3); // x[1, 0]
    assert!((values[1].to_f32() - 5.0).abs() < 1e-3); // x[1, 1]
    assert!((values[2].to_f32() - 3.0).abs() < 1e-3); // x[0, 2]
    Ok(())
}

#[test]
fn test_gather_2d_dim1() -> TensorResult<()> {
    let x = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    // Gather columns: [2, 0] for row 0, [1, 2] for row 1
    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ],
        vec![2, 2],
    )?;

    let result = x.gather(1, &indices)?;
    let values = result.sync_and_read();

    assert_eq!(result.shape().dims(), &[2, 2]);
    assert!((values[0].to_f32() - 3.0).abs() < 1e-3); // x[0, 2]
    assert!((values[1].to_f32() - 1.0).abs() < 1e-3); // x[0, 0]
    assert!((values[2].to_f32() - 5.0).abs() < 1e-3); // x[1, 1]
    assert!((values[3].to_f32() - 6.0).abs() < 1e-3); // x[1, 2]
    Ok(())
}

#[test]
fn test_gather_3d() -> TensorResult<()> {
    // [2, 2, 3] tensor
    let x = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 2, 3],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
        ],
        vec![2, 2, 3],
    )?;

    let result = x.gather(2, &indices)?;
    assert_eq!(result.shape().dims(), &[2, 2, 3]);
    Ok(())
}

#[test]
fn test_gather_duplicate_indices() -> TensorResult<()> {
    let x = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![3],
    )?;

    // Gather same index multiple times
    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ],
        vec![4],
    )?;

    let result = x.gather(0, &indices)?;
    let values = result.sync_and_read();

    for val in values {
        assert!((val.to_f32() - 20.0).abs() < 1e-3);
    }
    Ok(())
}

// ============================================================================
// Scatter Operations Tests
// ============================================================================

#[test]
fn test_scatter_1d_basic() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ZERO; 5], vec![5])?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
        ],
        vec![3],
    )?;

    let src = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![3],
    )?;

    let result = x.scatter(0, &indices, &src)?;
    let values = result.sync_and_read();

    assert_eq!(values.len(), 5);
    assert!((values[0].to_f32() - 10.0).abs() < 1e-3);
    assert!((values[1].to_f32()).abs() < 1e-3);
    assert!((values[2].to_f32() - 20.0).abs() < 1e-3);
    assert!((values[3].to_f32()).abs() < 1e-3);
    assert!((values[4].to_f32() - 30.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_scatter_1d_overwrite() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ONE; 5], vec![5])?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(3.0),
        ],
        vec![2],
    )?;

    let src = Tensor::from_vec(
        vec![
            f16::from_f32(100.0),
            f16::from_f32(200.0),
        ],
        vec![2],
    )?;

    let result = x.scatter(0, &indices, &src)?;
    let values = result.sync_and_read();

    assert!((values[0].to_f32() - 1.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 100.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 1.0).abs() < 1e-3);
    assert!((values[3].to_f32() - 200.0).abs() < 1e-3);
    assert!((values[4].to_f32() - 1.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_scatter_1d_duplicate() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ZERO; 5], vec![5])?;

    // Same index twice - last write wins
    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(2.0),
        ],
        vec![2],
    )?;

    let src = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
        ],
        vec![2],
    )?;

    let result = x.scatter(0, &indices, &src)?;
    let values = result.sync_and_read();

    assert!((values[2].to_f32() - 20.0).abs() < 1e-3); // Last write
    Ok(())
}

#[test]
fn test_scatter_2d_dim0() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ZERO; 12], vec![3, 4])?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
        ],
        vec![2, 4],
    )?;

    let src = Tensor::from_vec(
        (1..=8).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 4],
    )?;

    let result = x.scatter(0, &indices, &src)?;
    assert_eq!(result.shape().dims(), &[3, 4]);
    Ok(())
}

#[test]
fn test_scatter_2d_dim1() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ZERO; 12], vec![3, 4])?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(3.0),
            f16::from_f32(3.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(2.0),
            f16::from_f32(0.0),
            f16::from_f32(3.0),
            f16::from_f32(1.0),
        ],
        vec![3, 4],
    )?;

    let src = Tensor::from_vec(
        (1..=12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let result = x.scatter(1, &indices, &src)?;
    assert_eq!(result.shape().dims(), &[3, 4]);
    Ok(())
}

#[test]
fn test_scatter_partial() -> TensorResult<()> {
    // Scatter into only part of the tensor
    let x = Tensor::from_vec(vec![f16::from_f32(-1.0); 10], vec![10])?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(5.0),
            f16::from_f32(8.0),
        ],
        vec![3],
    )?;

    let src = Tensor::from_vec(
        vec![
            f16::from_f32(100.0),
            f16::from_f32(200.0),
            f16::from_f32(300.0),
        ],
        vec![3],
    )?;

    let result = x.scatter(0, &indices, &src)?;
    let values = result.sync_and_read();

    // Most values should remain -1.0
    assert!((values[0].to_f32() - (-1.0)).abs() < 1e-3);
    assert!((values[2].to_f32() - 100.0).abs() < 1e-3);
    assert!((values[5].to_f32() - 200.0).abs() < 1e-3);
    assert!((values[8].to_f32() - 300.0).abs() < 1e-3);
    Ok(())
}

// ============================================================================
// Gather + Scatter Round-trip Tests
// ============================================================================

#[test]
fn test_gather_scatter_roundtrip() -> TensorResult<()> {
    let original = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
        ],
        vec![5],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![5],
    )?;

    // Gather then scatter back
    let gathered = original.gather(0, &indices)?;
    let zeros = Tensor::from_vec(vec![f16::ZERO; 5], vec![5])?;
    let scattered = zeros.scatter(0, &indices, &gathered)?;

    let orig_vals = original.sync_and_read();
    let result_vals = scattered.sync_and_read();

    for i in 0..5 {
        assert!((orig_vals[i].to_f32() - result_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_scatter_gather_identity() -> TensorResult<()> {
    let src = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![3],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(3.0),
            f16::from_f32(5.0),
        ],
        vec![3],
    )?;

    let zeros = Tensor::from_vec(vec![f16::ZERO; 7], vec![7])?;
    let scattered = zeros.scatter(0, &indices, &src)?;
    let gathered = scattered.gather(0, &indices)?;

    let src_vals = src.sync_and_read();
    let result_vals = gathered.sync_and_read();

    for i in 0..3 {
        assert!((src_vals[i].to_f32() - result_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

// ============================================================================
// Embedding Tests
// ============================================================================

#[test]
fn test_embedding_basic() -> TensorResult<()> {
    // Vocabulary size 5, embedding dimension 3
    let weight = Tensor::from_vec(
        vec![
            // Token 0: [1, 2, 3]
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            // Token 1: [4, 5, 6]
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            // Token 2: [7, 8, 9]
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(9.0),
            // Token 3: [10, 11, 12]
            f16::from_f32(10.0),
            f16::from_f32(11.0),
            f16::from_f32(12.0),
            // Token 4: [13, 14, 15]
            f16::from_f32(13.0),
            f16::from_f32(14.0),
            f16::from_f32(15.0),
        ],
        vec![5, 3],
    )?;

    let token_ids = Tensor::from_vec(
        vec![f16::from_f32(2.0)], // Token 2
        vec![1],
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    assert_eq!(embeddings.shape().dims(), &[1, 3]);

    let values = embeddings.sync_and_read();
    assert!((values[0].to_f32() - 7.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 8.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 9.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_embedding_batch() -> TensorResult<()> {
    let weight = Tensor::from_vec(
        (0..15).map(|i| f16::from_f32(i as f32)).collect(),
        vec![5, 3],
    )?;

    // Batch of 4 tokens: [0, 1, 2, 3]
    let token_ids = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ],
        vec![4],
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    assert_eq!(embeddings.shape().dims(), &[4, 3]);
    Ok(())
}

#[test]
fn test_embedding_2d_tokens() -> TensorResult<()> {
    let weight = Tensor::from_vec(
        (0..15).map(|i| f16::from_f32(i as f32)).collect(),
        vec![5, 3],
    )?;

    // Token IDs with shape [2, 3] (batch=2, seq_len=3)
    let token_ids = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(0.0),
        ],
        vec![2, 3],
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    assert_eq!(embeddings.shape().dims(), &[2, 3, 3]);
    Ok(())
}

#[test]
fn test_embedding_repeated_tokens() -> TensorResult<()> {
    let weight = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32 * 10.0)).collect(),
        vec![4, 3],
    )?;

    // Same token repeated
    let token_ids = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(2.0),
            f16::from_f32(2.0),
        ],
        vec![3],
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    let values = embeddings.sync_and_read();

    // All should be the same (token 2's embedding)
    for i in 0..3 {
        let offset = i * 3;
        assert!((values[offset].to_f32() - 60.0).abs() < 1e-2);
        assert!((values[offset + 1].to_f32() - 70.0).abs() < 1e-2);
        assert!((values[offset + 2].to_f32() - 80.0).abs() < 1e-2);
    }
    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_gather_out_of_bounds() {
    let x = Tensor::from_vec(
        vec![f16::ONE; 5],
        vec![5],
    )
    .unwrap();

    let indices = Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(10.0)], // 10 is out of bounds
        vec![2],
    )
    .unwrap();

    assert!(x.gather(0, &indices).is_err());
}

#[test]
fn test_scatter_out_of_bounds() {
    let x = Tensor::from_vec(vec![f16::ZERO; 5], vec![5]).unwrap();

    let indices = Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(10.0)],
        vec![2],
    )
    .unwrap();

    let src = Tensor::from_vec(
        vec![f16::ONE; 2],
        vec![2],
    )
    .unwrap();

    assert!(x.scatter(0, &indices, &src).is_err());
}

#[test]
fn test_embedding_out_of_bounds() {
    let weight = Tensor::from_vec(
        vec![f16::ONE; 15],
        vec![5, 3],
    )
    .unwrap();

    let token_ids = Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(10.0)], // 10 >= vocab_size (5)
        vec![2],
    )
    .unwrap();

    assert!(weight.embedding(&token_ids).is_err());
}

#[test]
fn test_gather_invalid_dim() {
    let x = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3]).unwrap();
    let indices = Tensor::from_vec(vec![f16::ZERO; 6], vec![2, 3]).unwrap();

    // Dimension 2 doesn't exist
    assert!(x.gather(2, &indices).is_err());
}

#[test]
fn test_scatter_shape_mismatch() {
    let x = Tensor::from_vec(vec![f16::ZERO; 10], vec![10]).unwrap();

    let indices = Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![3],
    )
    .unwrap();

    // src has different shape than indices
    let src = Tensor::from_vec(
        vec![f16::ONE; 5],
        vec![5],
    )
    .unwrap();

    assert!(x.scatter(0, &indices, &src).is_err());
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_gather_large_tensor() -> TensorResult<()> {
    // Large 1D tensor
    let data: Vec<f16> = (0..1000).map(|i| f16::from_f32(i as f32)).collect();
    let x = Tensor::from_vec(data, vec![1000])?;

    let indices: Vec<f16> = (0..100).map(|i| f16::from_f32((i * 10) as f32)).collect();
    let indices_t = Tensor::from_vec(indices, vec![100])?;

    let result = x.gather(0, &indices_t)?;
    assert_eq!(result.shape().dims(), &[100]);

    let values = result.sync_and_read();
    for i in 0..100 {
        assert!((values[i].to_f32() - (i * 10) as f32).abs() < 1e-2);
    }
    Ok(())
}

#[test]
fn test_scatter_large_tensor() -> TensorResult<()> {
    let x = Tensor::from_vec(vec![f16::ZERO; 1000], vec![1000])?;

    let indices: Vec<f16> = (0..100).map(|i| f16::from_f32((i * 10) as f32)).collect();
    let indices_t = Tensor::from_vec(indices, vec![100])?;

    let src: Vec<f16> = (0..100).map(|i| f16::from_f32(i as f32 + 1000.0)).collect();
    let src_t = Tensor::from_vec(src, vec![100])?;

    let result = x.scatter(0, &indices_t, &src_t)?;
    let values = result.sync_and_read();

    for i in 0..100 {
        let idx = i * 10;
        assert!((values[idx].to_f32() - (i as f32 + 1000.0)).abs() < 1e-2);
    }
    Ok(())
}

#[test]
fn test_embedding_large_vocab() -> TensorResult<()> {
    // Large vocabulary: 1000 tokens, 128-dim embeddings
    let vocab_size = 1000;
    let d_model = 128;

    let weight_data: Vec<f16> = (0..vocab_size * d_model)
        .map(|i| f16::from_f32((i % 100) as f32))
        .collect();

    let weight = Tensor::from_vec(weight_data, vec![vocab_size, d_model])?;

    // Lookup 10 tokens
    let token_ids = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(10.0),
            f16::from_f32(100.0),
            f16::from_f32(200.0),
            f16::from_f32(500.0),
            f16::from_f32(999.0),
        ],
        vec![6],
    )?;

    let embeddings = weight.embedding(&token_ids)?;
    assert_eq!(embeddings.shape().dims(), &[6, d_model]);
    Ok(())
}
