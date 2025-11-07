/// Comprehensive tests for Attention Masking operations
///
/// Attention masking is crucial for Transformer models, especially:
/// - Causal (autoregressive) masking for decoders
/// - Padding masking for variable-length sequences
/// - Combined masking strategies
///
/// Tests cover:
/// - Causal mask generation
/// - Padding mask generation
/// - Mask application to attention scores
/// - Mask combination (logical AND)
/// - Integration with softmax
/// - Various sequence lengths and batch sizes
/// - Error cases

use tensorlogic::prelude::*;

// Helper function to assert f16 tensors are close
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

// Test Causal Mask Generation

#[test]
fn test_causal_mask_small() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test causal mask for small sequence
    let mask = Tensor::<f16>::causal_mask(3)?;
    let data = mask.sync_and_read();

    // Expected: [[1, 0, 0],
    //            [1, 1, 0],
    //            [1, 1, 1]]
    assert_eq!(data[0], f16::ONE);  // [0,0]
    assert_eq!(data[1], f16::ZERO); // [0,1]
    assert_eq!(data[2], f16::ZERO); // [0,2]

    assert_eq!(data[3], f16::ONE);  // [1,0]
    assert_eq!(data[4], f16::ONE);  // [1,1]
    assert_eq!(data[5], f16::ZERO); // [1,2]

    assert_eq!(data[6], f16::ONE);  // [2,0]
    assert_eq!(data[7], f16::ONE);  // [2,1]
    assert_eq!(data[8], f16::ONE);  // [2,2]

    assert_eq!(mask.shape().dims(), &[3, 3]);

    println!("✓ Causal mask small test passed");
    Ok(())
}

#[test]
fn test_causal_mask_various_sizes() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test causal masks of various sizes
    for seq_len in [1, 2, 4, 8, 16, 32, 64, 128] {
        let mask = Tensor::<f16>::causal_mask(seq_len)?;
        let data = mask.sync_and_read();

        assert_eq!(mask.shape().dims(), &[seq_len, seq_len]);
        assert_eq!(data.len(), seq_len * seq_len);

        // Verify lower triangular structure
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if j <= i {
                    assert_eq!(
                        data[idx], f16::ONE,
                        "Expected 1 at [{},{}] for seq_len={}",
                        i, j, seq_len
                    );
                } else {
                    assert_eq!(
                        data[idx], f16::ZERO,
                        "Expected 0 at [{},{}] for seq_len={}",
                        i, j, seq_len
                    );
                }
            }
        }
    }

    println!("✓ Causal mask various sizes test passed");
    Ok(())
}

#[test]
fn test_causal_mask_single_token() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test causal mask for single token (common in generation)
    let mask = Tensor::<f16>::causal_mask(1)?;
    let data = mask.sync_and_read();

    // Should be [[1]]
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], f16::ONE);

    println!("✓ Causal mask single token test passed");
    Ok(())
}

#[test]
fn test_causal_mask_count_zeros_ones() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Verify the count of zeros and ones in causal mask
    for seq_len in [3, 5, 10] {
        let mask = Tensor::<f16>::causal_mask(seq_len)?;
        let data = mask.sync_and_read();

        let ones_count = data.iter().filter(|&&x| x == f16::ONE).count();
        let zeros_count = data.iter().filter(|&&x| x == f16::ZERO).count();

        // For a seq_len x seq_len lower triangular matrix:
        // ones_count = 1 + 2 + 3 + ... + seq_len = seq_len * (seq_len + 1) / 2
        let expected_ones = seq_len * (seq_len + 1) / 2;
        let expected_zeros = seq_len * seq_len - expected_ones;

        assert_eq!(
            ones_count, expected_ones,
            "Incorrect ones count for seq_len={}",
            seq_len
        );
        assert_eq!(
            zeros_count, expected_zeros,
            "Incorrect zeros count for seq_len={}",
            seq_len
        );
    }

    println!("✓ Causal mask count test passed");
    Ok(())
}

// Test Padding Mask Generation

#[test]
fn test_padding_mask_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test basic padding mask
    let mask = Tensor::<f16>::padding_mask(&[2, 3], 4)?;
    let data = mask.sync_and_read();

    // Expected: [[1, 1, 0, 0],
    //            [1, 1, 1, 0]]

    // First sequence (length 2)
    assert_eq!(data[0], f16::ONE);
    assert_eq!(data[1], f16::ONE);
    assert_eq!(data[2], f16::ZERO);
    assert_eq!(data[3], f16::ZERO);

    // Second sequence (length 3)
    assert_eq!(data[4], f16::ONE);
    assert_eq!(data[5], f16::ONE);
    assert_eq!(data[6], f16::ONE);
    assert_eq!(data[7], f16::ZERO);

    assert_eq!(mask.shape().dims(), &[2, 4]);

    println!("✓ Padding mask basic test passed");
    Ok(())
}

#[test]
fn test_padding_mask_no_padding() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test padding mask when all sequences are full length
    let mask = Tensor::<f16>::padding_mask(&[4, 4, 4], 4)?;
    let data = mask.sync_and_read();

    // All should be ones (no padding)
    for &val in &data {
        assert_eq!(val, f16::ONE);
    }

    assert_eq!(mask.shape().dims(), &[3, 4]);

    println!("✓ Padding mask no padding test passed");
    Ok(())
}

#[test]
fn test_padding_mask_all_padding() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test padding mask with zero-length sequences
    let mask = Tensor::<f16>::padding_mask(&[0, 0], 4)?;
    let data = mask.sync_and_read();

    // All should be zeros (all padding)
    for &val in &data {
        assert_eq!(val, f16::ZERO);
    }

    assert_eq!(mask.shape().dims(), &[2, 4]);

    println!("✓ Padding mask all padding test passed");
    Ok(())
}

#[test]
fn test_padding_mask_various_lengths() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test padding mask with various sequence lengths
    let lengths = vec![1, 3, 5, 7, 10];
    let max_len = 10;

    let mask = Tensor::<f16>::padding_mask(&lengths, max_len)?;
    let data = mask.sync_and_read();

    assert_eq!(mask.shape().dims(), &[lengths.len(), max_len]);

    // Verify each sequence
    for (seq_idx, &len) in lengths.iter().enumerate() {
        for pos in 0..max_len {
            let idx = seq_idx * max_len + pos;
            if pos < len {
                assert_eq!(
                    data[idx], f16::ONE,
                    "Expected 1 at seq={}, pos={} (len={})",
                    seq_idx, pos, len
                );
            } else {
                assert_eq!(
                    data[idx], f16::ZERO,
                    "Expected 0 at seq={}, pos={} (len={})",
                    seq_idx, pos, len
                );
            }
        }
    }

    println!("✓ Padding mask various lengths test passed");
    Ok(())
}

#[test]
fn test_padding_mask_single_sequence() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test padding mask with single sequence
    let mask = Tensor::<f16>::padding_mask(&[3], 5)?;
    let data = mask.sync_and_read();

    // Expected: [1, 1, 1, 0, 0]
    assert_eq!(data, vec![f16::ONE, f16::ONE, f16::ONE, f16::ZERO, f16::ZERO]);
    assert_eq!(mask.shape().dims(), &[1, 5]);

    println!("✓ Padding mask single sequence test passed");
    Ok(())
}

// Test Mask Application

#[test]
fn test_apply_attention_mask_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test basic mask application
    let scores = Tensor::from_vec(
        vec![
            f16::from_f32(1.0), f16::from_f32(2.0),
            f16::from_f32(3.0), f16::from_f32(4.0),
        ],
        vec![2, 2],
    )?;

    let mask = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
        vec![2, 2],
    )?;

    let result = scores.apply_attention_mask(&mask)?;
    let data = result.sync_and_read();

    assert_eq!(data[0], f16::from_f32(1.0));
    assert_eq!(data[1], f16::from_f32(-10000.0)); // masked
    assert_eq!(data[2], f16::from_f32(3.0));
    assert_eq!(data[3], f16::from_f32(4.0));

    println!("✓ Apply attention mask basic test passed");
    Ok(())
}

#[test]
fn test_apply_attention_mask_all_ones() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test mask application with all ones (no masking)
    let scores = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![1, 3],
    )?;

    let mask = Tensor::from_vec(
        vec![f16::ONE, f16::ONE, f16::ONE],
        vec![1, 3],
    )?;

    let result = scores.apply_attention_mask(&mask)?;
    let data = result.sync_and_read();

    // Should remain unchanged
    assert_eq!(data[0], f16::from_f32(1.0));
    assert_eq!(data[1], f16::from_f32(2.0));
    assert_eq!(data[2], f16::from_f32(3.0));

    println!("✓ Apply attention mask all ones test passed");
    Ok(())
}

#[test]
fn test_apply_attention_mask_all_zeros() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test mask application with all zeros (mask everything)
    let scores = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2],
    )?;

    let mask = Tensor::from_vec(
        vec![f16::ZERO, f16::ZERO, f16::ZERO, f16::ZERO],
        vec![2, 2],
    )?;

    let result = scores.apply_attention_mask(&mask)?;
    let data = result.sync_and_read();

    // All should be -10000
    for &val in &data {
        assert_eq!(val, f16::from_f32(-10000.0));
    }

    println!("✓ Apply attention mask all zeros test passed");
    Ok(())
}

#[test]
fn test_apply_causal_mask_to_scores() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test applying causal mask to attention scores
    let seq_len = 4;

    // Create dummy attention scores
    let scores = Tensor::from_vec(vec![f16::ONE; seq_len * seq_len], vec![seq_len, seq_len])?;

    // Create and apply causal mask
    let causal_mask = Tensor::<f16>::causal_mask(seq_len)?;
    let masked_scores = scores.apply_attention_mask(&causal_mask)?;

    let data = masked_scores.sync_and_read();

    // Check upper triangle is masked
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j <= i {
                assert_eq!(data[idx], f16::ONE, "Expected 1 at [{},{}]", i, j);
            } else {
                assert_eq!(data[idx], f16::from_f32(-10000.0), "Expected -10000 at [{},{}]", i, j);
            }
        }
    }

    println!("✓ Apply causal mask to scores test passed");
    Ok(())
}

// Test Mask Combination

#[test]
fn test_combine_masks_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test basic mask combination (logical AND)
    let mask1 = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
        vec![2, 2],
    )?;

    let mask2 = Tensor::from_vec(
        vec![f16::ONE, f16::ONE, f16::ZERO, f16::ONE],
        vec![2, 2],
    )?;

    let combined = mask1.combine_masks(&mask2)?;
    let data = combined.sync_and_read();

    // Logical AND
    assert_eq!(data[0], f16::ONE);  // 1 & 1 = 1
    assert_eq!(data[1], f16::ZERO); // 0 & 1 = 0
    assert_eq!(data[2], f16::ZERO); // 1 & 0 = 0
    assert_eq!(data[3], f16::ONE);  // 1 & 1 = 1

    println!("✓ Combine masks basic test passed");
    Ok(())
}

#[test]
fn test_combine_masks_with_itself() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test combining mask with itself (should be identity)
    let mask = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ZERO, f16::ONE],
        vec![2, 2],
    )?;

    let combined = mask.combine_masks(&mask)?;
    let data = combined.sync_and_read();

    // Should be same as original
    assert_eq!(data[0], f16::ONE);
    assert_eq!(data[1], f16::ZERO);
    assert_eq!(data[2], f16::ZERO);
    assert_eq!(data[3], f16::ONE);

    println!("✓ Combine masks with itself test passed");
    Ok(())
}

#[test]
fn test_combine_causal_and_padding() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test combining causal mask with padding mask
    // This is a common pattern in transformer decoders

    let seq_len = 4;
    let causal = Tensor::<f16>::causal_mask(seq_len)?;

    // Create padding mask: first 3 positions are valid, last is padding
    let padding = Tensor::from_vec(
        vec![
            f16::ONE, f16::ONE, f16::ONE, f16::ZERO,
            f16::ONE, f16::ONE, f16::ONE, f16::ZERO,
            f16::ONE, f16::ONE, f16::ONE, f16::ZERO,
            f16::ONE, f16::ONE, f16::ONE, f16::ZERO,
        ],
        vec![seq_len, seq_len],
    )?;

    let combined = causal.combine_masks(&padding)?;
    let data = combined.sync_and_read();

    // Check: causal pattern in first 3 columns, all zeros in last column
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j == 3 {
                // Last column should be all zeros (padding)
                assert_eq!(data[idx], f16::ZERO);
            } else if j <= i {
                // Causal pattern in first 3 columns
                assert_eq!(data[idx], f16::ONE);
            } else {
                // Upper triangle of first 3 columns
                assert_eq!(data[idx], f16::ZERO);
            }
        }
    }

    println!("✓ Combine causal and padding test passed");
    Ok(())
}

// Integration Tests with Softmax

#[test]
fn test_mask_with_softmax() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test that masked positions become ~0 after softmax
    let scores = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![1, 3],
    )?;

    // Mask out the middle position
    let mask = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ONE],
        vec![1, 3],
    )?;

    let masked_scores = scores.apply_attention_mask(&mask)?;

    // Apply softmax
    let softmax_output = masked_scores.softmax()?; // dim=1
    let data = softmax_output.sync_and_read();

    // Middle position should be very close to 0
    assert!(
        data[1].to_f32() < 1e-4,
        "Masked position should be ~0 after softmax, got {}",
        data[1].to_f32()
    );

    // First and last positions should sum to ~1
    let sum = data[0].to_f32() + data[2].to_f32();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Unmasked positions should sum to ~1, got {}",
        sum
    );

    println!("✓ Mask with softmax test passed");
    Ok(())
}

#[test]
fn test_causal_attention_simulation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate causal attention: each position can only attend to previous positions
    let seq_len = 4;

    // Create attention scores (Q @ K^T)
    let scores = Tensor::from_vec(vec![f16::ONE; seq_len * seq_len], vec![seq_len, seq_len])?;

    // Apply causal mask
    let causal_mask = Tensor::<f16>::causal_mask(seq_len)?;
    let masked_scores = scores.apply_attention_mask(&causal_mask)?;

    // Apply softmax
    let attn_weights = masked_scores.softmax()?; // softmax over keys (dim=1)
    let data = attn_weights.sync_and_read();

    // For each query position, check attention weights
    for i in 0..seq_len {
        let mut sum = 0.0;

        for j in 0..seq_len {
            let idx = i * seq_len + j;
            let weight = data[idx].to_f32();

            if j > i {
                // Future positions should have ~0 weight
                assert!(
                    weight < 1e-4,
                    "Position {} should not attend to future position {}, got weight {}",
                    i, j, weight
                );
            } else {
                // Past positions should have positive weight
                assert!(
                    weight > 0.0,
                    "Position {} should attend to position {}, got weight {}",
                    i, j, weight
                );
                sum += weight;
            }
        }

        // Weights should sum to ~1
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Attention weights for position {} should sum to ~1, got {}",
            i, sum
        );
    }

    println!("✓ Causal attention simulation test passed");
    Ok(())
}

// Large Scale Tests

#[test]
fn test_causal_mask_large_sequence() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test causal mask for large sequence (e.g., 2048 tokens)
    let seq_len = 512; // Reduced from 2048 for faster testing

    let mask = Tensor::<f16>::causal_mask(seq_len)?;

    assert_eq!(mask.shape().dims(), &[seq_len, seq_len]);

    // Spot check a few positions
    let data = mask.sync_and_read();

    // First row: only [0,0] should be 1
    assert_eq!(data[0], f16::ONE);
    assert_eq!(data[1], f16::ZERO);

    // Last row: all should be 1
    let last_row_start = (seq_len - 1) * seq_len;
    for i in 0..seq_len {
        assert_eq!(data[last_row_start + i], f16::ONE);
    }

    // Middle diagonal
    let mid = seq_len / 2;
    assert_eq!(data[mid * seq_len + mid], f16::ONE);     // diagonal
    assert_eq!(data[mid * seq_len + mid - 1], f16::ONE); // before diagonal
    assert_eq!(data[mid * seq_len + mid + 1], f16::ZERO);// after diagonal

    println!("✓ Causal mask large sequence test passed");
    Ok(())
}

#[test]
fn test_padding_mask_large_batch() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test padding mask with large batch
    let batch_size = 64;
    let max_len = 128;

    // Create varying lengths
    let lengths: Vec<usize> = (0..batch_size)
        .map(|i| ((i * max_len / batch_size) + 1).min(max_len))
        .collect();

    let mask = Tensor::<f16>::padding_mask(&lengths, max_len)?;

    assert_eq!(mask.shape().dims(), &[batch_size, max_len]);

    // Verify shape
    let data = mask.sync_and_read();
    assert_eq!(data.len(), batch_size * max_len);

    // Spot check a few sequences
    for seq_idx in [0, batch_size / 4, batch_size / 2, batch_size - 1] {
        let len = lengths[seq_idx];
        let row_start = seq_idx * max_len;

        // Check valid positions
        assert_eq!(data[row_start], f16::ONE);
        if len > 0 {
            assert_eq!(data[row_start + len - 1], f16::ONE);
        }

        // Check padding positions
        if len < max_len {
            assert_eq!(data[row_start + len], f16::ZERO);
            assert_eq!(data[row_start + max_len - 1], f16::ZERO);
        }
    }

    println!("✓ Padding mask large batch test passed");
    Ok(())
}

// Error Handling Tests

#[test]
#[should_panic(expected = "ShapeMismatch")]
fn test_apply_mask_shape_mismatch() {
    // Test that shape mismatch causes error
    let scores = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2],
    ).unwrap();

    let mask = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ONE],
        vec![1, 3],
    ).unwrap();

    let _ = scores.apply_attention_mask(&mask).unwrap();
}

#[test]
#[should_panic(expected = "ShapeMismatch")]
fn test_combine_masks_shape_mismatch() {
    // Test that combining masks with different shapes causes error
    let mask1 = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO],
        vec![1, 2],
    ).unwrap();

    let mask2 = Tensor::from_vec(
        vec![f16::ONE, f16::ZERO, f16::ONE],
        vec![1, 3],
    ).unwrap();

    let _ = mask1.combine_masks(&mask2).unwrap();
}

#[test]
fn test_empty_padding_mask() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test edge case: empty batch
    let mask = Tensor::<f16>::padding_mask(&[], 10)?;

    assert_eq!(mask.shape().dims(), &[0, 10]);

    println!("✓ Empty padding mask test passed");
    Ok(())
}

#[test]
fn test_mask_application_preserves_shape() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    // Test that mask application preserves tensor shape
    let test_shapes = vec![
        vec![4, 4],
        vec![8, 8],
        vec![1, 16],
        vec![16, 1],
    ];

    for shape in test_shapes {
        let scores = Tensor::<f16>::ones(&device, shape.clone())?;
        let mask = Tensor::<f16>::ones(&device, shape.clone())?;

        let result = scores.apply_attention_mask(&mask)?;

        assert_eq!(result.shape().dims(), &shape, "Shape not preserved for {:?}", shape);
    }

    println!("✓ Mask application preserves shape test passed");
    Ok(())
}
