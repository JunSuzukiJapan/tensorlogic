/// Comprehensive tests for Rotary Position Embedding (RoPE)
///
/// RoPE is a crucial component for LLM position encoding.
/// Tests cover:
/// - Basic RoPE application
/// - Different dimensions (seq_len, n_heads, head_dim)
/// - Position offsets (for KV cache)
/// - Numerical stability
/// - Error cases (odd head_dim, wrong dimensions)

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use half::f16;
use serial_test::serial;

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

#[test]
#[serial]
fn test_rope_basic() -> TensorResult<()> {
    // Test basic RoPE application with simple input
    let device = MetalDevice::new()?;

    let seq_len = 2;
    let n_heads = 1;
    let head_dim = 4;

    // Create input tensor: [seq_len, n_heads, head_dim]
    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;

    // Apply RoPE with position offset 0
    let output = input.rope(0)?;

    // Check shape is preserved
    assert_eq!(output.shape().dims(), &[seq_len, n_heads, head_dim]);

    // Output should be different from input (rotation applied)
    let input_vec = input.sync_and_read();
    let output_vec = output.sync_and_read();

    // At least some values should have changed
    let mut changed_count = 0;
    for (i, o) in input_vec.iter().zip(output_vec.iter()) {
        if (i.to_f32() - o.to_f32()).abs() > 1e-3 {
            changed_count += 1;
        }
    }
    assert!(changed_count > 0, "RoPE should modify the input");

    println!("✓ RoPE basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_shape_preservation() -> TensorResult<()> {
    // Test that RoPE preserves tensor shape
    let test_shapes = vec![
        vec![4, 2, 8],      // [seq_len=4, n_heads=2, head_dim=8]
        vec![1, 1, 64],     // [seq_len=1, n_heads=1, head_dim=64]
        vec![10, 8, 32],    // [seq_len=10, n_heads=8, head_dim=32]
    ];

    for shape in test_shapes {
        let input = Tensor::<f16>::ones(shape.clone())?;
        let output = input.rope(0)?;
        assert_eq!(output.shape(), shape, "Shape not preserved for {:?}", shape);
    }

    println!("✓ RoPE shape preservation test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_position_offset() -> TensorResult<()> {
    // Test RoPE with different position offsets
    let seq_len = 3;
    let n_heads = 2;
    let head_dim = 8;

    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;

    // Apply RoPE with different position offsets
    let output_0 = input.rope(0)?;
    let output_5 = input.rope(5)?;
    let output_10 = input.rope(10)?;

    let vec_0 = output_0.sync_and_read();
    let vec_5 = output_5.sync_and_read();
    let vec_10 = output_10.sync_and_read();

    // Different offsets should produce different results
    let mut diff_count_0_5 = 0;
    let mut diff_count_5_10 = 0;

    for i in 0..vec_0.len() {
        if (vec_0[i].to_f32() - vec_5[i].to_f32()).abs() > 1e-3 {
            diff_count_0_5 += 1;
        }
        if (vec_5[i].to_f32() - vec_10[i].to_f32()).abs() > 1e-3 {
            diff_count_5_10 += 1;
        }
    }

    assert!(diff_count_0_5 > 0, "Position offsets 0 and 5 should produce different results");
    assert!(diff_count_5_10 > 0, "Position offsets 5 and 10 should produce different results");

    println!("✓ RoPE position offset test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_zeros_input() -> TensorResult<()> {
    // Test RoPE with all-zero input
    // RoPE of zeros should remain zeros (rotation of zero vector)
    let seq_len = 2;
    let n_heads = 1;
    let head_dim = 8;

    let input = Tensor::<f16>::zeros(&device, vec![seq_len, n_heads, head_dim])?;
    let output = input.rope(0)?;

    let result = output.sync_and_read();

    // All outputs should be approximately zero
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.to_f32().abs() < 1e-5,
            "Zero input should produce zero output, but got {} at index {}",
            val.to_f32(), i
        );
    }

    println!("✓ RoPE zeros input test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_deterministic() -> TensorResult<()> {
    // Test that RoPE is deterministic (same input + offset -> same output)
    let seq_len = 4;
    let n_heads = 2;
    let head_dim = 16;
    let position_offset = 7;

    let input = Tensor::<f16>::from_vec(
        (0..seq_len * n_heads * head_dim)
            .map(|i| f16::from_f32((i % 10) as f32 / 10.0))
            .collect(),
        vec![seq_len, n_heads, head_dim]
    )?;

    let output1 = input.rope(position_offset)?;
    let output2 = input.rope(position_offset)?;

    let vec1 = output1.sync_and_read();
    let vec2 = output2.sync_and_read();

    assert_tensor_close_f16(&vec1, &vec2, 1e-6);

    println!("✓ RoPE deterministic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_head_dim_variations() -> TensorResult<()> {
    // Test RoPE with various head dimensions (all even)
    let head_dims = vec![2, 4, 8, 16, 32, 64, 128];

    for head_dim in head_dims {
    let device = MetalDevice::new()?;
        let input = Tensor::<f16>::ones(&device, vec![2, 1, head_dim])?;
        let output = input.rope(0)?;

        assert_eq!(
            output.shape(),
            vec![2, 1, head_dim],
            "Failed for head_dim={}",
            head_dim
        );
    }

    println!("✓ RoPE head_dim variations test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_large_sequence() -> TensorResult<()> {
    // Test RoPE with large sequence length
    let seq_len = 256;  // Typical for inference
    let n_heads = 4;
    let head_dim = 64;

    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;
    let output = input.rope(0)?;

    assert_eq!(output.shape().dims(), &[seq_len, n_heads, head_dim]);

    // Check that output values are in reasonable range (not NaN or Inf)
    let result = output.sync_and_read();
    for (i, &val) in result.iter().enumerate() {
        let f = val.to_f32();
        assert!(f.is_finite(), "Non-finite value at index {}: {}", i, f);
        assert!(f.abs() <= 1000.0, "Value too large at index {}: {}", i, f);
    }

    println!("✓ RoPE large sequence test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_numerical_stability() -> TensorResult<()> {
    // Test numerical stability with various input magnitudes
    let seq_len = 4;
    let n_heads = 2;
    let head_dim = 8;

    // Test with small values
    let small_input = Tensor::<f16>::from_scalar(
        f16::from_f32(0.001),
        vec![seq_len, n_heads, head_dim]
    )?;
    let small_output = small_input.rope(0)?;
    let small_result = small_output.sync_and_read();

    for &val in &small_result {
        assert!(val.to_f32().is_finite(), "Small input produced non-finite value");
    }

    // Test with large values (within f16 range)
    let large_input = Tensor::<f16>::from_scalar(
        f16::from_f32(100.0),
        vec![seq_len, n_heads, head_dim]
    )?;
    let large_output = large_input.rope(0)?;
    let large_result = large_output.sync_and_read();

    for &val in &large_result {
        assert!(val.to_f32().is_finite(), "Large input produced non-finite value");
    }

    println!("✓ RoPE numerical stability test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_consistency_across_positions() -> TensorResult<()> {
    // Test that RoPE with position_offset=n on seq[0] equals
    // RoPE with position_offset=0 on seq[n]
    let n_heads = 1;
    let head_dim = 8;

    // Create a sequence of length 5
    let long_seq = Tensor::<f16>::from_vec(
        (0..5 * n_heads * head_dim)
            .map(|i| f16::from_f32((i % 10) as f32 / 10.0))
            .collect(),
        vec![5, n_heads, head_dim]
    )?;

    // Apply RoPE to the whole sequence
    let full_output = long_seq.rope(0)?;
    let full_vec = full_output.sync_and_read();

    // Extract position 2 from the full output
    let start_idx = 2 * n_heads * head_dim;
    let end_idx = 3 * n_heads * head_dim;
    let pos_2_from_full: Vec<f16> = full_vec[start_idx..end_idx].sync_and_read();

    // Now apply RoPE to just position 0 with offset 2
    let single_pos = Tensor::<f16>::from_vec(
        (0..n_heads * head_dim)
            .map(|i| f16::from_f32((i % 10) as f32 / 10.0))
            .collect(),
        vec![1, n_heads, head_dim]
    )?;
    let single_output = single_pos.rope(2)?;
    let single_vec = single_output.sync_and_read();

    // These should be approximately equal
    // Note: May have small differences due to numerical precision
    assert_tensor_close_f16(&pos_2_from_full, &single_vec, 1e-2);

    println!("✓ RoPE position consistency test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_multi_head() -> TensorResult<()> {
    // Test RoPE with multiple attention heads
    let seq_len = 4;
    let head_dims_and_heads = vec![
        (8, 1),   // Single head
        (16, 4),  // Multi-head
        (32, 8),  // Many heads
        (64, 32), // TinyLlama config
    ];

    for (head_dim, n_heads) in head_dims_and_heads {
    let device = MetalDevice::new()?;
        let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;
        let output = input.rope(0)?;

        assert_eq!(
            output.shape(),
            vec![seq_len, n_heads, head_dim],
            "Failed for n_heads={}, head_dim={}",
            n_heads, head_dim
        );

        // Check no NaN or Inf
        let result = output.sync_and_read();
        for &val in &result {
            assert!(val.to_f32().is_finite());
        }
    }

    println!("✓ RoPE multi-head test passed");
    Ok(())
}

// Error handling tests

#[test]
#[serial]
#[should_panic(expected = "even head_dim")]
fn test_rope_odd_head_dim() {
    // RoPE requires even head_dim for proper rotation
    let input = Tensor::<f16>::ones(&device, vec![2, 1, 5]).unwrap(); // head_dim=5 is odd
    let _ = input.rope(0).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "at least 3D")]
fn test_rope_insufficient_dimensions_2d() {
    let device = MetalDevice::new()?;
    // RoPE requires at least 3D tensor
    let input = Tensor::<f16>::ones(&device, vec![2, 4]).unwrap(); // Only 2D
    let _ = input.rope(0).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "at least 3D")]
fn test_rope_insufficient_dimensions_1d() {
    let device = MetalDevice::new()?;
    // RoPE requires at least 3D tensor
    let input = Tensor::<f16>::ones(&device, vec![8]).unwrap(); // Only 1D
    let _ = input.rope(0).unwrap();
}

#[test]
#[serial]
fn test_rope_4d_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test RoPE with 4D tensor (batch dimension)
    // Shape: [batch, seq_len, n_heads, head_dim]
    let batch_size = 2;
    let seq_len = 3;
    let n_heads = 4;
    let head_dim = 8;

    let input = Tensor::<f16>::ones(&device, vec![batch_size, seq_len, n_heads, head_dim])?;
    let output = input.rope(0)?;

    // Shape should be preserved
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, n_heads, head_dim]);

    // Last 3 dimensions are used for RoPE
    println!("✓ RoPE 4D tensor test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_single_position() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test RoPE with single position (seq_len=1)
    // This is common in autoregressive generation
    let seq_len = 1;
    let n_heads = 8;
    let head_dim = 64;

    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;
    let output = input.rope(0)?;

    assert_eq!(output.shape().dims(), &[seq_len, n_heads, head_dim]);

    println!("✓ RoPE single position test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_position_offset_range() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test RoPE with various position offset values
    let seq_len = 2;
    let n_heads = 2;
    let head_dim = 8;

    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;

    // Test offsets from 0 to 1000
    for offset in [0, 1, 10, 100, 500, 1000] {
        let output = input.rope(offset)?;

        // Should not panic or produce NaN
        let result = output.sync_and_read();
        for &val in &result {
            assert!(val.to_f32().is_finite(), "Non-finite value at offset {}", offset);
        }
    }

    println!("✓ RoPE position offset range test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_gradient_compatibility() -> TensorResult<()> {
    // Test that RoPE output can be used in gradient computation
    // (This is a prerequisite for training with RoPE)
    let seq_len = 2;
    let n_heads = 1;
    let head_dim = 8;

    let input = Tensor::<f16>::ones(&device, vec![seq_len, n_heads, head_dim])?;
    let output = input.rope(0)?;

    // Output should be valid for further operations
    let sum = output.sum()?;
    let sum_val = sum.sync_and_read()[0].to_f32();

    assert!(sum_val.is_finite(), "RoPE output sum should be finite");

    println!("✓ RoPE gradient compatibility test passed");
    Ok(())
}

#[test]
#[serial]
fn test_rope_kv_cache_simulation() -> TensorResult<()> {
    // Simulate KV cache scenario:
    // - First, process a prompt (position_offset=0)
    // - Then, generate tokens one by one (position_offset increases)

    let n_heads = 4;
    let head_dim = 16;

    // Initial prompt: 5 tokens
    let prompt = Tensor::<f16>::ones(&device, vec![5, n_heads, head_dim])?;
    let prompt_rope = prompt.rope(0)?;

    // Generate 3 new tokens, one at a time
    for i in 0..3 {
    let device = MetalDevice::new()?;
        let new_token = Tensor::<f16>::ones(&device, vec![1, n_heads, head_dim])?;
        let token_rope = new_token.rope(5 + i)?; // position_offset = prompt_len + i

        assert_eq!(token_rope.shape().dims(), &[1, n_heads, head_dim]);

        // Check no NaN
        let result = token_rope.sync_and_read();
        for &val in &result {
            assert!(val.to_f32().is_finite());
        }
    }

    println!("✓ RoPE KV cache simulation test passed");
    Ok(())
}
