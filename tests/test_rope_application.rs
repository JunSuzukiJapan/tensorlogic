use tensorlogic::{GGUFWeightCache, MetalDevice};
use tensorlogic::tensor::{Tensor, TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use std::env;

#[test]
fn test_rope_q_application_prefill() {
    // Test 1: Verify RoPE is applied to Q during PREFILL at position 0

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 1: RoPE to Q during PREFILL (position=0)");

    let n_heads = 32;
    let head_dim = 64;
    let seq_len = 34;

    // Create test Q: [34, 32, 64]
    let q_data: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();

    println!("  Input Q shape: {:?}", q.dims());

    // Apply RoPE at position 0 (PREFILL)
    let q_rope = q.rope(0).unwrap();
    let q_rope_data = q_rope.sync_and_read();

    println!("  RoPE at position 0:");
    println!("    First 8 values: {:?}", &q_rope_data[..8]);

    // Test: RoPE should change the values
    let diff: f32 = q_data.iter()
        .zip(q_rope_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_data.len() as f32;

    println!("    Mean absolute difference from original: {:.6}", diff);

    assert!(diff > 0.001, "RoPE should modify Q values, got diff={}", diff);

    println!("  ✅ RoPE correctly applied to Q at position 0");
}

#[test]
fn test_rope_q_application_decode() {
    // Test 2: Verify RoPE is applied to Q during DECODE at incrementing positions

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 2: RoPE to Q during DECODE (position=34, 35, 36...)");

    let n_heads = 32;
    let head_dim = 64;

    // Single token Q: [1, 32, 64]
    let q_data: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![1, n_heads, head_dim]).unwrap();

    println!("  Input Q shape: {:?}", q.dims());

    // Apply RoPE at DECODE positions
    let positions = vec![34, 35, 36, 37, 38];
    let mut rope_outputs = Vec::new();

    for &pos in &positions {
        let q_rope = q.rope(pos).unwrap();
        let q_rope_data = q_rope.sync_and_read();
        rope_outputs.push(q_rope_data);
        println!("    Position {}: first 4 values: {:?}", pos, &rope_outputs.last().unwrap()[..4]);
    }

    // Test 1: Each position should be different from original
    for (i, &pos) in positions.iter().enumerate() {
        let diff: f32 = q_data.iter()
            .zip(rope_outputs[i].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / q_data.len() as f32;

        println!("    Position {} diff from original: {:.6}", pos, diff);
        assert!(diff > 0.001, "RoPE at position {} should modify Q", pos);
    }

    // Test 2: Consecutive positions should be different
    for i in 0..positions.len() - 1 {
        let diff: f32 = rope_outputs[i].iter()
            .zip(rope_outputs[i + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / rope_outputs[i].len() as f32;

        println!("    Position {} vs {}: diff={:.6}", positions[i], positions[i + 1], diff);
        assert!(diff > 0.001, "Consecutive positions should differ");
    }

    println!("  ✅ RoPE correctly applied to Q at all DECODE positions");
}

#[test]
fn test_rope_q_k_symmetry() {
    // Test 3: Verify Q and K get RoPE at same position

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 3: Q and K RoPE symmetry (same position)");

    let n_q_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;

    // Test at position 34 (first DECODE token)
    let position = 34;

    // Q: [1, 32, 64]
    let q_data: Vec<f32> = (0..n_q_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![1, n_q_heads, head_dim]).unwrap();
    let q_rope = q.rope(position).unwrap();
    let q_rope_data = q_rope.sync_and_read();

    // K: [1, 4, 64] (GQA)
    let k_data: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let k = Tensor::from_vec_gpu(&device, k_data.clone(), vec![1, n_kv_heads, head_dim]).unwrap();
    let k_rope = k.rope(position).unwrap();
    let k_rope_data = k_rope.sync_and_read();

    println!("  Position: {}", position);
    println!("  Q shape: {:?}, K shape: {:?}", q.dims(), k.dims());

    // Test: Both should be modified by RoPE
    let q_diff: f32 = q_data.iter()
        .zip(q_rope_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_data.len() as f32;

    let k_diff: f32 = k_data.iter()
        .zip(k_rope_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / k_data.len() as f32;

    println!("  Q RoPE diff from original: {:.6}", q_diff);
    println!("  K RoPE diff from original: {:.6}", k_diff);

    assert!(q_diff > 0.001, "Q should be modified by RoPE");
    assert!(k_diff > 0.001, "K should be modified by RoPE");

    // Test: Q and K RoPE at same position should have consistent effect
    // (Pattern of change should be similar for corresponding head dimensions)
    println!("  ✅ Both Q and K correctly modified by RoPE at position {}", position);
}

#[test]
fn test_rope_position_zero_vs_nonzero() {
    // Test 4: Verify position 0 (PREFILL) differs from position 34 (DECODE)

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 4: Position 0 (PREFILL) vs Position 34 (DECODE)");

    let n_heads = 32;
    let head_dim = 64;

    // Same Q input: [1, 32, 64]
    let q_data: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data, vec![1, n_heads, head_dim]).unwrap();

    // Apply RoPE at position 0 (PREFILL)
    let q_rope_pos0 = q.rope(0).unwrap();
    let q_rope_pos0_data = q_rope_pos0.sync_and_read();

    // Apply RoPE at position 34 (first DECODE)
    let q_rope_pos34 = q.rope(34).unwrap();
    let q_rope_pos34_data = q_rope_pos34.sync_and_read();

    println!("  Position 0 first 8 values: {:?}", &q_rope_pos0_data[..8]);
    println!("  Position 34 first 8 values: {:?}", &q_rope_pos34_data[..8]);

    // Test: Position 0 and 34 should produce different outputs
    let diff: f32 = q_rope_pos0_data.iter()
        .zip(q_rope_pos34_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_rope_pos0_data.len() as f32;

    println!("  Mean absolute difference: {:.6}", diff);

    assert!(diff > 0.01,
            "Position 0 and 34 should produce significantly different RoPE, got diff={}", diff);

    println!("  ✅ PREFILL (pos=0) and DECODE (pos=34) produce different RoPE");
}

#[test]
fn test_rope_reshape_correctness() {
    // Test 5: Verify reshape before/after RoPE preserves data correctly

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 5: Reshape correctness for apply_rope_q function");

    let seq_len = 1;
    let n_embd = 2048;  // 32 heads * 64 head_dim
    let n_heads = 32;
    let head_dim = 64;

    // Q flat: [1, 2048]
    let q_flat_data: Vec<f32> = (0..seq_len * n_embd)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q_flat = Tensor::from_vec_gpu(&device, q_flat_data.clone(), vec![seq_len, n_embd]).unwrap();

    println!("  Q flat shape: {:?}", q_flat.dims());

    // Reshape to heads: [1, 32, 64]
    let q_heads = q_flat.reshape(vec![seq_len, n_heads, head_dim]).unwrap();
    println!("  Q heads shape: {:?}", q_heads.dims());

    // Apply RoPE
    let q_rope = q_heads.rope(34).unwrap();
    println!("  Q RoPE shape: {:?}", q_rope.dims());

    // Reshape back to flat: [1, 2048]
    let q_final = q_rope.reshape(vec![seq_len, n_embd]).unwrap();
    println!("  Q final shape: {:?}", q_final.dims());

    let q_final_data = q_final.sync_and_read();

    // Test 1: Final shape should be [1, 2048]
    assert_eq!(q_final.dims(), vec![1, 2048], "Final shape should be [1, 2048]");

    // Test 2: Values should be different after RoPE
    let diff: f32 = q_flat_data.iter()
        .zip(q_final_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_flat_data.len() as f32;

    println!("  Difference after reshape->rope->reshape: {:.6}", diff);

    assert!(diff > 0.001, "RoPE should modify values through reshape pipeline");

    // Test 3: No NaN or Inf values
    let has_nan = q_final_data.iter().any(|x| x.is_nan());
    let has_inf = q_final_data.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Result should not contain NaN");
    assert!(!has_inf, "Result should not contain Inf");

    println!("  ✅ Reshape pipeline works correctly for apply_rope_q");
}

#[test]
fn test_apply_rope_q_function_simulation() {
    // Test 6: Simulate the apply_rope_q function behavior

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTest 6: Simulate apply_rope_q function");

    // Simulate apply_rope_q(Q, seq_len, pos)
    // Input: Q [1, 2048], Output: Q [1, 2048] with RoPE at position
    let seq_len = 1;
    let n_embd = 2048;
    let position = 34;

    // Q flat: [1, 2048]
    let q_flat_data: Vec<f32> = (0..seq_len * n_embd)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q_flat = Tensor::from_vec_gpu(&device, q_flat_data.clone(), vec![seq_len, n_embd]).unwrap();

    println!("  Input Q shape: {:?}", q_flat.dims());
    println!("  Position: {}", position);

    // Step 1: Reshape [1, 2048] -> [1, 32, 64]
    let q_heads = q_flat.reshape(vec![seq_len, 32, 64]).unwrap();
    println!("  After reshape to heads: {:?}", q_heads.dims());

    // Step 2: Apply RoPE at position
    let q_rope = q_heads.rope(position).unwrap();
    println!("  After RoPE: {:?}", q_rope.dims());

    // Step 3: Reshape back [1, 32, 64] -> [1, 2048]
    let q_final = q_rope.reshape(vec![seq_len, n_embd]).unwrap();
    println!("  After reshape to flat: {:?}", q_final.dims());

    let q_final_data = q_final.sync_and_read();

    // Test 1: Shape preserved
    assert_eq!(q_final.dims(), vec![1, 2048], "Output shape should be [1, 2048]");

    // Test 2: Values changed by RoPE
    let diff: f32 = q_flat_data.iter()
        .zip(q_final_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_flat_data.len() as f32;

    println!("  Mean difference: {:.6}", diff);
    assert!(diff > 0.001, "RoPE should modify values, got diff={}", diff);

    // Test 3: No NaN or Inf
    let has_nan = q_final_data.iter().any(|x| x.is_nan());
    let has_inf = q_final_data.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Result should not contain NaN");
    assert!(!has_inf, "Result should not contain Inf");

    // Test 4: Different positions produce different results
    let q_rope_pos0 = q_heads.rope(0).unwrap().reshape(vec![seq_len, n_embd]).unwrap();
    let q_rope_pos0_data = q_rope_pos0.sync_and_read();

    let diff_positions: f32 = q_final_data.iter()
        .zip(q_rope_pos0_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_final_data.len() as f32;

    println!("  Difference position 0 vs {}: {:.6}", position, diff_positions);
    assert!(diff_positions > 0.01,
            "Different positions should produce different RoPE, got diff={}",
            diff_positions);

    println!("  ✅ apply_rope_q function simulation works correctly");
}
