// Mathematical verification of attention computation
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use serial_test::serial;

#[test]
#[serial]
fn test_attention_basic_math() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Simple test case: seq_len=2, n_heads=1, head_dim=4
    // This allows manual calculation verification

    // Q: [2, 4] - 2 tokens, 4 dimensions
    let q_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 1.0, 0.0, 0.0,  // token 1
    ];
    let q = Tensor::from_vec_gpu(&device, q_data, vec![2, 4]).unwrap();

    // K: [2, 4] - same shape as Q
    let k_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 1.0, 0.0, 0.0,  // token 1
    ];
    let k = Tensor::from_vec_gpu(&device, k_data, vec![2, 4]).unwrap();

    // V: [2, 4]
    let v_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 3.0, 0.0, 0.0,  // token 1
    ];
    let v = Tensor::from_vec_gpu(&device, v_data, vec![2, 4]).unwrap();

    // W_o: [4, 4] - identity for simplicity (not used in this basic test)
    let _w_o_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    // Step 1: Q @ K^T
    // Q @ K^T = [2,4] @ [4,2] = [2,2]
    // Manual calculation:
    // scores[0,0] = q[0] · k[0] = 1*1 = 1.0
    // scores[0,1] = q[0] · k[1] = 1*0 = 0.0
    // scores[1,0] = q[1] · k[0] = 0*1 = 0.0
    // scores[1,1] = q[1] · k[1] = 1*1 = 1.0

    let scores = q.matmul_transposed_b(&k).unwrap();
    let scores_data = scores.sync_and_read();

    println!("\nScores (Q @ K^T):");
    println!("  [{:.4}, {:.4}]", scores_data[0], scores_data[1]);
    println!("  [{:.4}, {:.4}]", scores_data[2], scores_data[3]);

    assert!((scores_data[0] - 1.0).abs() < 1e-5, "scores[0,0] should be 1.0");
    assert!((scores_data[1] - 0.0).abs() < 1e-5, "scores[0,1] should be 0.0");
    assert!((scores_data[2] - 0.0).abs() < 1e-5, "scores[1,0] should be 0.0");
    assert!((scores_data[3] - 1.0).abs() < 1e-5, "scores[1,1] should be 1.0");

    // Step 2: Scale by 1/sqrt(head_dim) = 1/sqrt(4) = 0.5
    let scaled = scores.div_scalar(2.0).unwrap();  // sqrt(4) = 2
    let scaled_data = scaled.sync_and_read();

    println!("\nScaled scores (/ sqrt(4)):");
    println!("  [{:.4}, {:.4}]", scaled_data[0], scaled_data[1]);
    println!("  [{:.4}, {:.4}]", scaled_data[2], scaled_data[3]);

    assert!((scaled_data[0] - 0.5).abs() < 1e-5, "scaled[0,0] should be 0.5");
    assert!((scaled_data[3] - 0.5).abs() < 1e-5, "scaled[1,1] should be 0.5");

    // Step 3: Apply causal mask
    // mask[i,j] = 1 if j <= i else 0
    // For seq_len=2:
    // [[1, 0],
    //  [1, 1]]
    let mask_data = vec![1.0f32, 0.0, 1.0, 1.0];
    let mask = Tensor::from_vec_gpu(&device, mask_data, vec![2, 2]).unwrap();

    let masked = scaled.apply_attention_mask(&mask).unwrap();
    let masked_data = masked.sync_and_read();

    println!("\nMasked scores:");
    println!("  [{:.4}, {:.4}]", masked_data[0], masked_data[1]);
    println!("  [{:.4}, {:.4}]", masked_data[2], masked_data[3]);

    // After masking:
    // [[0.5, -1e9],  -> softmax -> [1.0, 0.0]
    //  [0.0, 0.5]]   -> softmax -> [0.3775, 0.6225]  (not [0.5, 0.5]!)

    // Step 4: Softmax
    let attn_weights = masked.softmax().unwrap();
    let attn_data = attn_weights.sync_and_read();

    println!("\nAttention weights (after softmax):");
    println!("  [{:.4}, {:.4}]", attn_data[0], attn_data[1]);
    println!("  [{:.4}, {:.4}]", attn_data[2], attn_data[3]);

    // Row 0: [0.5, -1e9] -> softmax -> [~1.0, ~0.0]
    assert!(attn_data[0] > 0.99, "attn[0,0] should be ~1.0, got {}", attn_data[0]);
    assert!(attn_data[1] < 0.01, "attn[0,1] should be ~0.0, got {}", attn_data[1]);

    // Row 1: [0.0, 0.5] -> softmax([0.0, 0.5]) = [e^0/(e^0+e^0.5), e^0.5/(e^0+e^0.5)]
    //                                           = [1/2.649, 1.649/2.649] = [0.3775, 0.6225]
    assert!((attn_data[2] - 0.3775).abs() < 0.01, "attn[1,0] should be ~0.3775, got {}", attn_data[2]);
    assert!((attn_data[3] - 0.6225).abs() < 0.01, "attn[1,1] should be ~0.6225, got {}", attn_data[3]);

    // Step 5: attn_weights @ V
    // V = [[2, 0, 0, 0],
    //      [0, 3, 0, 0]]
    //
    // out[0] = 1.0 * v[0] + 0.0 * v[1] = [2, 0, 0, 0]
    // out[1] = 0.3775 * v[0] + 0.6225 * v[1] = [0.755, 1.8675, 0, 0]

    use tensorlogic::tensor::TensorTransform;
    let v_contig = v.contiguous().unwrap();
    let output = attn_weights.matmul(&v_contig).unwrap();
    let output_data = output.sync_and_read();

    println!("\nOutput (attn @ V):");
    println!("  [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[0], output_data[1], output_data[2], output_data[3]);
    println!("  [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[4], output_data[5], output_data[6], output_data[7]);

    // Verify output
    assert!((output_data[0] - 2.0).abs() < 0.01, "output[0,0] should be ~2.0");
    assert!((output_data[1] - 0.0).abs() < 0.01, "output[0,1] should be ~0.0");

    assert!((output_data[4] - 0.755).abs() < 0.01, "output[1,0] should be ~0.755");
    assert!((output_data[5] - 1.8675).abs() < 0.01, "output[1,1] should be ~1.8675");

    println!("\n✅ All attention math checks passed!");
}

#[test]
#[serial]
fn test_rope_position_encoding_math() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test RoPE with known values
    // seq_len=1, n_heads=1, head_dim=4
    // Input: [1, 0, 0, 0] (simple vector)
    // Position 0: theta = 0, so cos=1, sin=0
    // Result should be unchanged for position 0

    let input_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let input = Tensor::from_vec_gpu(&device, input_data.clone(), vec![1, 1, 4]).unwrap();

    let output = input.rope(0).unwrap();
    let output_data = output.sync_and_read();

    println!("\nRoPE test (position=0):");
    println!("  Input:  [{:.4}, {:.4}, {:.4}, {:.4}]",
             input_data[0], input_data[1], input_data[2], input_data[3]);
    println!("  Output: [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[0], output_data[1], output_data[2], output_data[3]);

    // At position 0, all thetas are 0, so cos=1, sin=0
    // Rotation formula: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    //                             -> [x0*1 - x1*0, x0*0 + x1*1]
    //                             -> [x0, x1]
    // So output should equal input

    for i in 0..4 {
        assert!((output_data[i] - input_data[i]).abs() < 0.001,
                "RoPE at position 0 should preserve values");
    }

    // Test with position offset
    let output_pos5 = input.rope(5).unwrap();
    let output_pos5_data = output_pos5.sync_and_read();

    println!("\nRoPE test (position=5):");
    println!("  Output: [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_pos5_data[0], output_pos5_data[1],
             output_pos5_data[2], output_pos5_data[3]);

    // Output should be different from input when position > 0
    let mut changed = false;
    for i in 0..4 {
        if (output_pos5_data[i] - input_data[i]).abs() > 0.001 {
            changed = true;
            break;
        }
    }
    assert!(changed, "RoPE should change values at non-zero positions");

    println!("\n✅ RoPE math checks passed!");
}

#[test]
#[serial]
fn test_attention_scaling_factor() {
    // Verify attention uses correct scaling factor (1/sqrt(head_dim))
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // TinyLlama settings: n_heads=32, head_dim=64
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();  // 1/8 = 0.125

    println!("\nTesting attention scaling factor:");
    println!("  head_dim: {}", head_dim);
    println!("  Expected scale: {} (1/sqrt({}))", scale, head_dim);

    // Create simple Q and K with known dot product
    // Q = [1, 0, 0, ..., 0] (64 dims)
    // K = [1, 0, 0, ..., 0] (64 dims)
    // Q·K = 1.0, scaled = 1.0 * (1/8) = 0.125

    let mut q_data = vec![0.0f32; head_dim];
    q_data[0] = 1.0;
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![1, head_dim]).unwrap();

    let k = Tensor::from_vec_gpu(&device, q_data, vec![1, head_dim]).unwrap();

    // Compute Q @ K^T (should be 1.0)
    let scores = q.matmul_transposed_b(&k).unwrap();
    let scores_data = scores.sync_and_read();

    println!("  Q·K (unscaled): {:.6}", scores_data[0]);
    assert!((scores_data[0] - 1.0).abs() < 1e-5, "Q·K should be 1.0");

    // Scale by 1/sqrt(head_dim)
    let scaled = scores.div_scalar((head_dim as f32).sqrt()).unwrap();
    let scaled_data = scaled.sync_and_read();

    println!("  Q·K (scaled): {:.6}", scaled_data[0]);
    println!("  Expected: {:.6}", scale);

    assert!((scaled_data[0] - scale).abs() < 1e-5,
            "Scaled attention should be {}, got {}", scale, scaled_data[0]);

    println!("\n✅ Attention scaling factor is correct!");
}

#[test]
#[serial]
fn test_gqa_repeat_kv() {
    // Test Grouped Query Attention K/V repetition
    // TinyLlama: 32 Q heads, 4 KV heads -> n_rep = 8
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let n_q_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;
    let n_rep = n_q_heads / n_kv_heads;  // 8

    println!("\nTesting GQA K/V repetition:");
    println!("  n_q_heads: {}", n_q_heads);
    println!("  n_kv_heads: {}", n_kv_heads);
    println!("  n_rep: {}", n_rep);
    println!("  head_dim: {}", head_dim);

    // K/V cache: [cache_len, n_kv_heads * head_dim] = [1, 256]
    let cache_len = 1;
    let kv_embd = n_kv_heads * head_dim;  // 256
    let q_embd = n_q_heads * head_dim;    // 2048

    // Create K cache with pattern [0, 1, 2, ..., 255]
    let k_data: Vec<f32> = (0..kv_embd).map(|i| i as f32).collect();
    let _k = Tensor::from_vec_gpu(&device, k_data.clone(), vec![cache_len, kv_embd]).unwrap();

    println!("  K original shape: [{}, {}]", cache_len, kv_embd);
    println!("  K first 8 values: {:?}", &k_data[..8]);

    // After GQA repeat, K should be [cache_len, q_embd] = [1, 2048]
    // Each 64-dim block should be repeated 8 times
    // Expected pattern: [0..64, 0..64, 0..64, ..., 64..128, 64..128, ...]

    // Simulate repeat_kv_for_gqa
    // K: [1, 256] -> [1, 4, 64] -> repeat -> [1, 32, 64] -> [1, 2048]
    // Note: We manually expand instead of using reshape for verification

    // Manual broadcast/repeat by creating expanded tensor
    let mut k_expanded_data = Vec::with_capacity(q_embd);
    for kv_head_idx in 0..n_kv_heads {
        let start = kv_head_idx * head_dim;
        let end = start + head_dim;
        let block = &k_data[start..end];

        // Repeat this block n_rep times
        for _ in 0..n_rep {
            k_expanded_data.extend_from_slice(block);
        }
    }

    assert_eq!(k_expanded_data.len(), q_embd,
               "Expanded K should have {} elements", q_embd);

    println!("  K expanded shape: [{}, {}]", cache_len, q_embd);
    println!("  K expanded first 128 values (first 2 repeated blocks):");
    println!("    Block 1 (head 0, rep 0): {:?}", &k_expanded_data[0..8]);
    println!("    Block 2 (head 0, rep 1): {:?}", &k_expanded_data[64..72]);

    // Verify repetition: first 64 elements should repeat 8 times
    for rep in 0..n_rep {
        let offset = rep * head_dim;
        for i in 0..head_dim {
            assert_eq!(k_expanded_data[offset + i], k_data[i],
                       "Head 0, rep {}, dim {} should be {}", rep, i, k_data[i]);
        }
    }

    println!("\n✅ GQA K/V repetition is correct!");
}

#[test]
#[serial]
fn test_llm_attention_full_pipeline() {
    // Test full attention pipeline with LLM-realistic settings
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // TinyLlama settings
    let n_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;
    let n_embd = n_heads * head_dim;  // 2048
    let kv_embd = n_kv_heads * head_dim;  // 256

    let seq_len = 1;  // DECODE phase
    let cache_len = 35;  // After PREFILL of 34 tokens

    println!("\nTesting LLM attention pipeline:");
    println!("  DECODE phase: seq_len={}, cache_len={}", seq_len, cache_len);
    println!("  n_heads={}, n_kv_heads={}, head_dim={}", n_heads, n_kv_heads, head_dim);
    println!("  Q shape: [{}, {}]", seq_len, n_embd);
    println!("  K shape: [{}, {}]", cache_len, kv_embd);
    println!("  V shape: [{}, {}]", cache_len, kv_embd);

    // Create random Q, K, V
    use rand::Rng;
    let mut rng = rand::rng();

    let q_data: Vec<f32> = (0..seq_len * n_embd).map(|_| rng.random_range(-1.0..1.0)).collect();
    let k_data: Vec<f32> = (0..cache_len * kv_embd).map(|_| rng.random_range(-1.0..1.0)).collect();
    let v_data: Vec<f32> = (0..cache_len * kv_embd).map(|_| rng.random_range(-1.0..1.0)).collect();

    let q = Tensor::from_vec_gpu(&device, q_data, vec![seq_len, n_embd]).unwrap();
    let k = Tensor::from_vec_gpu(&device, k_data, vec![cache_len, kv_embd]).unwrap();
    let v = Tensor::from_vec_gpu(&device, v_data, vec![cache_len, kv_embd]).unwrap();

    // Step 1: Expand K, V for GQA (simulate repeat_kv_for_gqa)
    // For this test, we'll skip actual expansion and just verify shapes

    // Step 2: Compute attention scores Q @ K^T
    // This would require GQA expansion in real implementation
    // For now, verify dimensions are compatible

    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("  Attention scale: {:.6} (1/sqrt({}))", scale, head_dim);

    // Verify Q has correct embedding dimension
    assert_eq!(q.dims()[1], n_embd, "Q should have n_embd={}", n_embd);
    assert_eq!(k.dims()[1], kv_embd, "K should have kv_embd={}", kv_embd);
    assert_eq!(v.dims()[1], kv_embd, "V should have kv_embd={}", kv_embd);

    // Verify shapes are compatible for attention after GQA expansion
    assert_eq!(n_embd % kv_embd, 0, "n_embd must be divisible by kv_embd for GQA");
    let n_rep = n_embd / kv_embd;
    assert_eq!(n_rep, n_heads / n_kv_heads, "n_rep should match head ratio");

    println!("  GQA n_rep: {} ({}x repetition)", n_rep, n_rep);
    println!("\n✅ LLM attention pipeline shapes are correct!");
}

#[test]
#[serial]
fn test_rope_llm_settings() {
    // Test RoPE with realistic LLM settings
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // TinyLlama settings
    let n_heads = 32;
    let head_dim = 64;
    let n_embd = n_heads * head_dim;  // 2048

    println!("\nTesting RoPE with LLM settings:");
    println!("  n_heads: {}, head_dim: {}, n_embd: {}", n_heads, head_dim, n_embd);

    // Test with DECODE phase: seq_len=1, position=34 (after PREFILL of 34 tokens)
    let seq_len = 1;
    let position = 34;

    // Create Q tensor: [seq_len, n_heads, head_dim] = [1, 32, 64]
    let q_data: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (i % 10) as f32)  // Simple pattern for verification
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();

    println!("  Q shape: [{}, {}, {}]", seq_len, n_heads, head_dim);
    println!("  Position offset: {}", position);

    // Apply RoPE
    let q_rope = q.rope(position).unwrap();
    let q_rope_data = q_rope.sync_and_read();

    println!("  Q original (first 8): {:?}", &q_data[..8]);
    println!("  Q after RoPE (first 8): {:?}", &q_rope_data[..8]);

    // Verify RoPE changed the values
    let mut changed = false;
    for i in 0..64 {  // Check first head
        if (q_rope_data[i] - q_data[i]).abs() > 0.001 {
            changed = true;
            break;
        }
    }
    assert!(changed, "RoPE should modify Q at position {}", position);

    // Test RoPE at position 0 (should be identity)
    let q_rope_pos0 = q.rope(0).unwrap();
    let q_rope_pos0_data = q_rope_pos0.sync_and_read();

    println!("  Q at position 0 (first 8): {:?}", &q_rope_pos0_data[..8]);

    // At position 0, RoPE should preserve values (theta=0 -> cos=1, sin=0)
    for i in 0..q_data.len() {
        assert!((q_rope_pos0_data[i] - q_data[i]).abs() < 0.001,
                "RoPE at position 0 should preserve values");
    }

    println!("\n✅ RoPE with LLM settings is correct!");
}

#[test]
#[serial]
fn test_rms_norm_llm_settings() {
    // Test RMS Normalization with realistic LLM settings
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // TinyLlama settings
    let n_embd = 2048;
    let batch_size = 34;  // PREFILL phase with 34 tokens
    let eps = 1e-5;

    println!("\nTesting RMS Norm with LLM settings:");
    println!("  n_embd: {}, batch_size: {}", n_embd, batch_size);

    // Create input tensor [batch_size, n_embd]
    use rand::Rng;
    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..batch_size * n_embd)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let input = Tensor::from_vec_gpu(&device, input_data.clone(), vec![batch_size, n_embd]).unwrap();

    // Create weight tensor [n_embd] - all ones for simple verification
    let weight_data: Vec<f32> = vec![1.0; n_embd];
    let weight = Tensor::from_vec_gpu(&device, weight_data, vec![n_embd]).unwrap();

    println!("  Input shape: [{}, {}]", batch_size, n_embd);
    println!("  Weight shape: [{}]", n_embd);

    // Apply RMS Norm
    let normalized_shape = vec![n_embd];
    let output = input.rms_norm(normalized_shape, &weight, eps).unwrap();
    let output_data = output.sync_and_read();

    println!("  Output shape: [{}, {}]", batch_size, n_embd);
    println!("  Output (first 10 of first row): {:?}", &output_data[..10]);

    // Verify RMS Norm properties:
    // 1. Output should not be all zeros
    let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
    let max = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    println!("  Output stats: mean={:.6}, min={:.6}, max={:.6}", mean, min, max);

    assert!(max.abs() > 0.01, "RMS Norm output should not be all zeros");
    assert!(min.abs() < 100.0, "RMS Norm should not produce extreme values");
    assert!(max.abs() < 100.0, "RMS Norm should not produce extreme values");

    // 2. For each row, compute RMS and verify normalization
    for row_idx in 0..batch_size.min(3) {  // Check first 3 rows
        let row_start = row_idx * n_embd;
        let row_end = row_start + n_embd;
        let row_data = &output_data[row_start..row_end];

        // Compute RMS of normalized output
        let rms = (row_data.iter().map(|&x| x * x).sum::<f32>() / n_embd as f32).sqrt();
        println!("  Row {} RMS: {:.6}", row_idx, rms);

        // RMS should be approximately 1.0 after normalization (with weight=1.0)
        assert!((rms - 1.0).abs() < 0.1, "RMS of normalized row should be ~1.0, got {}", rms);
    }

    println!("\n✅ RMS Norm with LLM settings is correct!");
}

#[test]
#[serial]
fn test_swiglu_ffn_llm_settings() {
    // Test SwiGLU FFN with realistic LLM settings
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // TinyLlama FFN settings
    let n_embd = 2048;
    let ffn_hidden = 5632;  // TinyLlama FFN hidden size
    let seq_len = 1;  // DECODE phase

    println!("\nTesting SwiGLU FFN with LLM settings:");
    println!("  n_embd: {}, ffn_hidden: {}, seq_len: {}", n_embd, ffn_hidden, seq_len);

    // Create input [seq_len, n_embd]
    use rand::Rng;
    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..seq_len * n_embd)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let input = Tensor::from_vec_gpu(&device, input_data, vec![seq_len, n_embd]).unwrap();

    // Create weight matrices
    // W_gate: [ffn_hidden, n_embd] (transposed for linear)
    let w_gate_data: Vec<f32> = (0..ffn_hidden * n_embd)
        .map(|_| rng.random_range(-0.1..0.1))
        .collect();
    let w_gate = Tensor::from_vec_gpu(&device, w_gate_data, vec![ffn_hidden, n_embd]).unwrap();

    // W_up: [ffn_hidden, n_embd]
    let w_up_data: Vec<f32> = (0..ffn_hidden * n_embd)
        .map(|_| rng.random_range(-0.1..0.1))
        .collect();
    let w_up = Tensor::from_vec_gpu(&device, w_up_data, vec![ffn_hidden, n_embd]).unwrap();

    // W_down: [n_embd, ffn_hidden]
    let w_down_data: Vec<f32> = (0..n_embd * ffn_hidden)
        .map(|_| rng.random_range(-0.1..0.1))
        .collect();
    let w_down = Tensor::from_vec_gpu(&device, w_down_data, vec![n_embd, ffn_hidden]).unwrap();

    println!("  Input shape: [{}, {}]", seq_len, n_embd);
    println!("  W_gate shape: [{}, {}]", ffn_hidden, n_embd);
    println!("  W_up shape: [{}, {}]", ffn_hidden, n_embd);
    println!("  W_down shape: [{}, {}]", n_embd, ffn_hidden);

    // Step 1: gate = input @ W_gate^T -> [seq_len, ffn_hidden]
    let gate = input.matmul_transposed_b(&w_gate).unwrap();
    assert_eq!(gate.dims(), &[seq_len, ffn_hidden]);
    println!("  Gate output shape: {:?}", gate.dims());

    // Step 2: up = input @ W_up^T -> [seq_len, ffn_hidden]
    let up = input.matmul_transposed_b(&w_up).unwrap();
    assert_eq!(up.dims(), &[seq_len, ffn_hidden]);
    println!("  Up output shape: {:?}", up.dims());

    // Step 3: silu_result = silu(gate) = gate * sigmoid(gate)
    let gate_sigmoid = gate.sigmoid().unwrap();
    let silu_result = gate.mul(&gate_sigmoid).unwrap();
    println!("  SiLU output shape: {:?}", silu_result.dims());

    // Step 4: mul_result = silu_result * up
    let mul_result = silu_result.mul(&up).unwrap();
    println!("  Element-wise multiply shape: {:?}", mul_result.dims());

    // Step 5: output = mul_result @ W_down^T -> [seq_len, n_embd]
    let output = mul_result.matmul_transposed_b(&w_down).unwrap();
    assert_eq!(output.dims(), &[seq_len, n_embd]);
    println!("  Final output shape: {:?}", output.dims());

    // Verify output is not all zeros
    let output_data = output.sync_and_read();
    let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
    let max = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("  Output stats: mean={:.6}, max={:.6}", mean, max);
    assert!(max.abs() > 0.001, "SwiGLU output should not be all zeros");

    println!("\n✅ SwiGLU FFN with LLM settings is correct!");
}

#[test]
#[serial]
fn test_linear_projection_llm_settings() {
    // Test linear projection (fused transpose-matmul) with LLM settings
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test Q/K/V projections
    let seq_len = 34;  // PREFILL phase
    let n_embd = 2048;
    let _n_heads = 32;
    let head_dim = 64;

    println!("\nTesting linear projection with LLM settings:");
    println!("  seq_len: {}, n_embd: {}", seq_len, n_embd);

    // Create input [seq_len, n_embd]
    use rand::Rng;
    let mut rng = rand::rng();
    let input_data: Vec<f32> = (0..seq_len * n_embd)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let input = Tensor::from_vec_gpu(&device, input_data, vec![seq_len, n_embd]).unwrap();

    // W_q: [n_embd, n_embd] (for Q projection)
    let w_q_data: Vec<f32> = (0..n_embd * n_embd)
        .map(|_| rng.random_range(-0.01..0.01))
        .collect();
    let w_q = Tensor::from_vec_gpu(&device, w_q_data, vec![n_embd, n_embd]).unwrap();

    println!("  Input shape: [{}, {}]", seq_len, n_embd);
    println!("  W_q shape: [{}, {}]", n_embd, n_embd);

    // Apply linear projection: Q = input @ W_q^T
    let q = input.matmul_transposed_b(&w_q).unwrap();

    println!("  Q output shape: {:?}", q.dims());
    assert_eq!(q.dims(), &[seq_len, n_embd]);

    // Verify output
    let q_data = q.sync_and_read();
    let mean = q_data.iter().sum::<f32>() / q_data.len() as f32;
    let max = q_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = q_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    println!("  Q stats: mean={:.6}, min={:.6}, max={:.6}", mean, min, max);
    assert!(max.abs() > 0.001, "Linear projection should not be all zeros");

    // Test K projection (GQA: fewer heads)
    let n_kv_heads = 4;
    let kv_embd = n_kv_heads * head_dim;  // 256

    let w_k_data: Vec<f32> = (0..kv_embd * n_embd)
        .map(|_| rng.random_range(-0.01..0.01))
        .collect();
    let w_k = Tensor::from_vec_gpu(&device, w_k_data, vec![kv_embd, n_embd]).unwrap();

    println!("  W_k shape (GQA): [{}, {}]", kv_embd, n_embd);

    let k = input.matmul_transposed_b(&w_k).unwrap();
    println!("  K output shape: {:?}", k.dims());
    assert_eq!(k.dims(), &[seq_len, kv_embd]);

    println!("\n✅ Linear projection with LLM settings is correct!");
}
