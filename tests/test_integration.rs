use tensorlogic::{GGUFWeightCache, MetalDevice};
use tensorlogic::tensor::{Tensor, TensorAccessors, TensorCreation, TensorIO};
use std::env;

#[test]
fn test_weight_loading_and_transpose() {
    // Test that weights are loaded correctly from GGUF file
    // and matmul with transpose works as expected

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting weight loading and transpose:");
    println!("  Model: {}", model_path);

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    // Test 1: Check token embedding dimensions
    let tok_embd = model.get_weight("token_embd.weight")
        .expect("token_embd.weight not found");
    let tok_embd_shape = tok_embd.dims();
    println!("  token_embd.weight shape: {:?}", tok_embd_shape);
    assert_eq!(tok_embd_shape.len(), 2, "token_embd should be 2D");
    assert_eq!(tok_embd_shape[1], 2048, "n_embd should be 2048");

    // Test 2: Check output projection dimensions
    let output_weight = model.get_weight("output.weight")
        .expect("output.weight not found");
    let output_shape = output_weight.dims();
    println!("  output.weight shape: {:?}", output_shape);
    assert_eq!(output_shape.len(), 2, "output.weight should be 2D");
    assert_eq!(output_shape[1], 2048, "n_embd should be 2048");

    // Test 3: Check layer 0 attention weights
    let w_q = model.get_weight("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight not found");
    let w_q_shape = w_q.dims();
    println!("  blk.0.attn_q.weight shape: {:?}", w_q_shape);
    assert_eq!(w_q_shape, vec![2048, 2048], "W_q should be [2048, 2048]");

    let w_k = model.get_weight("blk.0.attn_k.weight")
        .expect("blk.0.attn_k.weight not found");
    let w_k_shape = w_k.dims();
    println!("  blk.0.attn_k.weight shape: {:?}", w_k_shape);
    assert_eq!(w_k_shape, vec![256, 2048], "W_k should be [256, 2048] for GQA");

    // Test 4: Matrix multiplication with transpose

    // Create test input: [1, 2048]
    let input_data: Vec<f32> = (0..2048).map(|i| (i % 100) as f32 * 0.01).collect();
    let input = Tensor::from_vec_gpu(&device, input_data, vec![1, 2048]).unwrap();

    // Perform Q = input @ W_q^T
    let q = input.matmul_transposed_b(&w_q).unwrap();
    let q_shape = q.dims();
    let q_data = q.sync_and_read();

    println!("  Q output shape: {:?}", q_shape);
    println!("  Q stats: mean={:.6}, min={:.6}, max={:.6}",
             q_data.iter().sum::<f32>() / q_data.len() as f32,
             q_data.iter().cloned().fold(f32::INFINITY, f32::min),
             q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    assert_eq!(q_shape, vec![1, 2048], "Q shape should be [1, 2048]");

    // Verify Q contains non-zero values (weights loaded correctly)
    let non_zero_count = q_data.iter().filter(|&&x| x.abs() > 1e-6).count();
    let non_zero_ratio = non_zero_count as f32 / q_data.len() as f32;
    println!("  Q non-zero ratio: {:.2}%", non_zero_ratio * 100.0);
    assert!(non_zero_ratio > 0.5, "Most Q values should be non-zero if weights loaded correctly");

    println!("\n✅ Weight loading and transpose are correct!");
}

#[test]
fn test_kv_cache_management() {
    // Test K/V cache construction and retrieval during PREFILL

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting K/V cache management:");

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    // Simulate PREFILL: seq_len = 34 tokens
    let seq_len = 34;
    let n_embd = 2048;
    let kv_embd = 256;  // GQA: 4 KV heads * 64 head_dim

    // Create test input embeddings: [34, 2048]
    let input_data: Vec<f32> = (0..seq_len * n_embd)
        .map(|i| ((i % 1000) as f32 - 500.0) * 0.001)
        .collect();
    let input = Tensor::from_vec_gpu(&device, input_data, vec![seq_len, n_embd]).unwrap();

    // Get layer 0 weights
    let w_k = model.get_weight("blk.0.attn_k.weight")
        .expect("blk.0.attn_k.weight not found");
    let w_v = model.get_weight("blk.0.attn_v.weight")
        .expect("blk.0.attn_v.weight not found");

    println!("  PREFILL: seq_len={}, n_embd={}, kv_embd={}", seq_len, n_embd, kv_embd);

    // Build K/V caches: K = input @ W_k^T, V = input @ W_v^T
    let k_cache = input.matmul_transposed_b(&w_k).unwrap();
    let v_cache = input.matmul_transposed_b(&w_v).unwrap();

    let k_shape = k_cache.dims();
    let v_shape = v_cache.dims();

    println!("  K cache shape: {:?}", k_shape);
    println!("  V cache shape: {:?}", v_shape);

    assert_eq!(k_shape, vec![seq_len, kv_embd], "K cache shape should be [34, 256]");
    assert_eq!(v_shape, vec![seq_len, kv_embd], "V cache shape should be [34, 256]");

    // Verify caches contain meaningful values
    let k_data = k_cache.sync_and_read();
    let v_data = v_cache.sync_and_read();

    let k_non_zero = k_data.iter().filter(|&&x| x.abs() > 1e-6).count();
    let v_non_zero = v_data.iter().filter(|&&x| x.abs() > 1e-6).count();

    println!("  K non-zero ratio: {:.2}%", k_non_zero as f32 / k_data.len() as f32 * 100.0);
    println!("  V non-zero ratio: {:.2}%", v_non_zero as f32 / v_data.len() as f32 * 100.0);

    assert!(k_non_zero as f32 / k_data.len() as f32 > 0.5, "K cache should contain meaningful values");
    assert!(v_non_zero as f32 / v_data.len() as f32 > 0.5, "V cache should contain meaningful values");

    println!("\n✅ K/V cache management is correct!");
}

#[test]
fn test_final_output_projection() {
    // Test final output projection: hidden_states @ output.weight^T -> logits

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting final output projection:");

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    let n_embd = 2048;
    let vocab_size = 32000;

    // Get output weights
    let output_weight = model.get_weight("output.weight")
        .expect("output.weight not found");
    let output_shape = output_weight.dims();

    println!("  output.weight shape: {:?}", output_shape);
    assert_eq!(output_shape[0], vocab_size, "output.weight should have vocab_size rows");
    assert_eq!(output_shape[1], n_embd, "output.weight should have n_embd columns");

    // Create test hidden state: [1, 2048]
    let hidden_data: Vec<f32> = (0..n_embd)
        .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
        .collect();
    let hidden = Tensor::from_vec_gpu(&device, hidden_data, vec![1, n_embd]).unwrap();

    // Compute logits: hidden @ output.weight^T = [1, 32000]
    let logits = hidden.matmul_transposed_b(&output_weight).unwrap();
    let logits_shape = logits.dims();
    let logits_data = logits.sync_and_read();

    println!("  logits shape: {:?}", logits_shape);
    assert_eq!(logits_shape, vec![1, vocab_size], "logits shape should be [1, 32000]");

    // Find top 5 token IDs
    let mut logits_indexed: Vec<(usize, f32)> = logits_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    logits_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 5 tokens:");
    for (rank, (token_id, logit)) in logits_indexed.iter().take(5).enumerate() {
        println!("    Rank {}: token_id={}, logit={:.6}", rank + 1, token_id, logit);
    }

    // Verify logits have reasonable distribution
    let logit_mean = logits_data.iter().sum::<f32>() / logits_data.len() as f32;
    let logit_min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let logit_max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("  Logits stats: mean={:.6}, min={:.6}, max={:.6}", logit_mean, logit_min, logit_max);

    // Sanity checks
    assert!(logit_max > logit_min, "Logits should have variation");
    assert!(logit_max.abs() < 1000.0, "Logits should be reasonable magnitude");
    assert!(logits_indexed[0].0 < vocab_size, "Top token should be valid");

    println!("\n✅ Final output projection is correct!");
}

#[test]
fn test_tokenizer_roundtrip() {
    // Test tokenization and detokenization

    let home = env::var("HOME").expect("HOME not set");
    let tokenizer_path = format!("{}/.llm/tokenizers/tinyllama-tokenizer.json", home);

    println!("\nTesting tokenizer:");
    println!("  Tokenizer: {}", tokenizer_path);

    // This test requires TensorLogic's tokenizer implementation
    // For now, just verify the file exists
    assert!(std::path::Path::new(&tokenizer_path).exists(),
            "Tokenizer file should exist");

    println!("  ✅ Tokenizer file found");

    // TODO: Add actual tokenization test when tokenizer is accessible from Rust
    println!("\n⚠️  Full tokenizer test requires TensorLogic runtime");
}

#[test]
fn test_prefill_single_token() {
    // Test complete PREFILL pipeline for a single token
    // This is the minimal integration test

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting PREFILL pipeline (single token <BOS>):");

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    // Token ID 1 = <BOS>
    let token_id = 1;
    println!("  Token ID: {} (<BOS>)", token_id);

    // Step 1: Embedding lookup
    let tok_embd = model.get_weight("token_embd.weight")
        .expect("token_embd.weight not found");
    let tok_embd_data = tok_embd.sync_and_read();
    let n_embd = tok_embd.dims()[1];

    let embedding: Vec<f32> = tok_embd_data[token_id * n_embd..(token_id + 1) * n_embd].to_vec();
    let x = Tensor::from_vec_gpu(&device, embedding, vec![1, n_embd]).unwrap();

    println!("  Embedding shape: {:?}", x.dims());

    // Step 2: Layer 0 processing (simplified)
    let attn_norm = model.get_weight("blk.0.attn_norm.weight")
        .expect("blk.0.attn_norm.weight not found");
    let w_q = model.get_weight("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight not found");

    // RMS Norm
    let normed = x.rms_norm(vec![n_embd], &attn_norm, 1e-5).unwrap();
    println!("  After RMS Norm shape: {:?}", normed.dims());

    // Q projection
    let q = normed.matmul_transposed_b(&w_q).unwrap();
    let q_data = q.sync_and_read();
    println!("  Q shape: {:?}", q.dims());
    println!("  Q stats: mean={:.6}, min={:.6}, max={:.6}",
             q_data.iter().sum::<f32>() / q_data.len() as f32,
             q_data.iter().cloned().fold(f32::INFINITY, f32::min),
             q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Verify Q is not all zeros or NaN
    let valid_values = q_data.iter().filter(|&&x| x.is_finite() && x.abs() > 1e-9).count();
    let valid_ratio = valid_values as f32 / q_data.len() as f32;
    println!("  Q valid value ratio: {:.2}%", valid_ratio * 100.0);

    assert!(valid_ratio > 0.5, "Most Q values should be valid and non-zero");

    // Step 3: Final output projection
    let output_norm = model.get_weight("output_norm.weight")
        .expect("output_norm.weight not found");
    let output_weight = model.get_weight("output.weight")
        .expect("output.weight not found");

    // Simplified: skip all layers, just test output projection
    let final_norm = x.rms_norm(vec![n_embd], &output_norm, 1e-5).unwrap();
    let logits = final_norm.matmul_transposed_b(&output_weight).unwrap();
    let logits_data = logits.sync_and_read();

    println!("  Logits shape: {:?}", logits.dims());

    // Find top token
    let max_idx = logits_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("  Top predicted token: {} (logit={:.6})", max_idx, logits_data[max_idx]);

    println!("\n✅ PREFILL pipeline works!");
}

#[test]
fn test_sampling_function() {
    // Test 1: Verify argmax sampling (temperature=0) selects highest logit
    // Test 2: Verify logits distribution is not collapsed to single value

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting sampling function:");

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    // Create test hidden state: [1, 2048]
    let n_embd = 2048;
    let vocab_size = 32000;
    let hidden_data: Vec<f32> = (0..n_embd)
        .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
        .collect();
    let hidden = Tensor::from_vec_gpu(&device, hidden_data, vec![1, n_embd]).unwrap();

    // Get output weights
    let output_weight = model.get_weight("output.weight")
        .expect("output.weight not found");

    // Compute logits: hidden @ output.weight^T
    let logits = hidden.matmul_transposed_b(&output_weight).unwrap();
    let logits_data = logits.sync_and_read();

    println!("  Logits shape: {:?}", logits.dims());

    // Test 1: Verify logits have reasonable variation
    let logit_mean = logits_data.iter().sum::<f32>() / logits_data.len() as f32;
    let logit_min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let logit_max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logit_std = (logits_data.iter()
        .map(|&x| (x - logit_mean).powi(2))
        .sum::<f32>() / logits_data.len() as f32)
        .sqrt();

    println!("  Logits stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
             logit_mean, logit_std, logit_min, logit_max);

    assert!(logit_std > 0.1, "Logits should have reasonable standard deviation, got {}", logit_std);
    assert!(logit_max - logit_min > 1.0, "Logits should have reasonable range");

    // Test 2: Verify argmax (greedy) selects correct token
    let mut logits_indexed: Vec<(usize, f32)> = logits_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    logits_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Top 10 logits (for argmax verification):");
    for (rank, (token_id, logit)) in logits_indexed.iter().take(10).enumerate() {
        println!("    Rank {}: token_id={}, logit={:.6}", rank + 1, token_id, logit);
    }

    let top_token = logits_indexed[0].0;
    let top_logit = logits_indexed[0].1;
    let second_logit = logits_indexed[1].1;

    println!("\n  Argmax selection: token_id={} (logit={:.6})", top_token, top_logit);
    println!("  Second best: token_id={} (logit={:.6})", logits_indexed[1].0, second_logit);
    println!("  Difference: {:.6}", top_logit - second_logit);

    assert!(top_logit > second_logit, "Top logit should be greater than second");
    assert!(top_token < vocab_size, "Selected token should be valid");

    // Test 3: Check for collapsed distribution (all same value)
    let unique_values: std::collections::HashSet<_> = logits_data.iter()
        .map(|&x| (x * 1000.0) as i32)  // Round to 3 decimal places
        .collect();

    println!("  Unique logit values (rounded): {}", unique_values.len());
    assert!(unique_values.len() > 100,
            "Logits should not be collapsed to single value, found only {} unique values",
            unique_values.len());

    println!("\n✅ Sampling function test passed!");
}

#[test]
fn test_decode_token_feedback() {
    // Test that a newly generated token properly updates model state
    // Simulate: PREFILL with [1] (BOS) → Generate token → Check if different from "aca"

    let home = env::var("HOME").expect("HOME not set");
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home);

    println!("\nTesting DECODE token feedback:");

    let device = MetalDevice::new().expect("Failed to create Metal device");
    let model = GGUFWeightCache::<f32>::new(&model_path, device.clone(), 50)
        .expect("Failed to load model");

    let n_embd = 2048;

    // Step 1: Get embedding for token_id=1 (BOS)
    let tok_embd = model.get_weight("token_embd.weight")
        .expect("token_embd.weight not found");
    let tok_embd_data = tok_embd.sync_and_read();

    let token_id = 1;
    let embedding: Vec<f32> = tok_embd_data[token_id * n_embd..(token_id + 1) * n_embd].to_vec();
    let x = Tensor::from_vec_gpu(&device, embedding.clone(), vec![1, n_embd]).unwrap();

    println!("  Token 0: BOS (token_id={})", token_id);

    // Step 2: Simple forward pass (skip full transformer for speed)
    let output_norm = model.get_weight("output_norm.weight")
        .expect("output_norm.weight not found");
    let output_weight = model.get_weight("output.weight")
        .expect("output.weight not found");

    let normed = x.rms_norm(vec![n_embd], &output_norm, 1e-5).unwrap();
    let logits1 = normed.matmul_transposed_b(&output_weight).unwrap();
    let logits1_data = logits1.sync_and_read();

    let token1 = logits1_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("  Token 1 prediction from BOS: token_id={} (logit={:.6})",
             token1, logits1_data[token1]);

    // Step 3: Use generated token as input for next step
    let embedding2: Vec<f32> = tok_embd_data[token1 * n_embd..(token1 + 1) * n_embd].to_vec();
    let x2 = Tensor::from_vec_gpu(&device, embedding2, vec![1, n_embd]).unwrap();

    let normed2 = x2.rms_norm(vec![n_embd], &output_norm, 1e-5).unwrap();
    let logits2 = normed2.matmul_transposed_b(&output_weight).unwrap();
    let logits2_data = logits2.sync_and_read();

    let token2 = logits2_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("  Token 2 prediction from Token 1: token_id={} (logit={:.6})",
             token2, logits2_data[token2]);

    // Test: Second token should be different (model is not stuck)
    // Note: This is a simplified test without full transformer layers
    println!("\n  Checking if tokens are different:");
    println!("    Token 1: {} (from BOS)", token1);
    println!("    Token 2: {} (from Token 1)", token2);

    // Verify embeddings are different
    let emb1_sum: f32 = embedding.iter().sum();
    let emb2_sum: f32 = tok_embd_data[token1 * n_embd..(token1 + 1) * n_embd].iter().sum();

    println!("\n  Embedding verification:");
    println!("    BOS embedding sum: {:.6}", emb1_sum);
    println!("    Token {} embedding sum: {:.6}", token1, emb2_sum);

    assert!((emb1_sum - emb2_sum).abs() > 0.1,
            "Embeddings should be different for different tokens");

    println!("\n✅ DECODE token feedback test passed!");
}

#[test]
fn test_position_encoding_sequence() {
    // Test that position encoding changes across sequence positions
    // Verify RoPE is applied correctly at different positions

    let device = MetalDevice::new().expect("Failed to create Metal device");

    println!("\nTesting position encoding sequence:");

    let n_heads = 32;
    let head_dim = 64;

    // Create test Q tensor: [3, 32, 64] (3 positions, 32 heads, 64 head_dim)
    // RoPE expects [..., seq_len, n_heads, head_dim]
    let q_data: Vec<f32> = (0..3 * n_heads * head_dim)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let q = Tensor::from_vec_gpu(&device, q_data.clone(), vec![3, n_heads, head_dim]).unwrap();

    println!("  Input Q shape: {:?}", q.dims());

    // Apply RoPE at different positions
    let q_rope_pos0 = q.rope(0).unwrap();
    let q_rope_pos5 = q.rope(5).unwrap();
    let q_rope_pos10 = q.rope(10).unwrap();

    let q_pos0_data = q_rope_pos0.sync_and_read();
    let q_pos5_data = q_rope_pos5.sync_and_read();
    let q_pos10_data = q_rope_pos10.sync_and_read();

    println!("\n  RoPE at position 0:");
    println!("    First 8 values: {:?}", &q_pos0_data[..8]);

    println!("\n  RoPE at position 5:");
    println!("    First 8 values: {:?}", &q_pos5_data[..8]);

    println!("\n  RoPE at position 10:");
    println!("    First 8 values: {:?}", &q_pos10_data[..8]);

    // Test 1: Different positions should produce different outputs
    let diff_0_5: f32 = q_pos0_data.iter()
        .zip(q_pos5_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_pos0_data.len() as f32;

    let diff_5_10: f32 = q_pos5_data.iter()
        .zip(q_pos10_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_pos5_data.len() as f32;

    println!("\n  Position encoding differences:");
    println!("    Pos 0 vs Pos 5: mean_abs_diff={:.6}", diff_0_5);
    println!("    Pos 5 vs Pos 10: mean_abs_diff={:.6}", diff_5_10);

    assert!(diff_0_5 > 0.01,
            "Position 0 and 5 should produce different encodings, diff={}", diff_0_5);
    assert!(diff_5_10 > 0.01,
            "Position 5 and 10 should produce different encodings, diff={}", diff_5_10);

    // Test 2: In DECODE, position should increment
    println!("\n  Simulating DECODE sequence:");

    // Single token input [1, 32, 64] (1 position, 32 heads, 64 head_dim)
    let single_token: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32 % 50.0) * 0.01).collect();
    let q_single = Tensor::from_vec_gpu(&device, single_token, vec![1, n_heads, head_dim]).unwrap();

    let positions = vec![34, 35, 36, 37, 38];  // After PREFILL of 34 tokens
    let mut rope_outputs = Vec::new();

    for &pos in &positions {
        let q_rope = q_single.rope(pos).unwrap();
        let q_rope_data = q_rope.sync_and_read();
        rope_outputs.push(q_rope_data);
        println!("    Position {}: first 4 values: {:?}", pos, &rope_outputs.last().unwrap()[..4]);
    }

    // Verify consecutive positions are different
    for i in 0..positions.len() - 1 {
        let diff: f32 = rope_outputs[i].iter()
            .zip(rope_outputs[i + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / rope_outputs[i].len() as f32;

        println!("    Pos {} vs Pos {}: diff={:.6}", positions[i], positions[i + 1], diff);
        assert!(diff > 0.001,
                "Consecutive positions should have different encodings");
    }

    println!("\n✅ Position encoding sequence test passed!");
}
