/// Verify RoPE, GQA, and Attention with small, hand-calculable inputs
/// This allows us to manually verify each step of the computation

fn main() {
    println!("=== Small Case Verification ===\n");

    verify_rope_small();
    println!();
    verify_gqa_small();
    println!();
    verify_attention_small();
}

fn verify_rope_small() {
    println!("--- RoPE Verification (Small Case) ---");

    // Input: 2 heads, 4 dimensions each
    // Head 0: [1.0, 2.0, 3.0, 4.0]
    // Head 1: [5.0, 6.0, 7.0, 8.0]

    let input = vec![
        1.0f32, 2.0, 3.0, 4.0,  // Head 0
        5.0, 6.0, 7.0, 8.0,     // Head 1
    ];

    let n_heads = 2;
    let head_dim = 4;
    let rope_base = 10000.0f32;
    let position = 0;

    println!("Input shape: [{}, {}]", n_heads, head_dim);
    println!("Input values:");
    println!("  Head 0: {:?}", &input[0..4]);
    println!("  Head 1: {:?}", &input[4..8]);
    println!();

    // Apply RoPE
    let mut output = vec![0.0f32; n_heads * head_dim];

    for head in 0..n_heads {
        for pair_idx in 0..(head_dim / 2) {
            let idx0 = head * head_dim + pair_idx * 2;
            let idx1 = idx0 + 1;

            let x0 = input[idx0];
            let x1 = input[idx1];

            // Calculate frequency
            let exponent = (2 * pair_idx) as f32 / head_dim as f32;
            let freq = 1.0 / rope_base.powf(exponent);

            // Calculate angle
            let theta = position as f32 * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Apply rotation
            let out0 = x0 * cos_theta - x1 * sin_theta;
            let out1 = x0 * sin_theta + x1 * cos_theta;

            output[idx0] = out0;
            output[idx1] = out1;

            println!("Head {}, Pair {} (dims {}-{}):", head, pair_idx, idx0 % head_dim, idx1 % head_dim);
            println!("  freq={:.6}, theta={:.6}", freq, theta);
            println!("  cos={:.6}, sin={:.6}", cos_theta, sin_theta);
            println!("  input=({:.2}, {:.2})", x0, x1);
            println!("  output=({:.6}, {:.6})", out0, out1);
        }
    }

    println!();
    println!("Expected at position=0:");
    println!("  cos(0)=1, sin(0)=0");
    println!("  Output should match input exactly");
    println!();
    println!("Output values:");
    println!("  Head 0: {:?}", &output[0..4]);
    println!("  Head 1: {:?}", &output[4..8]);

    // Verify
    let mut all_match = true;
    for i in 0..input.len() {
        if (input[i] - output[i]).abs() > 1e-6 {
            all_match = false;
            println!("  ❌ Mismatch at index {}: input={:.6}, output={:.6}",
                     i, input[i], output[i]);
        }
    }

    if all_match {
        println!("✅ RoPE at position=0 preserves input correctly");
    } else {
        println!("❌ RoPE has unexpected differences");
    }
}

fn verify_gqa_small() {
    println!("--- GQA Expansion Verification (Small Case) ---");

    // Input: 2 KV heads, 4 dimensions each
    // KV head 0: [1.0, 2.0, 3.0, 4.0]
    // KV head 1: [5.0, 6.0, 7.0, 8.0]
    //
    // Expand to 4 Q heads (expansion factor = 2)
    // Expected:
    // Q head 0: [1.0, 2.0, 3.0, 4.0]  (from KV 0)
    // Q head 1: [1.0, 2.0, 3.0, 4.0]  (from KV 0)
    // Q head 2: [5.0, 6.0, 7.0, 8.0]  (from KV 1)
    // Q head 3: [5.0, 6.0, 7.0, 8.0]  (from KV 1)

    let kv_input = vec![
        1.0f32, 2.0, 3.0, 4.0,  // KV head 0
        5.0, 6.0, 7.0, 8.0,     // KV head 1
    ];

    let n_kv_heads = 2;
    let n_q_heads = 4;
    let head_dim = 4;
    let expansion_factor = n_q_heads / n_kv_heads;

    println!("Input KV heads: {}", n_kv_heads);
    println!("Output Q heads: {}", n_q_heads);
    println!("Expansion factor: {}", expansion_factor);
    println!();

    println!("Input values:");
    println!("  KV head 0: {:?}", &kv_input[0..4]);
    println!("  KV head 1: {:?}", &kv_input[4..8]);
    println!();

    // Expand
    let mut q_output = vec![0.0f32; n_q_heads * head_dim];

    for kv_head in 0..n_kv_heads {
        for replica in 0..expansion_factor {
            let q_head = kv_head * expansion_factor + replica;

            for dim in 0..head_dim {
                let src_idx = kv_head * head_dim + dim;
                let dst_idx = q_head * head_dim + dim;
                q_output[dst_idx] = kv_input[src_idx];
            }

            println!("KV head {} → Q head {}", kv_head, q_head);
        }
    }

    println!();
    println!("Output values:");
    for q_head in 0..n_q_heads {
        let start = q_head * head_dim;
        let end = start + head_dim;
        println!("  Q head {}: {:?}", q_head, &q_output[start..end]);
    }

    // Verify
    println!();
    println!("Verification:");
    for q_head in 0..n_q_heads {
        let kv_head = q_head / expansion_factor;
        let match_expected = (0..head_dim).all(|dim| {
            let q_idx = q_head * head_dim + dim;
            let kv_idx = kv_head * head_dim + dim;
            q_output[q_idx] == kv_input[kv_idx]
        });

        if match_expected {
            println!("  ✅ Q head {} correctly replicates KV head {}", q_head, kv_head);
        } else {
            println!("  ❌ Q head {} does not match KV head {}", q_head, kv_head);
        }
    }
}

fn verify_attention_small() {
    println!("--- Attention Verification (Small Case) ---");

    // Simplified: 1 sequence, 2 heads, 4 dimensions
    // Q: 2 heads × 4 dims
    // K: 2 heads × 4 dims
    // V: 2 heads × 4 dims

    let q = vec![
        1.0f32, 0.0, 0.0, 0.0,  // Head 0
        0.0, 1.0, 0.0, 0.0,     // Head 1
    ];

    let k = vec![
        1.0f32, 0.0, 0.0, 0.0,  // Head 0 (same as Q[0], should give high score)
        0.0, 0.0, 1.0, 0.0,     // Head 1 (orthogonal to Q[1])
    ];

    let v = vec![
        2.0f32, 2.0, 2.0, 2.0,  // Head 0
        3.0, 3.0, 3.0, 3.0,     // Head 1
    ];

    let n_heads = 2;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();  // 1/2 = 0.5

    println!("Q (2 heads × 4 dims):");
    println!("  Head 0: {:?}", &q[0..4]);
    println!("  Head 1: {:?}", &q[4..8]);
    println!();

    println!("K (2 heads × 4 dims):");
    println!("  Head 0: {:?}", &k[0..4]);
    println!("  Head 1: {:?}", &k[4..8]);
    println!();

    println!("V (2 heads × 4 dims):");
    println!("  Head 0: {:?}", &v[0..4]);
    println!("  Head 1: {:?}", &v[4..8]);
    println!();

    println!("Scale factor: 1/sqrt({}) = {:.4}", head_dim, scale);
    println!();

    // Compute attention for each head
    let mut output = vec![0.0f32; n_heads * head_dim];

    for head in 0..n_heads {
        let q_start = head * head_dim;
        let k_start = head * head_dim;
        let v_start = head * head_dim;

        // Compute Q·K (dot product)
        let mut score = 0.0f32;
        for i in 0..head_dim {
            score += q[q_start + i] * k[k_start + i];
        }

        // Scale
        let scaled_score = score * scale;

        // Softmax (with single value, it's just exp(x) / exp(x) = 1.0)
        let attn_weight = 1.0f32;  // In this simple case

        // Weighted sum of V
        for i in 0..head_dim {
            output[head * head_dim + i] = v[v_start + i] * attn_weight;
        }

        println!("Head {}:", head);
        println!("  Q·K = {:.4}", score);
        println!("  Scaled = {:.4}", scaled_score);
        println!("  Attention weight = {:.4}", attn_weight);
        println!("  Output = {:?}", &output[head * head_dim..(head + 1) * head_dim]);
    }

    println!();
    println!("Expected:");
    println!("  Head 0: Q·K = 1.0, scaled = 0.5, output = V[0]");
    println!("  Head 1: Q·K = 0.0, scaled = 0.0, output = V[1]");

    // Note: In real attention with multiple sequence positions,
    // softmax would normalize across all positions.
    // This is a simplified single-position case.
}
