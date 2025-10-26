// Test GQA (Grouped Query Attention) expansion
//
// Verifies that K/V head expansion from 4 to 32 heads is correct
// Each of the 4 K/V heads should be replicated 8 times to match 32 Q heads

fn main() {
    println!("=== GQA Expansion Test ===\n");

    // TinyLlama configuration
    let n_q_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;
    let expansion_factor = n_q_heads / n_kv_heads;  // 8

    println!("Configuration:");
    println!("  Q heads: {}", n_q_heads);
    println!("  KV heads: {}", n_kv_heads);
    println!("  Head dimension: {}", head_dim);
    println!("  Expansion factor: {} (each KV head → {} Q heads)", expansion_factor, expansion_factor);
    println!();

    // Create test K tensor: [1, 4, 64]
    // Use distinct values for each KV head to verify correct expansion
    let seq_len = 1;
    let mut k_tensor = vec![0.0f32; seq_len * n_kv_heads * head_dim];

    for kv_head in 0..n_kv_heads {
        for dim in 0..head_dim {
            let idx = kv_head * head_dim + dim;
            // Each KV head has a distinct pattern: kv_head * 100 + dim
            k_tensor[idx] = (kv_head as f32 * 100.0) + (dim as f32);
        }
    }

    println!("Input K tensor [1, 4, 64]:");
    println!("  KV head 0, first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_tensor[0], k_tensor[1], k_tensor[2], k_tensor[3]);
    println!("  KV head 1, first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_tensor[64], k_tensor[65], k_tensor[66], k_tensor[67]);
    println!("  KV head 2, first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_tensor[128], k_tensor[129], k_tensor[130], k_tensor[131]);
    println!("  KV head 3, first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_tensor[192], k_tensor[193], k_tensor[194], k_tensor[195]);
    println!();

    // Expected expansion:
    // KV head 0 → Q heads 0-7
    // KV head 1 → Q heads 8-15
    // KV head 2 → Q heads 16-23
    // KV head 3 → Q heads 24-31

    println!("Expected expansion pattern:");
    println!("  KV head 0 (values 0-63)    → Q heads 0-7");
    println!("  KV head 1 (values 100-163) → Q heads 8-15");
    println!("  KV head 2 (values 200-263) → Q heads 16-23");
    println!("  KV head 3 (values 300-363) → Q heads 24-31");
    println!();

    // Simulate TensorLogic's expansion approach:
    // Step 1: reshape [1, 4, 64] -> [1, 4, 1, 64]
    // Step 2: broadcast [1, 4, 1, 64] -> [1, 4, 8, 64]
    // Step 3: reshape [1, 4, 8, 64] -> [1, 32, 64]

    let mut k_expanded = vec![0.0f32; seq_len * n_q_heads * head_dim];

    for kv_head in 0..n_kv_heads {
        for replica in 0..expansion_factor {
            let q_head = kv_head * expansion_factor + replica;
            for dim in 0..head_dim {
                let src_idx = kv_head * head_dim + dim;
                let dst_idx = q_head * head_dim + dim;
                k_expanded[dst_idx] = k_tensor[src_idx];
            }
        }
    }

    println!("Expanded K tensor [1, 32, 64]:");
    println!("  Q head 0 (from KV 0), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[0], k_expanded[1], k_expanded[2], k_expanded[3]);
    println!("  Q head 7 (from KV 0), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[7*64], k_expanded[7*64+1], k_expanded[7*64+2], k_expanded[7*64+3]);
    println!("  Q head 8 (from KV 1), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[8*64], k_expanded[8*64+1], k_expanded[8*64+2], k_expanded[8*64+3]);
    println!("  Q head 15 (from KV 1), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[15*64], k_expanded[15*64+1], k_expanded[15*64+2], k_expanded[15*64+3]);
    println!("  Q head 24 (from KV 3), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[24*64], k_expanded[24*64+1], k_expanded[24*64+2], k_expanded[24*64+3]);
    println!("  Q head 31 (from KV 3), first 4 dims: {:.1} {:.1} {:.1} {:.1}",
        k_expanded[31*64], k_expanded[31*64+1], k_expanded[31*64+2], k_expanded[31*64+3]);
    println!();

    // Verify correctness
    println!("Verification:");
    let mut correct = true;
    for q_head in 0..n_q_heads {
        let kv_head = q_head / expansion_factor;
        for dim in 0..head_dim {
            let q_idx = q_head * head_dim + dim;
            let kv_idx = kv_head * head_dim + dim;
            let expected = k_tensor[kv_idx];
            let actual = k_expanded[q_idx];
            if (expected - actual).abs() > 0.001 {
                println!("  ❌ Mismatch at Q head {}, dim {}: expected {:.1}, got {:.1}",
                    q_head, dim, expected, actual);
                correct = false;
            }
        }
    }

    if correct {
        println!("  ✅ All values match expected pattern");
        println!("  ✅ GQA expansion is mathematically correct");
    } else {
        println!("  ❌ GQA expansion has errors");
    }
    println!();

    println!("📝 Key insight:");
    println!("  Each KV head must be replicated exactly {} times", expansion_factor);
    println!("  The expansion maintains the grouping structure:");
    for kv in 0..n_kv_heads {
        let start_q = kv * expansion_factor;
        let end_q = start_q + expansion_factor - 1;
        println!("    KV head {} → Q heads {}-{}", kv, start_q, end_q);
    }
}
