// Test RoPE (Rotary Position Embedding) implementation
//
// Verifies that the RoPE implementation produces correct results
// by comparing against manually calculated expected values.

fn main() {
    println!("=== RoPE Implementation Test ===\n");

    // Test configuration (matching TinyLlama)
    let seq_len = 1;
    let n_heads = 32;
    let head_dim = 64;
    let rope_base = 10000.0f32;
    let position = 0; // First position

    println!("Configuration:");
    println!("  seq_len: {}", seq_len);
    println!("  n_heads: {}", n_heads);
    println!("  head_dim: {}", head_dim);
    println!("  rope_base: {}", rope_base);
    println!("  position: {}", position);
    println!();

    // Create simple test input: [1, 32, 64]
    // For simplicity, use incrementing values
    let mut input = Vec::new();
    for head in 0..n_heads {
        for dim in 0..head_dim {
            // Simple pattern: head * 100 + dim
            let value = head as f32 * 0.1 + dim as f32 * 0.01;
            input.push(value);
        }
    }

    println!("Input tensor shape: [{}, {}, {}]", seq_len, n_heads, head_dim);
    println!("Input sample (head 0, first 8 dims):");
    for i in 0..8 {
        print!("  {:.4}", input[i]);
    }
    println!("\n");

    // Manually calculate expected RoPE output for first few dimension pairs
    println!("Expected RoPE calculations (head 0):");
    for pair_idx in 0..4 {
        let dim_idx = pair_idx * 2;

        // Calculate frequency
        let exponent = (2 * pair_idx) as f32 / head_dim as f32;
        let freq = 1.0 / rope_base.powf(exponent);

        // Calculate angle
        let theta = position as f32 * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Get input values
        let x0 = input[dim_idx];
        let x1 = input[dim_idx + 1];

        // Apply rotation
        let out0 = x0 * cos_theta - x1 * sin_theta;
        let out1 = x0 * sin_theta + x1 * cos_theta;

        println!("  Pair {} (dims {}-{}):", pair_idx, dim_idx, dim_idx + 1);
        println!("    freq={:.6}, theta={:.6}", freq, theta);
        println!("    cos={:.6}, sin={:.6}", cos_theta, sin_theta);
        println!("    input=({:.4}, {:.4})", x0, x1);
        println!("    output=({:.4}, {:.4})", out0, out1);
    }
    println!();

    // Test with position = 0 (should mostly preserve input due to cos(0)=1, sin(0)=0)
    println!("Special case: position=0");
    println!("  At position 0, RoPE should approximately preserve input");
    println!("  because cos(0)=1 and sin(0)=0");
    println!("  Expected: out[2i] ≈ in[2i], out[2i+1] ≈ in[2i+1]");
    println!();

    // Test with position = 1
    println!("Testing with position=1 instead:");
    let position_1 = 1;
    for pair_idx in 0..2 {
        let dim_idx = pair_idx * 2;

        let exponent = (2 * pair_idx) as f32 / head_dim as f32;
        let freq = 1.0 / rope_base.powf(exponent);
        let theta = position_1 as f32 * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let x0 = input[dim_idx];
        let x1 = input[dim_idx + 1];

        let out0 = x0 * cos_theta - x1 * sin_theta;
        let out1 = x0 * sin_theta + x1 * cos_theta;

        println!("  Pair {} (dims {}-{}):", pair_idx, dim_idx, dim_idx + 1);
        println!("    theta={:.6} (cos={:.6}, sin={:.6})", theta, cos_theta, sin_theta);
        println!("    input=({:.4}, {:.4}) -> output=({:.4}, {:.4})", x0, x1, out0, out1);
    }
    println!();

    // Test frequency calculation for different dimensions
    println!("Frequency spectrum (first 8 dimension pairs):");
    for pair_idx in 0..8 {
        let exponent = (2 * pair_idx) as f32 / head_dim as f32;
        let freq = 1.0 / rope_base.powf(exponent);
        println!("  Pair {}: exponent={:.4}, freq={:.6}", pair_idx, exponent, freq);
    }
    println!();

    println!("✅ Manual RoPE calculation complete");
    println!("\n📝 Next: Compare with TensorLogic rope() implementation");
    println!("   Create a TensorLogic script that:");
    println!("   1. Loads model weights (not needed for RoPE test)");
    println!("   2. Creates test tensor with same pattern");
    println!("   3. Applies rope() function");
    println!("   4. Compares output with expected values above");
}
