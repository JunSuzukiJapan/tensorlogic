/// Debug Layer 0 inference step by step
///
/// Execute Layer 0 with token "Hello" (15043) and print intermediate values

use std::path::PathBuf;
use tensorlogic::device::MetalDevice;
use tensorlogic::model::Model;
use tensorlogic::tensor::Tensor;

fn print_first_values(name: &str, tensor: &Tensor, count: usize) {
    let data = tensor.to_vec();
    let values: Vec<f32> = data.iter().take(count).map(|x| x.to_f32()).collect();
    println!("  {} (first {}): {:?}", name, count, values);
    println!("  {} shape: {:?}", name, tensor.dims());
}

fn main() {
    println!("=== Layer 0 Step-by-Step Debug ===\n");

    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Load model
    let home = std::env::var("HOME").expect("HOME not set");
    let model_path = PathBuf::from(&home).join(".llm/models/tinyllama-1.1b-chat-q4_0.gguf");

    println!("Loading model from: {:?}\n", model_path);
    let model = Model::load(&model_path, &device)
        .expect("Failed to load model");

    // Input: "Hello" = token 15043
    let token_id: usize = 15043;
    println!("Input token: {} (\"Hello\")\n", token_id);

    // Step 1: Embedding lookup
    println!("Step 1: Embedding Lookup");
    let token_embd = model.get_tensor("token_embd.weight")
        .expect("Failed to get token_embd.weight");

    let input_ids_f32 = vec![token_id as f32];
    let input_ids_f16: Vec<half::f16> = input_ids_f32.iter()
        .map(|&x| half::f16::from_f32(x))
        .collect();

    let input_ids_tensor = Tensor::from_vec_metal(&device, input_ids_f16, vec![1])
        .expect("Failed to create input tensor");

    let h = token_embd.embedding(&input_ids_tensor)
        .expect("Failed to perform embedding");

    print_first_values("Embedding output", &h, 10);
    println!();

    // Step 2: Attention Norm
    println!("Step 2: Attention Norm (RMS Norm)");
    let attn_norm = model.get_tensor("blk.0.attn_norm.weight")
        .expect("Failed to get attn_norm");

    let h_normed = h.rms_norm(vec![2048], &attn_norm, 1e-6)
        .expect("Failed to apply RMS norm");

    print_first_values("After RMS norm", &h_normed, 10);
    println!();

    // Step 3: Q/K/V Projections
    println!("Step 3: Q/K/V Projections");
    let W_q = model.get_tensor("blk.0.attn_q.weight")
        .expect("Failed to get W_q");
    let W_k = model.get_tensor("blk.0.attn_k.weight")
        .expect("Failed to get W_k");
    let W_v = model.get_tensor("blk.0.attn_v.weight")
        .expect("Failed to get W_v");

    println!("  W_q shape: {:?}", W_q.dims());
    println!("  W_k shape: {:?}", W_k.dims());
    println!("  W_v shape: {:?}", W_v.dims());

    // Linear function performs: x @ W.T
    // W_q is [2048, 2048], so W_q.T is [2048, 2048]
    // h_normed is [1, 2048]
    // Result: [1, 2048]

    let W_q_t = W_q.transpose().expect("Failed to transpose W_q");
    let Q = h_normed.matmul(&W_q_t).expect("Failed to compute Q");
    print_first_values("Q", &Q, 10);

    let W_k_t = W_k.transpose().expect("Failed to transpose W_k");
    let K = h_normed.matmul(&W_k_t).expect("Failed to compute K");
    print_first_values("K", &K, 10);

    let W_v_t = W_v.transpose().expect("Failed to transpose W_v");
    let V = h_normed.matmul(&W_v_t).expect("Failed to compute V");
    print_first_values("V", &V, 10);
    println!();

    // Step 4: Reshape for heads
    println!("Step 4: Reshape for Multi-Head Attention");
    let Q_heads = Q.reshape(vec![1, 32, 64])
        .expect("Failed to reshape Q");
    let K_heads = K.reshape(vec![1, 4, 64])
        .expect("Failed to reshape K");
    let V_heads = V.reshape(vec![1, 4, 64])
        .expect("Failed to reshape V");

    println!("  Q_heads shape: {:?}", Q_heads.dims());
    println!("  K_heads shape: {:?}", K_heads.dims());
    println!("  V_heads shape: {:?}", V_heads.dims());
    println!();

    // Step 5: Apply RoPE
    println!("Step 5: Apply RoPE (Rotary Position Embedding)");
    let Q_rope = Q_heads.rope()
        .expect("Failed to apply RoPE to Q");
    let K_rope = K_heads.rope()
        .expect("Failed to apply RoPE to K");

    print_first_values("Q after RoPE", &Q_rope, 10);
    print_first_values("K after RoPE", &K_rope, 10);
    println!();

    // Step 6: Expand K and V for GQA
    println!("Step 6: Expand K/V for Grouped Query Attention");
    // K_heads: [1, 4, 64] → [1, 32, 64]
    let K_exp = K_rope.reshape(vec![1, 4, 1, 64])
        .expect("Failed to reshape K for broadcast");
    let K_target_shape = vec![1, 4, 8, 64];
    let K_broadcast = K_exp.broadcast_to(&K_target_shape)
        .expect("Failed to broadcast K");
    let K_expanded = K_broadcast.reshape(vec![1, 32, 64])
        .expect("Failed to reshape K_expanded");

    let V_exp = V_heads.reshape(vec![1, 4, 1, 64])
        .expect("Failed to reshape V for broadcast");
    let V_target_shape = vec![1, 4, 8, 64];
    let V_broadcast = V_exp.broadcast_to(&V_target_shape)
        .expect("Failed to broadcast V");
    let V_expanded = V_broadcast.reshape(vec![1, 32, 64])
        .expect("Failed to reshape V_expanded");

    println!("  K_expanded shape: {:?}", K_expanded.dims());
    println!("  V_expanded shape: {:?}", V_expanded.dims());
    print_first_values("K_expanded", &K_expanded, 10);
    print_first_values("V_expanded", &V_expanded, 10);
    println!();

    // Step 7: Attention scores
    println!("Step 7: Compute Attention Scores");
    // Q: [1, 32, 64], K: [1, 32, 64]
    // scores = Q @ K.T = [1, 32, 32]
    let K_t = K_expanded.transpose().expect("Failed to transpose K");
    println!("  K_t shape: {:?}", K_t.dims());

    // Need einsum("ihd,jhd->ihj") which is Q @ K.T over the last dimension
    // For single token, this simplifies
    let scores_raw = Q_rope.matmul(&K_t)
        .expect("Failed to compute attention scores");

    println!("  scores_raw shape: {:?}", scores_raw.dims());
    print_first_values("scores_raw", &scores_raw, 10);

    // Scale by 1/sqrt(64) = 1/8 = 0.125
    let scale_factor = half::f16::from_f32(0.125);
    let scale_vec = vec![scale_factor];
    let scale_tensor = Tensor::from_vec_metal(&device, scale_vec, vec![1])
        .expect("Failed to create scale tensor");

    let scores_scaled = scores_raw.multiply(&scale_tensor)
        .expect("Failed to scale scores");

    print_first_values("scores_scaled", &scores_scaled, 10);
    println!();

    // Step 8: Softmax
    println!("Step 8: Apply Softmax");
    let attn = scores_scaled.softmax()
        .expect("Failed to compute softmax");

    print_first_values("attention weights", &attn, 10);
    println!();

    // Step 9: Weighted sum with V
    println!("Step 9: Weighted Sum with V");
    // attn: [1, 32, 32], V: [1, 32, 64]
    // output = attn @ V = [1, 32, 64]
    let V_t = V_expanded.transpose().expect("Failed to transpose V");
    let attn_out = attn.matmul(&V_t)
        .expect("Failed to compute attention output");

    print_first_values("attention output", &attn_out, 10);
    println!("  attention output shape: {:?}", attn_out.dims());
    println!();

    // Step 10: Reshape and output projection
    println!("Step 10: Reshape and Output Projection");
    let attn_reshaped = attn_out.reshape(vec![1, 2048])
        .expect("Failed to reshape attention output");

    let W_o = model.get_tensor("blk.0.attn_output.weight")
        .expect("Failed to get W_o");

    let W_o_t = W_o.transpose().expect("Failed to transpose W_o");
    let attn_final = attn_reshaped.matmul(&W_o_t)
        .expect("Failed to apply output projection");

    print_first_values("Layer 0 attention final", &attn_final, 10);
    println!();

    // Step 11: Residual connection
    println!("Step 11: Residual Connection");
    let h_after_attn = h.add(&attn_final)
        .expect("Failed to add residual");

    print_first_values("After residual (h + attn)", &h_after_attn, 10);
    println!();

    println!("=== Layer 0 forward pass complete ===");
    println!("\nNext: Run FFN and compare final Layer 0 output");
}
