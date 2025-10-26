// TinyLlama Chat Demo with Residual Scaling Fix
// Fixes magnitude drift issue by scaling residual connections
//
// Usage:
//   Set residual_scale = 0.150756 (recommended: 1/sqrt(2*num_layers))
//   Set residual_scale = 1.0 to disable scaling (original behavior)

// SiLU activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
}

// SwiGLU FFN
fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    result := linear(intermediate, W_down)
}

// Grouped Query Attention (GQA) - Dynamic seq_len version
fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    // Constants
    let num_q_heads = 32
    let num_kv_heads = 4
    let head_dim = 64
    let group_size = 8

    // Dynamic seq_len calculation from Q shape
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    // Create shape arrays as tensors
    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    // Reshape to heads
    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    // Apply RoPE to Q and K (NOT V)
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA: Expand K, V from 4 to 32 heads
    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)
    let attn_weights = softmax(scaled_scores, 2)

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Output projection
    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    result := linear(attn_reshaped, W_o)
}

// Single Transformer Layer WITH RESIDUAL SCALING
// residual_scale: Scaling factor for residual connections
//   - Recommended: 1/sqrt(2*num_layers) to prevent magnitude drift
//   - Example: For 22 layers, use 0.150756 (= 1/sqrt(44))
//   - Set to 1.0 to disable scaling (original behavior)
fn transformer_layer_scaled(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?],
    residual_scale: float
) -> float16[?, ?] {
    // Pre-attention RMSNorm
    let x_norm1 = rms_norm(x, attn_norm)

    // QKV projections
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)

    // GQA Attention
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

    // Residual with scaling
    let attn_out_scaled = attn_out * residual_scale
    let x1 = x + attn_out_scaled

    // Pre-FFN RMSNorm
    let x_norm2 = rms_norm(x1, ffn_norm)

    // SwiGLU FFN
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

    // Final residual with scaling
    let ffn_out_scaled = ffn_out * residual_scale
    result := x1 + ffn_out_scaled
}

main {
    print("=== TinyLlama 22-Layer Chat with Residual Scaling ===")
    print("")

    // ========================================
    // RESIDUAL SCALING CONFIGURATION
    // ========================================
    // Scale factor: 1/sqrt(2*num_layers)
    // For 22 layers: 1/sqrt(44) = 0.150756
    //
    // To DISABLE scaling (test original behavior):
    //   Set residual_scale = 1.0
    // ========================================
    let residual_scale = 0.151
    print("Residual scale factor:", residual_scale)
    print("(Recommended: 1/sqrt(2*22) = 0.150756)")
    print("")

    // Load model and tokenizer
    print("[1/4] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model loaded: TinyLlama 1.1B Chat")
    print("")

    // Prepare embeddings and output weights
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load ALL 22 layers
    print("[2/4] Loading 22 Transformer layers...")

    // Layer 0
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    // Layer 1
    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")

    // Layer 2
    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")

    // Layer 3
    let W_q_3 = get_tensor(model, "blk.3.attn_q.weight")
    let W_k_3 = get_tensor(model, "blk.3.attn_k.weight")
    let W_v_3 = get_tensor(model, "blk.3.attn_v.weight")
    let W_o_3 = get_tensor(model, "blk.3.attn_output.weight")
    let attn_norm_3 = get_tensor(model, "blk.3.attn_norm.weight")
    let W_gate_3 = get_tensor(model, "blk.3.ffn_gate.weight")
    let W_up_3 = get_tensor(model, "blk.3.ffn_up.weight")
    let W_down_3 = get_tensor(model, "blk.3.ffn_down.weight")
    let ffn_norm_3 = get_tensor(model, "blk.3.ffn_norm.weight")

    // Layer 4
    let W_q_4 = get_tensor(model, "blk.4.attn_q.weight")
    let W_k_4 = get_tensor(model, "blk.4.attn_k.weight")
    let W_v_4 = get_tensor(model, "blk.4.attn_v.weight")
    let W_o_4 = get_tensor(model, "blk.4.attn_output.weight")
    let attn_norm_4 = get_tensor(model, "blk.4.attn_norm.weight")
    let W_gate_4 = get_tensor(model, "blk.4.ffn_gate.weight")
    let W_up_4 = get_tensor(model, "blk.4.ffn_up.weight")
    let W_down_4 = get_tensor(model, "blk.4.ffn_down.weight")
    let ffn_norm_4 = get_tensor(model, "blk.4.ffn_norm.weight")

    // Layer 5
    let W_q_5 = get_tensor(model, "blk.5.attn_q.weight")
    let W_k_5 = get_tensor(model, "blk.5.attn_k.weight")
    let W_v_5 = get_tensor(model, "blk.5.attn_v.weight")
    let W_o_5 = get_tensor(model, "blk.5.attn_output.weight")
    let attn_norm_5 = get_tensor(model, "blk.5.attn_norm.weight")
    let W_gate_5 = get_tensor(model, "blk.5.ffn_gate.weight")
    let W_up_5 = get_tensor(model, "blk.5.ffn_up.weight")
    let W_down_5 = get_tensor(model, "blk.5.ffn_down.weight")
    let ffn_norm_5 = get_tensor(model, "blk.5.ffn_norm.weight")

    // Layer 6
    let W_q_6 = get_tensor(model, "blk.6.attn_q.weight")
    let W_k_6 = get_tensor(model, "blk.6.attn_k.weight")
    let W_v_6 = get_tensor(model, "blk.6.attn_v.weight")
    let W_o_6 = get_tensor(model, "blk.6.attn_output.weight")
    let attn_norm_6 = get_tensor(model, "blk.6.attn_norm.weight")
    let W_gate_6 = get_tensor(model, "blk.6.ffn_gate.weight")
    let W_up_6 = get_tensor(model, "blk.6.ffn_up.weight")
    let W_down_6 = get_tensor(model, "blk.6.ffn_down.weight")
    let ffn_norm_6 = get_tensor(model, "blk.6.ffn_norm.weight")

    // Layer 7
    let W_q_7 = get_tensor(model, "blk.7.attn_q.weight")
    let W_k_7 = get_tensor(model, "blk.7.attn_k.weight")
    let W_v_7 = get_tensor(model, "blk.7.attn_v.weight")
    let W_o_7 = get_tensor(model, "blk.7.attn_output.weight")
    let attn_norm_7 = get_tensor(model, "blk.7.attn_norm.weight")
    let W_gate_7 = get_tensor(model, "blk.7.ffn_gate.weight")
    let W_up_7 = get_tensor(model, "blk.7.ffn_up.weight")
    let W_down_7 = get_tensor(model, "blk.7.ffn_down.weight")
    let ffn_norm_7 = get_tensor(model, "blk.7.ffn_norm.weight")

    // Layer 8
    let W_q_8 = get_tensor(model, "blk.8.attn_q.weight")
    let W_k_8 = get_tensor(model, "blk.8.attn_k.weight")
    let W_v_8 = get_tensor(model, "blk.8.attn_v.weight")
    let W_o_8 = get_tensor(model, "blk.8.attn_output.weight")
    let attn_norm_8 = get_tensor(model, "blk.8.attn_norm.weight")
    let W_gate_8 = get_tensor(model, "blk.8.ffn_gate.weight")
    let W_up_8 = get_tensor(model, "blk.8.ffn_up.weight")
    let W_down_8 = get_tensor(model, "blk.8.ffn_down.weight")
    let ffn_norm_8 = get_tensor(model, "blk.8.ffn_norm.weight")

    // Layer 9
    let W_q_9 = get_tensor(model, "blk.9.attn_q.weight")
    let W_k_9 = get_tensor(model, "blk.9.attn_k.weight")
    let W_v_9 = get_tensor(model, "blk.9.attn_v.weight")
    let W_o_9 = get_tensor(model, "blk.9.attn_output.weight")
    let attn_norm_9 = get_tensor(model, "blk.9.attn_norm.weight")
    let W_gate_9 = get_tensor(model, "blk.9.ffn_gate.weight")
    let W_up_9 = get_tensor(model, "blk.9.ffn_up.weight")
    let W_down_9 = get_tensor(model, "blk.9.ffn_down.weight")
    let ffn_norm_9 = get_tensor(model, "blk.9.ffn_norm.weight")

    // Layer 10
    let W_q_10 = get_tensor(model, "blk.10.attn_q.weight")
    let W_k_10 = get_tensor(model, "blk.10.attn_k.weight")
    let W_v_10 = get_tensor(model, "blk.10.attn_v.weight")
    let W_o_10 = get_tensor(model, "blk.10.attn_output.weight")
    let attn_norm_10 = get_tensor(model, "blk.10.attn_norm.weight")
    let W_gate_10 = get_tensor(model, "blk.10.ffn_gate.weight")
    let W_up_10 = get_tensor(model, "blk.10.ffn_up.weight")
    let W_down_10 = get_tensor(model, "blk.10.ffn_down.weight")
    let ffn_norm_10 = get_tensor(model, "blk.10.ffn_norm.weight")

    // Layer 11
    let W_q_11 = get_tensor(model, "blk.11.attn_q.weight")
    let W_k_11 = get_tensor(model, "blk.11.attn_k.weight")
    let W_v_11 = get_tensor(model, "blk.11.attn_v.weight")
    let W_o_11 = get_tensor(model, "blk.11.attn_output.weight")
    let attn_norm_11 = get_tensor(model, "blk.11.attn_norm.weight")
    let W_gate_11 = get_tensor(model, "blk.11.ffn_gate.weight")
    let W_up_11 = get_tensor(model, "blk.11.ffn_up.weight")
    let W_down_11 = get_tensor(model, "blk.11.ffn_down.weight")
    let ffn_norm_11 = get_tensor(model, "blk.11.ffn_norm.weight")

    // Layer 12
    let W_q_12 = get_tensor(model, "blk.12.attn_q.weight")
    let W_k_12 = get_tensor(model, "blk.12.attn_k.weight")
    let W_v_12 = get_tensor(model, "blk.12.attn_v.weight")
    let W_o_12 = get_tensor(model, "blk.12.attn_output.weight")
    let attn_norm_12 = get_tensor(model, "blk.12.attn_norm.weight")
    let W_gate_12 = get_tensor(model, "blk.12.ffn_gate.weight")
    let W_up_12 = get_tensor(model, "blk.12.ffn_up.weight")
    let W_down_12 = get_tensor(model, "blk.12.ffn_down.weight")
    let ffn_norm_12 = get_tensor(model, "blk.12.ffn_norm.weight")

    // Layer 13
    let W_q_13 = get_tensor(model, "blk.13.attn_q.weight")
    let W_k_13 = get_tensor(model, "blk.13.attn_k.weight")
    let W_v_13 = get_tensor(model, "blk.13.attn_v.weight")
    let W_o_13 = get_tensor(model, "blk.13.attn_output.weight")
    let attn_norm_13 = get_tensor(model, "blk.13.attn_norm.weight")
    let W_gate_13 = get_tensor(model, "blk.13.ffn_gate.weight")
    let W_up_13 = get_tensor(model, "blk.13.ffn_up.weight")
    let W_down_13 = get_tensor(model, "blk.13.ffn_down.weight")
    let ffn_norm_13 = get_tensor(model, "blk.13.ffn_norm.weight")

    // Layer 14
    let W_q_14 = get_tensor(model, "blk.14.attn_q.weight")
    let W_k_14 = get_tensor(model, "blk.14.attn_k.weight")
    let W_v_14 = get_tensor(model, "blk.14.attn_v.weight")
    let W_o_14 = get_tensor(model, "blk.14.attn_output.weight")
    let attn_norm_14 = get_tensor(model, "blk.14.attn_norm.weight")
    let W_gate_14 = get_tensor(model, "blk.14.ffn_gate.weight")
    let W_up_14 = get_tensor(model, "blk.14.ffn_up.weight")
    let W_down_14 = get_tensor(model, "blk.14.ffn_down.weight")
    let ffn_norm_14 = get_tensor(model, "blk.14.ffn_norm.weight")

    // Layer 15
    let W_q_15 = get_tensor(model, "blk.15.attn_q.weight")
    let W_k_15 = get_tensor(model, "blk.15.attn_k.weight")
    let W_v_15 = get_tensor(model, "blk.15.attn_v.weight")
    let W_o_15 = get_tensor(model, "blk.15.attn_output.weight")
    let attn_norm_15 = get_tensor(model, "blk.15.attn_norm.weight")
    let W_gate_15 = get_tensor(model, "blk.15.ffn_gate.weight")
    let W_up_15 = get_tensor(model, "blk.15.ffn_up.weight")
    let W_down_15 = get_tensor(model, "blk.15.ffn_down.weight")
    let ffn_norm_15 = get_tensor(model, "blk.15.ffn_norm.weight")

    // Layer 16
    let W_q_16 = get_tensor(model, "blk.16.attn_q.weight")
    let W_k_16 = get_tensor(model, "blk.16.attn_k.weight")
    let W_v_16 = get_tensor(model, "blk.16.attn_v.weight")
    let W_o_16 = get_tensor(model, "blk.16.attn_output.weight")
    let attn_norm_16 = get_tensor(model, "blk.16.attn_norm.weight")
    let W_gate_16 = get_tensor(model, "blk.16.ffn_gate.weight")
    let W_up_16 = get_tensor(model, "blk.16.ffn_up.weight")
    let W_down_16 = get_tensor(model, "blk.16.ffn_down.weight")
    let ffn_norm_16 = get_tensor(model, "blk.16.ffn_norm.weight")

    // Layer 17
    let W_q_17 = get_tensor(model, "blk.17.attn_q.weight")
    let W_k_17 = get_tensor(model, "blk.17.attn_k.weight")
    let W_v_17 = get_tensor(model, "blk.17.attn_v.weight")
    let W_o_17 = get_tensor(model, "blk.17.attn_output.weight")
    let attn_norm_17 = get_tensor(model, "blk.17.attn_norm.weight")
    let W_gate_17 = get_tensor(model, "blk.17.ffn_gate.weight")
    let W_up_17 = get_tensor(model, "blk.17.ffn_up.weight")
    let W_down_17 = get_tensor(model, "blk.17.ffn_down.weight")
    let ffn_norm_17 = get_tensor(model, "blk.17.ffn_norm.weight")

    // Layer 18
    let W_q_18 = get_tensor(model, "blk.18.attn_q.weight")
    let W_k_18 = get_tensor(model, "blk.18.attn_k.weight")
    let W_v_18 = get_tensor(model, "blk.18.attn_v.weight")
    let W_o_18 = get_tensor(model, "blk.18.attn_output.weight")
    let attn_norm_18 = get_tensor(model, "blk.18.attn_norm.weight")
    let W_gate_18 = get_tensor(model, "blk.18.ffn_gate.weight")
    let W_up_18 = get_tensor(model, "blk.18.ffn_up.weight")
    let W_down_18 = get_tensor(model, "blk.18.ffn_down.weight")
    let ffn_norm_18 = get_tensor(model, "blk.18.ffn_norm.weight")

    // Layer 19
    let W_q_19 = get_tensor(model, "blk.19.attn_q.weight")
    let W_k_19 = get_tensor(model, "blk.19.attn_k.weight")
    let W_v_19 = get_tensor(model, "blk.19.attn_v.weight")
    let W_o_19 = get_tensor(model, "blk.19.attn_output.weight")
    let attn_norm_19 = get_tensor(model, "blk.19.attn_norm.weight")
    let W_gate_19 = get_tensor(model, "blk.19.ffn_gate.weight")
    let W_up_19 = get_tensor(model, "blk.19.ffn_up.weight")
    let W_down_19 = get_tensor(model, "blk.19.ffn_down.weight")
    let ffn_norm_19 = get_tensor(model, "blk.19.ffn_norm.weight")

    // Layer 20
    let W_q_20 = get_tensor(model, "blk.20.attn_q.weight")
    let W_k_20 = get_tensor(model, "blk.20.attn_k.weight")
    let W_v_20 = get_tensor(model, "blk.20.attn_v.weight")
    let W_o_20 = get_tensor(model, "blk.20.attn_output.weight")
    let attn_norm_20 = get_tensor(model, "blk.20.attn_norm.weight")
    let W_gate_20 = get_tensor(model, "blk.20.ffn_gate.weight")
    let W_up_20 = get_tensor(model, "blk.20.ffn_up.weight")
    let W_down_20 = get_tensor(model, "blk.20.ffn_down.weight")
    let ffn_norm_20 = get_tensor(model, "blk.20.ffn_norm.weight")

    // Layer 21
    let W_q_21 = get_tensor(model, "blk.21.attn_q.weight")
    let W_k_21 = get_tensor(model, "blk.21.attn_k.weight")
    let W_v_21 = get_tensor(model, "blk.21.attn_v.weight")
    let W_o_21 = get_tensor(model, "blk.21.attn_output.weight")
    let attn_norm_21 = get_tensor(model, "blk.21.attn_norm.weight")
    let W_gate_21 = get_tensor(model, "blk.21.ffn_gate.weight")
    let W_up_21 = get_tensor(model, "blk.21.ffn_up.weight")
    let W_down_21 = get_tensor(model, "blk.21.ffn_down.weight")
    let ffn_norm_21 = get_tensor(model, "blk.21.ffn_norm.weight")

    print("      ✓ All 22 layers loaded")
    print("")

    // Prepare input
    print("[3/4] Preparing chat...")
    let prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("      User: \" Hello \"")
    print("      Tokens:", tokens)
    print("")

    // Forward pass
    print("[4/4] Generating response...")
    print("      (Using ALL 22 Transformer layers + Temperature=0.8)")
    print("")

    let e = embedding(embed_table, tokens)

    // Apply all 22 layers WITH residual scaling
    let h0 = transformer_layer_scaled(e, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0, residual_scale)
    let h1 = transformer_layer_scaled(h0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1, residual_scale)
    let h2 = transformer_layer_scaled(h1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2, residual_scale)
    let h3 = transformer_layer_scaled(h2, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3, residual_scale)
    let h4 = transformer_layer_scaled(h3, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4, residual_scale)
    let h5 = transformer_layer_scaled(h4, W_q_5, W_k_5, W_v_5, W_o_5, attn_norm_5, W_gate_5, W_up_5, W_down_5, ffn_norm_5, residual_scale)
    let h6 = transformer_layer_scaled(h5, W_q_6, W_k_6, W_v_6, W_o_6, attn_norm_6, W_gate_6, W_up_6, W_down_6, ffn_norm_6, residual_scale)
    let h7 = transformer_layer_scaled(h6, W_q_7, W_k_7, W_v_7, W_o_7, attn_norm_7, W_gate_7, W_up_7, W_down_7, ffn_norm_7, residual_scale)
    let h8 = transformer_layer_scaled(h7, W_q_8, W_k_8, W_v_8, W_o_8, attn_norm_8, W_gate_8, W_up_8, W_down_8, ffn_norm_8, residual_scale)
    let h9 = transformer_layer_scaled(h8, W_q_9, W_k_9, W_v_9, W_o_9, attn_norm_9, W_gate_9, W_up_9, W_down_9, ffn_norm_9, residual_scale)
    let h10 = transformer_layer_scaled(h9, W_q_10, W_k_10, W_v_10, W_o_10, attn_norm_10, W_gate_10, W_up_10, W_down_10, ffn_norm_10, residual_scale)
    let h11 = transformer_layer_scaled(h10, W_q_11, W_k_11, W_v_11, W_o_11, attn_norm_11, W_gate_11, W_up_11, W_down_11, ffn_norm_11, residual_scale)
    let h12 = transformer_layer_scaled(h11, W_q_12, W_k_12, W_v_12, W_o_12, attn_norm_12, W_gate_12, W_up_12, W_down_12, ffn_norm_12, residual_scale)
    let h13 = transformer_layer_scaled(h12, W_q_13, W_k_13, W_v_13, W_o_13, attn_norm_13, W_gate_13, W_up_13, W_down_13, ffn_norm_13, residual_scale)
    let h14 = transformer_layer_scaled(h13, W_q_14, W_k_14, W_v_14, W_o_14, attn_norm_14, W_gate_14, W_up_14, W_down_14, ffn_norm_14, residual_scale)
    let h15 = transformer_layer_scaled(h14, W_q_15, W_k_15, W_v_15, W_o_15, attn_norm_15, W_gate_15, W_up_15, W_down_15, ffn_norm_15, residual_scale)
    let h16 = transformer_layer_scaled(h15, W_q_16, W_k_16, W_v_16, W_o_16, attn_norm_16, W_gate_16, W_up_16, W_down_16, ffn_norm_16, residual_scale)
    let h17 = transformer_layer_scaled(h16, W_q_17, W_k_17, W_v_17, W_o_17, attn_norm_17, W_gate_17, W_up_17, W_down_17, ffn_norm_17, residual_scale)
    let h18 = transformer_layer_scaled(h17, W_q_18, W_k_18, W_v_18, W_o_18, attn_norm_18, W_gate_18, W_up_18, W_down_18, ffn_norm_18, residual_scale)
    let h19 = transformer_layer_scaled(h18, W_q_19, W_k_19, W_v_19, W_o_19, attn_norm_19, W_gate_19, W_up_19, W_down_19, ffn_norm_19, residual_scale)
    let h20 = transformer_layer_scaled(h19, W_q_20, W_k_20, W_v_20, W_o_20, attn_norm_20, W_gate_20, W_up_20, W_down_20, ffn_norm_20, residual_scale)
    let h21 = transformer_layer_scaled(h20, W_q_21, W_k_21, W_v_21, W_o_21, attn_norm_21, W_gate_21, W_up_21, W_down_21, ffn_norm_21, residual_scale)

    // Output
    let h_final = rms_norm(h21, output_norm)
    let logits = linear(h_final, output_weight)

    // Generate token
    print("      [1/5] Generating...")
    let token1 = temperature_sample(logits, 0.8)
    print("            Token:", token1)

    print("")
    print("=== Generation Complete ===")
    print("")
    print("With residual scaling, logits should have proper distribution.")
    print("Expected: Wide range (-10 to +20) with clear top token.")
    print("")
}
