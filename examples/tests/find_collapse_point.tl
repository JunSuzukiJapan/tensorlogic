// Test: Find exact layer where collapse begins
// Test layers: 0, 1, 2, 3, 5, 7, 10

// SiLU activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
}

// SwiGLU FFN
fn swiglu_ffn(x: float16[?, ?], W_gate: float16[?, ?], W_up: float16[?, ?], W_down: float16[?, ?]) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    result = linear(intermediate, W_down)
}

// Grouped Query Attention
fn tinyllama_gqa_attention(Q: float16[?, ?], K: float16[?, ?], V: float16[?, ?], W_o: float16[?, ?]) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    result = linear(attn_reshaped, W_o)
}

// Single Transformer Layer
fn transformer_layer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?]
) -> float16[?, ?] {
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    result = x1 + ffn_out
}

main {
    print("=== Finding Exact Collapse Point (Layers 0, 1, 2, 3, 5, 7, 10) ===\n")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let tokens = tokenize(tokenizer, "<s>", false)
    
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let e = embedding(embed_table, tokens)

    // Load all layer weights (0-10)
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")

    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")

    let W_q_3 = get_tensor(model, "blk.3.attn_q.weight")
    let W_k_3 = get_tensor(model, "blk.3.attn_k.weight")
    let W_v_3 = get_tensor(model, "blk.3.attn_v.weight")
    let W_o_3 = get_tensor(model, "blk.3.attn_output.weight")
    let attn_norm_3 = get_tensor(model, "blk.3.attn_norm.weight")
    let W_gate_3 = get_tensor(model, "blk.3.ffn_gate.weight")
    let W_up_3 = get_tensor(model, "blk.3.ffn_up.weight")
    let W_down_3 = get_tensor(model, "blk.3.ffn_down.weight")
    let ffn_norm_3 = get_tensor(model, "blk.3.ffn_norm.weight")

    let W_q_4 = get_tensor(model, "blk.4.attn_q.weight")
    let W_k_4 = get_tensor(model, "blk.4.attn_k.weight")
    let W_v_4 = get_tensor(model, "blk.4.attn_v.weight")
    let W_o_4 = get_tensor(model, "blk.4.attn_output.weight")
    let attn_norm_4 = get_tensor(model, "blk.4.attn_norm.weight")
    let W_gate_4 = get_tensor(model, "blk.4.ffn_gate.weight")
    let W_up_4 = get_tensor(model, "blk.4.ffn_up.weight")
    let W_down_4 = get_tensor(model, "blk.4.ffn_down.weight")
    let ffn_norm_4 = get_tensor(model, "blk.4.ffn_norm.weight")

    let W_q_5 = get_tensor(model, "blk.5.attn_q.weight")
    let W_k_5 = get_tensor(model, "blk.5.attn_k.weight")
    let W_v_5 = get_tensor(model, "blk.5.attn_v.weight")
    let W_o_5 = get_tensor(model, "blk.5.attn_output.weight")
    let attn_norm_5 = get_tensor(model, "blk.5.attn_norm.weight")
    let W_gate_5 = get_tensor(model, "blk.5.ffn_gate.weight")
    let W_up_5 = get_tensor(model, "blk.5.ffn_up.weight")
    let W_down_5 = get_tensor(model, "blk.5.ffn_down.weight")
    let ffn_norm_5 = get_tensor(model, "blk.5.ffn_norm.weight")

    let W_q_6 = get_tensor(model, "blk.6.attn_q.weight")
    let W_k_6 = get_tensor(model, "blk.6.attn_k.weight")
    let W_v_6 = get_tensor(model, "blk.6.attn_v.weight")
    let W_o_6 = get_tensor(model, "blk.6.attn_output.weight")
    let attn_norm_6 = get_tensor(model, "blk.6.attn_norm.weight")
    let W_gate_6 = get_tensor(model, "blk.6.ffn_gate.weight")
    let W_up_6 = get_tensor(model, "blk.6.ffn_up.weight")
    let W_down_6 = get_tensor(model, "blk.6.ffn_down.weight")
    let ffn_norm_6 = get_tensor(model, "blk.6.ffn_norm.weight")

    let W_q_7 = get_tensor(model, "blk.7.attn_q.weight")
    let W_k_7 = get_tensor(model, "blk.7.attn_k.weight")
    let W_v_7 = get_tensor(model, "blk.7.attn_v.weight")
    let W_o_7 = get_tensor(model, "blk.7.attn_output.weight")
    let attn_norm_7 = get_tensor(model, "blk.7.attn_norm.weight")
    let W_gate_7 = get_tensor(model, "blk.7.ffn_gate.weight")
    let W_up_7 = get_tensor(model, "blk.7.ffn_up.weight")
    let W_down_7 = get_tensor(model, "blk.7.ffn_down.weight")
    let ffn_norm_7 = get_tensor(model, "blk.7.ffn_norm.weight")

    let W_q_8 = get_tensor(model, "blk.8.attn_q.weight")
    let W_k_8 = get_tensor(model, "blk.8.attn_k.weight")
    let W_v_8 = get_tensor(model, "blk.8.attn_v.weight")
    let W_o_8 = get_tensor(model, "blk.8.attn_output.weight")
    let attn_norm_8 = get_tensor(model, "blk.8.attn_norm.weight")
    let W_gate_8 = get_tensor(model, "blk.8.ffn_gate.weight")
    let W_up_8 = get_tensor(model, "blk.8.ffn_up.weight")
    let W_down_8 = get_tensor(model, "blk.8.ffn_down.weight")
    let ffn_norm_8 = get_tensor(model, "blk.8.ffn_norm.weight")

    let W_q_9 = get_tensor(model, "blk.9.attn_q.weight")
    let W_k_9 = get_tensor(model, "blk.9.attn_k.weight")
    let W_v_9 = get_tensor(model, "blk.9.attn_v.weight")
    let W_o_9 = get_tensor(model, "blk.9.attn_output.weight")
    let attn_norm_9 = get_tensor(model, "blk.9.attn_norm.weight")
    let W_gate_9 = get_tensor(model, "blk.9.ffn_gate.weight")
    let W_up_9 = get_tensor(model, "blk.9.ffn_up.weight")
    let W_down_9 = get_tensor(model, "blk.9.ffn_down.weight")
    let ffn_norm_9 = get_tensor(model, "blk.9.ffn_norm.weight")

    let W_q_10 = get_tensor(model, "blk.10.attn_q.weight")
    let W_k_10 = get_tensor(model, "blk.10.attn_k.weight")
    let W_v_10 = get_tensor(model, "blk.10.attn_v.weight")
    let W_o_10 = get_tensor(model, "blk.10.attn_output.weight")
    let attn_norm_10 = get_tensor(model, "blk.10.attn_norm.weight")
    let W_gate_10 = get_tensor(model, "blk.10.ffn_gate.weight")
    let W_up_10 = get_tensor(model, "blk.10.ffn_up.weight")
    let W_down_10 = get_tensor(model, "blk.10.ffn_down.weight")
    let ffn_norm_10 = get_tensor(model, "blk.10.ffn_norm.weight")

    print("Weights loaded. Running layers...\n")

    // Layer 0
    let h0 = transformer_layer(e, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let norm0 = rms_norm(h0, output_norm)
    let logits0 = linear(norm0, output_weight)
    let token0 = temperature_sample(logits0, 0.0)
    print("[Layer 0]  Token:", token0)

    // Layer 1
    let h1 = transformer_layer(h0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let norm1 = rms_norm(h1, output_norm)
    let logits1 = linear(norm1, output_weight)
    let token1 = temperature_sample(logits1, 0.0)
    print("[Layer 1]  Token:", token1)

    // Layer 2
    let h2 = transformer_layer(h1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)
    let norm2 = rms_norm(h2, output_norm)
    let logits2 = linear(norm2, output_weight)
    let token2 = temperature_sample(logits2, 0.0)
    print("[Layer 2]  Token:", token2)

    // Layer 3
    let h3 = transformer_layer(h2, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3)
    let norm3 = rms_norm(h3, output_norm)
    let logits3 = linear(norm3, output_weight)
    let token3 = temperature_sample(logits3, 0.0)
    print("[Layer 3]  Token:", token3)

    // Layer 4
    let h4 = transformer_layer(h3, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4)
    let norm4 = rms_norm(h4, output_norm)
    let logits4 = linear(norm4, output_weight)
    let token4 = temperature_sample(logits4, 0.0)
    print("[Layer 4]  Token:", token4)

    // Layer 5
    let h5 = transformer_layer(h4, W_q_5, W_k_5, W_v_5, W_o_5, attn_norm_5, W_gate_5, W_up_5, W_down_5, ffn_norm_5)
    let norm5 = rms_norm(h5, output_norm)
    let logits5 = linear(norm5, output_weight)
    let token5 = temperature_sample(logits5, 0.0)
    print("[Layer 5]  Token:", token5)

    // Layer 6
    let h6 = transformer_layer(h5, W_q_6, W_k_6, W_v_6, W_o_6, attn_norm_6, W_gate_6, W_up_6, W_down_6, ffn_norm_6)
    let norm6 = rms_norm(h6, output_norm)
    let logits6 = linear(norm6, output_weight)
    let token6 = temperature_sample(logits6, 0.0)
    print("[Layer 6]  Token:", token6)

    // Layer 7
    let h7 = transformer_layer(h6, W_q_7, W_k_7, W_v_7, W_o_7, attn_norm_7, W_gate_7, W_up_7, W_down_7, ffn_norm_7)
    let norm7 = rms_norm(h7, output_norm)
    let logits7 = linear(norm7, output_weight)
    let token7 = temperature_sample(logits7, 0.0)
    print("[Layer 7]  Token:", token7)

    // Layer 8
    let h8 = transformer_layer(h7, W_q_8, W_k_8, W_v_8, W_o_8, attn_norm_8, W_gate_8, W_up_8, W_down_8, ffn_norm_8)
    let norm8 = rms_norm(h8, output_norm)
    let logits8 = linear(norm8, output_weight)
    let token8 = temperature_sample(logits8, 0.0)
    print("[Layer 8]  Token:", token8)

    // Layer 9
    let h9 = transformer_layer(h8, W_q_9, W_k_9, W_v_9, W_o_9, attn_norm_9, W_gate_9, W_up_9, W_down_9, ffn_norm_9)
    let norm9 = rms_norm(h9, output_norm)
    let logits9 = linear(norm9, output_weight)
    let token9 = temperature_sample(logits9, 0.0)
    print("[Layer 9]  Token:", token9)

    // Layer 10
    let h10 = transformer_layer(h9, W_q_10, W_k_10, W_v_10, W_o_10, attn_norm_10, W_gate_10, W_up_10, W_down_10, ffn_norm_10)
    let norm10 = rms_norm(h10, output_norm)
    let logits10 = linear(norm10, output_weight)
    let token10 = temperature_sample(logits10, 0.0)
    print("[Layer 10] Token:", token10)

    print("\n=== Analysis ===")
    print("Check the logit differences in the debug output above")
    print("If difference drops below 0.5, that's where collapse begins")
}
