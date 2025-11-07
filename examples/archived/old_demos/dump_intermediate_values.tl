// Dump intermediate values to verify computation

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let gate_act = silu(gate)
    let up = matmul(x, W_up)
    let intermediate = gate_act * up
    result = matmul(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_heads, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    result = matmul(attn_reshaped, W_o)
}

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
    let Q = matmul(x_norm1, W_q)
    let K = matmul(x_norm1, W_k)
    let V = matmul(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    result = x1 + ffn_out
}

main {
    print("=== Dump Intermediate Values ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("Loading layer 0...")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("")

    print("Input: BOS token [1]")
    let tokens = [1.0]
    print("")

    print("Step 1: Embedding")
    let e = embedding(embed_table, tokens)
    print("  Shape:", shape(e))
    print("")

    print("Step 2: RMS Norm")
    let x_norm = rms_norm(e, attn_norm_0)
    print("  Shape:", shape(x_norm))
    print("")

    print("Step 3: Q/K/V Projection")
    let Q = matmul(x_norm, W_q_0)
    let K = matmul(x_norm, W_k_0)
    let V = matmul(x_norm, W_v_0)
    print("  Q shape:", shape(Q))
    print("  K shape:", shape(K))
    print("  V shape:", shape(V))
    print("")

    print("Step 4: Reshape for multi-head")
    let Q_heads = reshape(Q, [1.0, 32.0, 64.0])
    let K_heads = reshape(K, [1.0, 4.0, 64.0])
    let V_heads = reshape(V, [1.0, 4.0, 64.0])
    print("  Q_heads shape:", shape(Q_heads))
    print("  K_heads shape:", shape(K_heads))
    print("  V_heads shape:", shape(V_heads))
    print("")

    print("Step 5: GQA Expansion")
    let K_with_group = reshape(K_heads, [1.0, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_with_group, [1.0, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [1.0, 32.0, 64.0])
    print("  K_expanded shape:", shape(K_expanded))
    print("")

    print("Step 6: Attention scores")
    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    print("  Scores shape:", shape(scores))
    let scaled_scores = scores * 0.125
    print("  Scaled scores shape:", shape(scaled_scores))
    print("")

    print("Step 7: Softmax")
    let attn_weights = softmax(scaled_scores, 2)
    print("  Attention weights shape:", shape(attn_weights))
    print("")

    print("Step 8: Apply attention to V")
    let V_expanded = reshape(reshape(V_heads, [1.0, 4.0, 1.0, 64.0]), [1.0, 32.0, 64.0])
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("  Attention output shape:", shape(attn_output))
    print("")

    print("Step 9: Output projection")
    let attn_reshaped = reshape(attn_output, [1.0, 2048.0])
    let attn_proj = matmul(attn_reshaped, W_o_0)
    print("  Projected output shape:", shape(attn_proj))
    print("")

    print("Step 10: Residual + FFN")
    let x1 = e + attn_proj
    let x_norm2 = rms_norm(x1, ffn_norm_0)
    let ffn_out = swiglu_ffn(x_norm2, W_gate_0, W_up_0, W_down_0)
    let h0 = x1 + ffn_out
    print("  Layer 0 output shape:", shape(h0))
    print("")

    print("Step 11: Final projection")
    let final_norm = rms_norm(h0, output_norm)
    let logits = matmul(final_norm, output_weight)
    print("  Logits shape:", shape(logits))
    print("")

    let token = temperature_sample(logits, 0.0)
    print("Final sampled token:", token)
    print("")

    print("✅ All intermediate values dumped!")
}
