// Layer-by-layer debugging - find where NaN first appears

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
}

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
    result := linear(attn_reshaped, W_o)
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
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    result := x1 + ffn_out
}

main {
    print("=== Layer-by-Layer NaN Debug ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("[Step 1] Embedding")
    let embed_table = get_tensor(model, "token_embd.weight")
    let tokens = [1]
    let x = embedding(embed_table, tokens)
    print("  x (first 5):", x)
    print("")

    print("[Step 2] Layer 0 - Attention Norm")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let x_norm1 = rms_norm(x, attn_norm_0)
    print("  x_norm1 (first 5):", x_norm1)
    print("")

    print("[Step 3] Layer 0 - Q projection")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let Q = linear(x_norm1, W_q_0)
    print("  Q (first 5):", Q)
    print("")

    print("[Step 4] Layer 0 - K projection")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let K = linear(x_norm1, W_k_0)
    print("  K (first 5):", K)
    print("")

    print("[Step 5] Layer 0 - V projection")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let V = linear(x_norm1, W_v_0)
    print("  V (first 5):", V)
    print("")

    print("[Step 6] Layer 0 - Attention")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o_0)
    print("  attn_out (first 5):", attn_out)
    print("")

    print("[Step 7] Layer 0 - Residual 1")
    let x1 = x + attn_out
    print("  x1 (first 5):", x1)
    print("")

    print("[Step 8] Layer 0 - FFN Norm")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let x_norm2 = rms_norm(x1, ffn_norm_0)
    print("  x_norm2 (first 5):", x_norm2)
    print("")

    print("[Step 9] Layer 0 - FFN")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_out = swiglu_ffn(x_norm2, W_gate_0, W_up_0, W_down_0)
    print("  ffn_out (first 5):", ffn_out)
    print("")

    print("[Step 10] Layer 0 - Residual 2")
    let x2 = x1 + ffn_out
    print("  x2 (final layer 0 output, first 5):", x2)
    print("")

    print("[Step 11] Final Norm")
    let output_norm = get_tensor(model, "output_norm.weight")
    let x_final = rms_norm(x2, output_norm)
    print("  x_final (first 5):", x_final)
    print("")

    print("[Step 12] LM Head")
    let lm_head = get_tensor(model, "output.weight")
    let logits = linear(x_final, lm_head)
    print("  logits (first 5):", logits)
    print("")

    print("=== Debug Complete ===")
    print("Check above for any NaN or Inf values")
}
