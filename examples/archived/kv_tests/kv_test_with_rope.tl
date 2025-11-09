// KV cache test WITH RoPE - minimal version

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let up = matmul(x, W_up)
    matmul(silu(gate) * up, W_down)
}

fn attention_with_cache(
    Q: float16[?, ?],
    K_cache: float16[?, ?],
    V_cache: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape = shape(Q)
    let seq_len_f = Q_shape[0]
    let K_shape = shape(K_cache)
    let cache_len_f = K_shape[0]

    print("      [Attn] Q:", seq_len_f, " K cache:", cache_len_f)

    print("      [Attn] Reshape Q")
    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    print("      [Attn] RoPE Q")
    let Q_rope = rope(Q_heads)
    print("      [Attn] RoPE Q done")

    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    print("      [Attn] GQA expansion")
    let K_exp = reshape(K_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    print("      [Attn] Einsum 1")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("      [Attn] Softmax")
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    print("      [Attn] Einsum 2")
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    print("      [Attn] Output proj")
    let reshaped = reshape(out, [seq_len_f, 2048.0])
    let attn_result = matmul(reshaped, W_o)
    print("      [Attn] Done")
    attn_result
}

main {
    print("=== KV Cache Test (WITH RoPE) ===")
    print("")

    print("[1/3] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    print("    ✓ Loaded")
    print("")

    print("[2/3] Token 1...")
    let tokens = [1]
    let e1 = embedding(embed_table, tokens)
    print("    Embedding:", shape(e1))

    let x_norm_0 = rms_norm(e1, attn_norm_0)
    let Q_0 = matmul(x_norm_0, W_q_0)
    let K_0 = matmul(x_norm_0, W_k_0)
    let V_0 = matmul(x_norm_0, W_v_0)

    print("    RoPE K")
    let K_0_heads = reshape(K_0, [shape(K_0)[0], 4.0, 64.0])
    let K_0_rope = rope(K_0_heads)
    let K_cache_0 = reshape(K_0_rope, [shape(K_0)[0], 256.0])
    let V_cache_0 = V_0
    print("    RoPE K done")

    print("    Attention")
    let attn_0 = attention_with_cache(Q_0, K_cache_0, V_cache_0, W_o_0)
    print("    FFN")
    let h0_res = e1 + attn_0
    let h0 = h0_res + swiglu_ffn(rms_norm(h0_res, ffn_norm_0), W_gate_0, W_up_0, W_down_0)

    print("    Sample")
    let norm1 = rms_norm(h0, output_norm)
    let logits1 = matmul(norm1, output_weight)
    let t1 = temperature_sample(logits1, 0.0)
    print("    Token 1:", t1)
    print("")

    print("[3/3] Token 2...")
    let new_token_list = [t1]
    let e2 = embedding(embed_table, new_token_list)
    print("    New embedding:", shape(e2))

    let x_norm_0_t2 = rms_norm(e2, attn_norm_0)
    let Q_0_t2 = matmul(x_norm_0_t2, W_q_0)
    let K_0_t2 = matmul(x_norm_0_t2, W_k_0)
    let V_0_t2 = matmul(x_norm_0_t2, W_v_0)

    print("    RoPE new K")
    let K_0_t2_heads = reshape(K_0_t2, [shape(K_0_t2)[0], 4.0, 64.0])
    let K_0_t2_rope = rope(K_0_t2_heads)
    let K_0_t2_flat = reshape(K_0_t2_rope, [shape(K_0_t2)[0], 256.0])
    print("    RoPE new K done")

    print("    Concat")
    let K_cache_0_all = concat(K_cache_0, K_0_t2_flat, 0.0)
    let V_cache_0_all = concat(V_cache_0, V_0_t2, 0.0)
    print("    K cache after concat:", shape(K_cache_0_all))

    print("    Attention")
    let attn_0_t2 = attention_with_cache(Q_0_t2, K_cache_0_all, V_cache_0_all, W_o_0)
    print("    FFN")
    let h0_t2_res = e2 + attn_0_t2
    let h0_t2 = h0_t2_res + swiglu_ffn(rms_norm(h0_t2_res, ffn_norm_0), W_gate_0, W_up_0, W_down_0)

    print("    Sample")
    let norm2 = rms_norm(h0_t2, output_norm)
    let logits2 = matmul(norm2, output_weight)
    let t2 = temperature_sample(logits2, 0.0)
    print("    Token 2:", t2)

    print("")
    print("========================================")
    print("Tokens:", t1, "->", t2)
    print("========================================")
    print("")
    print("✅ Test complete!")
}
