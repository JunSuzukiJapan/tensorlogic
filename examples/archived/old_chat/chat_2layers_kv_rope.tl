// 2-layer chat with KV cache + RoPE using full tokenized prompt

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let up = matmul(x, W_up)
    result = matmul(silu(gate) * up, W_down)
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

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let Q_rope = rope(Q_heads)

    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_exp = reshape(K_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    let reshaped = reshape(out, [seq_len_f, 2048.0])
    let attn_result = matmul(reshaped, W_o)
    result = attn_result
}

main {
    print("=== 2-Layer Chat with KV Cache + RoPE ===")
    print("")

    print("[1/4] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

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

    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    print("    ✓ Loaded")
    print("")

    print("[2/4] Tokenizing...")
    let user_input = "Hello"
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\n" + user_input + "\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("    ✓ Tokenized")
    print("")

    // ====== TOKEN 1 ======
    print("[3/4] Token 1 (full forward)...")
    let e1 = embedding(embed_table, tokens)
    print("    Embedding:", shape(e1))

    // Layer 0
    print("    Layer 0")
    let x_norm_0 = rms_norm(e1, attn_norm_0)
    let Q_0 = matmul(x_norm_0, W_q_0)
    let K_0 = matmul(x_norm_0, W_k_0)
    let V_0 = matmul(x_norm_0, W_v_0)
    let K_0_heads = reshape(K_0, [shape(K_0)[0], 4.0, 64.0])
    let K_0_rope = rope(K_0_heads)
    let K_cache_0 = reshape(K_0_rope, [shape(K_0)[0], 256.0])
    let V_cache_0 = reshape(V_0, [shape(V_0)[0], 256.0])
    let attn_0 = attention_with_cache(Q_0, K_cache_0, V_cache_0, W_o_0)
    let h0_res = e1 + attn_0
    let h0 = h0_res + swiglu_ffn(rms_norm(h0_res, ffn_norm_0), W_gate_0, W_up_0, W_down_0)

    // Layer 1
    print("    Layer 1")
    let x_norm_1 = rms_norm(h0, attn_norm_1)
    let Q_1 = matmul(x_norm_1, W_q_1)
    let K_1 = matmul(x_norm_1, W_k_1)
    let V_1 = matmul(x_norm_1, W_v_1)
    let K_1_heads = reshape(K_1, [shape(K_1)[0], 4.0, 64.0])
    let K_1_rope = rope(K_1_heads)
    let K_cache_1 = reshape(K_1_rope, [shape(K_1)[0], 256.0])
    let V_cache_1 = reshape(V_1, [shape(V_1)[0], 256.0])
    let attn_1 = attention_with_cache(Q_1, K_cache_1, V_cache_1, W_o_1)
    let h1_res = h0 + attn_1
    let h1 = h1_res + swiglu_ffn(rms_norm(h1_res, ffn_norm_1), W_gate_1, W_up_1, W_down_1)

    print("    Final norm + sample")
    let norm1 = rms_norm(h1, output_norm)
    let logits1 = matmul(norm1, output_weight)
    let t1 = temperature_sample(logits1, 0.0)
    print("    Token 1:", t1)
    print("")

    // ====== TOKEN 2 ======
    print("[4/4] Token 2 (KV cache)...")
    let new_token_list = [t1]
    let e2 = embedding(embed_table, new_token_list)

    // Layer 0
    print("    Layer 0")
    let x_norm_0_t2 = rms_norm(e2, attn_norm_0)
    let Q_0_t2 = matmul(x_norm_0_t2, W_q_0)
    let K_0_t2 = matmul(x_norm_0_t2, W_k_0)
    let V_0_t2 = matmul(x_norm_0_t2, W_v_0)
    let K_0_t2_heads = reshape(K_0_t2, [shape(K_0_t2)[0], 4.0, 64.0])
    let K_0_t2_rope = rope(K_0_t2_heads)
    let K_0_t2_flat = reshape(K_0_t2_rope, [shape(K_0_t2)[0], 256.0])
    let V_0_t2_flat = reshape(V_0_t2, [shape(V_0_t2)[0], 256.0])
    let K_cache_0_all = concat(K_cache_0, K_0_t2_flat, 0.0)
    let V_cache_0_all = concat(V_cache_0, V_0_t2_flat, 0.0)
    let attn_0_t2 = attention_with_cache(Q_0_t2, K_cache_0_all, V_cache_0_all, W_o_0)
    let h0_t2_res = e2 + attn_0_t2
    let h0_t2 = h0_t2_res + swiglu_ffn(rms_norm(h0_t2_res, ffn_norm_0), W_gate_0, W_up_0, W_down_0)

    // Layer 1
    print("    Layer 1")
    let x_norm_1_t2 = rms_norm(h0_t2, attn_norm_1)
    let Q_1_t2 = matmul(x_norm_1_t2, W_q_1)
    let K_1_t2 = matmul(x_norm_1_t2, W_k_1)
    let V_1_t2 = matmul(x_norm_1_t2, W_v_1)
    let K_1_t2_heads = reshape(K_1_t2, [shape(K_1_t2)[0], 4.0, 64.0])
    let K_1_t2_rope = rope(K_1_t2_heads)
    let K_1_t2_flat = reshape(K_1_t2_rope, [shape(K_1_t2)[0], 256.0])
    let V_1_t2_flat = reshape(V_1_t2, [shape(V_1_t2)[0], 256.0])
    let K_cache_1_all = concat(K_cache_1, K_1_t2_flat, 0.0)
    let V_cache_1_all = concat(V_cache_1, V_1_t2_flat, 0.0)
    let attn_1_t2 = attention_with_cache(Q_1_t2, K_cache_1_all, V_cache_1_all, W_o_1)
    let h1_t2_res = h0_t2 + attn_1_t2
    let h1_t2 = h1_t2_res + swiglu_ffn(rms_norm(h1_t2_res, ffn_norm_1), W_gate_1, W_up_1, W_down_1)

    print("    Final norm + sample")
    let norm2 = rms_norm(h1_t2, output_norm)
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
