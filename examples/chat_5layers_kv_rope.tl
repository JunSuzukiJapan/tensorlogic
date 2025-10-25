// 5-Layer Chat with KV Cache + RoPE
// Tests scaling performance with incremental layer count

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let up = matmul(x, W_up)
    result := matmul(silu(gate) * up, W_down)
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
    result := matmul(reshaped, W_o)
}

main {
    print("=== 5-Layer Chat with KV Cache + RoPE ===")
    print("")

    print("[1/4] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load 5 layers
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
    print("    ✓ 5 layers loaded")
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

    // Layer 0
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

    // Layer 2
    let x_norm_2 = rms_norm(h1, attn_norm_2)
    let Q_2 = matmul(x_norm_2, W_q_2)
    let K_2 = matmul(x_norm_2, W_k_2)
    let V_2 = matmul(x_norm_2, W_v_2)
    let K_2_heads = reshape(K_2, [shape(K_2)[0], 4.0, 64.0])
    let K_2_rope = rope(K_2_heads)
    let K_cache_2 = reshape(K_2_rope, [shape(K_2)[0], 256.0])
    let V_cache_2 = reshape(V_2, [shape(V_2)[0], 256.0])
    let attn_2 = attention_with_cache(Q_2, K_cache_2, V_cache_2, W_o_2)
    let h2_res = h1 + attn_2
    let h2 = h2_res + swiglu_ffn(rms_norm(h2_res, ffn_norm_2), W_gate_2, W_up_2, W_down_2)

    // Layer 3
    let x_norm_3 = rms_norm(h2, attn_norm_3)
    let Q_3 = matmul(x_norm_3, W_q_3)
    let K_3 = matmul(x_norm_3, W_k_3)
    let V_3 = matmul(x_norm_3, W_v_3)
    let K_3_heads = reshape(K_3, [shape(K_3)[0], 4.0, 64.0])
    let K_3_rope = rope(K_3_heads)
    let K_cache_3 = reshape(K_3_rope, [shape(K_3)[0], 256.0])
    let V_cache_3 = reshape(V_3, [shape(V_3)[0], 256.0])
    let attn_3 = attention_with_cache(Q_3, K_cache_3, V_cache_3, W_o_3)
    let h3_res = h2 + attn_3
    let h3 = h3_res + swiglu_ffn(rms_norm(h3_res, ffn_norm_3), W_gate_3, W_up_3, W_down_3)

    // Layer 4
    let x_norm_4 = rms_norm(h3, attn_norm_4)
    let Q_4 = matmul(x_norm_4, W_q_4)
    let K_4 = matmul(x_norm_4, W_k_4)
    let V_4 = matmul(x_norm_4, W_v_4)
    let K_4_heads = reshape(K_4, [shape(K_4)[0], 4.0, 64.0])
    let K_4_rope = rope(K_4_heads)
    let K_cache_4 = reshape(K_4_rope, [shape(K_4)[0], 256.0])
    let V_cache_4 = reshape(V_4, [shape(V_4)[0], 256.0])
    let attn_4 = attention_with_cache(Q_4, K_cache_4, V_cache_4, W_o_4)
    let h4_res = h3 + attn_4
    let h4 = h4_res + swiglu_ffn(rms_norm(h4_res, ffn_norm_4), W_gate_4, W_up_4, W_down_4)

    let norm1 = rms_norm(h4, output_norm)
    let logits1 = matmul(norm1, output_weight)
    let t1 = temperature_sample(logits1, 0.8)
    print("    Token 1:", t1)

    let gen_tokens = append(tokens, t1)

    // CRITICAL: Clean up intermediate tensors to free GPU memory
    print("    Cleaning up intermediate tensors...")
    // Keep only KV caches and essential variables for Token 2
    cleanup(
        "K_cache_0", "V_cache_0",
        "K_cache_1", "V_cache_1",
        "K_cache_2", "V_cache_2",
        "K_cache_3", "V_cache_3",
        "K_cache_4", "V_cache_4",
        "model", "tokenizer", "embed_table", "output_norm", "output_weight",
        "W_q_0", "W_k_0", "W_v_0", "W_o_0", "attn_norm_0", "ffn_norm_0",
        "W_gate_0", "W_up_0", "W_down_0",
        "W_q_1", "W_k_1", "W_v_1", "W_o_1", "attn_norm_1", "ffn_norm_1",
        "W_gate_1", "W_up_1", "W_down_1",
        "W_q_2", "W_k_2", "W_v_2", "W_o_2", "attn_norm_2", "ffn_norm_2",
        "W_gate_2", "W_up_2", "W_down_2",
        "W_q_3", "W_k_3", "W_v_3", "W_o_3", "attn_norm_3", "ffn_norm_3",
        "W_gate_3", "W_up_3", "W_down_3",
        "W_q_4", "W_k_4", "W_v_4", "W_o_4", "attn_norm_4", "ffn_norm_4",
        "W_gate_4", "W_up_4", "W_down_4",
        "tokens", "gen_tokens", "t1"
    )
    print("    ✓ Intermediate tensors cleaned up - buffers recycled to pool")

    // ====== TOKEN 2 ======
    print("")
    print("[4/4] Token 2 (KV cache)...")
    print("    Creating token list...")
    // Only embed the NEW token
    let new_token_list = [t1]
    print("    Embedding new token...")
    let e2 = embedding(embed_table, new_token_list)
    print("    Embedding done, shape:", shape(e2))

    // Layer 0
    print("    Layer 0 - RMS norm...")
    let x_norm_0_t2 = rms_norm(e2, attn_norm_0)
    print("    Layer 0 - QKV matmul...")
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

    // Layer 2
    let x_norm_2_t2 = rms_norm(h1_t2, attn_norm_2)
    let Q_2_t2 = matmul(x_norm_2_t2, W_q_2)
    let K_2_t2 = matmul(x_norm_2_t2, W_k_2)
    let V_2_t2 = matmul(x_norm_2_t2, W_v_2)
    let K_2_t2_heads = reshape(K_2_t2, [shape(K_2_t2)[0], 4.0, 64.0])
    let K_2_t2_rope = rope(K_2_t2_heads)
    let K_2_t2_flat = reshape(K_2_t2_rope, [shape(K_2_t2)[0], 256.0])
    let V_2_t2_flat = reshape(V_2_t2, [shape(V_2_t2)[0], 256.0])
    let K_cache_2_all = concat(K_cache_2, K_2_t2_flat, 0.0)
    let V_cache_2_all = concat(V_cache_2, V_2_t2_flat, 0.0)
    let attn_2_t2 = attention_with_cache(Q_2_t2, K_cache_2_all, V_cache_2_all, W_o_2)
    let h2_t2_res = h1_t2 + attn_2_t2
    let h2_t2 = h2_t2_res + swiglu_ffn(rms_norm(h2_t2_res, ffn_norm_2), W_gate_2, W_up_2, W_down_2)

    // Layer 3
    let x_norm_3_t2 = rms_norm(h2_t2, attn_norm_3)
    let Q_3_t2 = matmul(x_norm_3_t2, W_q_3)
    let K_3_t2 = matmul(x_norm_3_t2, W_k_3)
    let V_3_t2 = matmul(x_norm_3_t2, W_v_3)
    let K_3_t2_heads = reshape(K_3_t2, [shape(K_3_t2)[0], 4.0, 64.0])
    let K_3_t2_rope = rope(K_3_t2_heads)
    let K_3_t2_flat = reshape(K_3_t2_rope, [shape(K_3_t2)[0], 256.0])
    let V_3_t2_flat = reshape(V_3_t2, [shape(V_3_t2)[0], 256.0])
    let K_cache_3_all = concat(K_cache_3, K_3_t2_flat, 0.0)
    let V_cache_3_all = concat(V_cache_3, V_3_t2_flat, 0.0)
    let attn_3_t2 = attention_with_cache(Q_3_t2, K_cache_3_all, V_cache_3_all, W_o_3)
    let h3_t2_res = h2_t2 + attn_3_t2
    let h3_t2 = h3_t2_res + swiglu_ffn(rms_norm(h3_t2_res, ffn_norm_3), W_gate_3, W_up_3, W_down_3)

    // Layer 4
    let x_norm_4_t2 = rms_norm(h3_t2, attn_norm_4)
    let Q_4_t2 = matmul(x_norm_4_t2, W_q_4)
    let K_4_t2 = matmul(x_norm_4_t2, W_k_4)
    let V_4_t2 = matmul(x_norm_4_t2, W_v_4)
    let K_4_t2_heads = reshape(K_4_t2, [shape(K_4_t2)[0], 4.0, 64.0])
    let K_4_t2_rope = rope(K_4_t2_heads)
    let K_4_t2_flat = reshape(K_4_t2_rope, [shape(K_4_t2)[0], 256.0])
    let V_4_t2_flat = reshape(V_4_t2, [shape(V_4_t2)[0], 256.0])
    let K_cache_4_all = concat(K_cache_4, K_4_t2_flat, 0.0)
    let V_cache_4_all = concat(V_cache_4, V_4_t2_flat, 0.0)
    let attn_4_t2 = attention_with_cache(Q_4_t2, K_cache_4_all, V_cache_4_all, W_o_4)
    let h4_t2_res = h3_t2 + attn_4_t2
    let h4_t2 = h4_t2_res + swiglu_ffn(rms_norm(h4_t2_res, ffn_norm_4), W_gate_4, W_up_4, W_down_4)

    let norm2 = rms_norm(h4_t2, output_norm)
    let logits2 = matmul(norm2, output_weight)
    let t2 = temperature_sample(logits2, 0.8)
    print("    Token 2:", t2)

    print("")
    print("========================================")
    let final_tokens = append(gen_tokens, t2)
    let response = detokenize(tokenizer, final_tokens, true)
    print("Response:", response)
    print("========================================")
    print("")
    print("✅ 5-layer test complete!")
}
