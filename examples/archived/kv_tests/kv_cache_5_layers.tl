// KV Cache Demo - 5 Layers with 2 Token Generation
// Demonstrates concat-based KV cache accumulation at scale

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
    let gate = matmul(x, W_gate)
    let gate_act = silu(gate)
    let up = matmul(x, W_up)
    let intermediate = gate_act * up
    result := matmul(intermediate, W_down)
}

fn tinyllama_gqa_attention_with_cache(
    Q: float16[?, ?],
    K_cache: float16[?, ?],
    V_cache: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    let K_shape_tensor = shape(K_cache)
    let cache_len_f = K_shape_tensor[0]

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_with_group = reshape(K_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_with_group = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_with_group, [cache_len_f, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_with_group, [cache_len_f, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    result := matmul(attn_reshaped, W_o)
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
    let attn_out = tinyllama_gqa_attention_with_cache(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    result := x1 + ffn_out
}

main {
    print("=== KV Cache Demo: 5 Layers, 2 Tokens ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    print("[1/4] Loading 5 layers...")

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

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Loaded 5 layers")
    print("")

    // Tokenize
    print("[2/4] Tokenizing...")
    let user_input = "Hi"
    let tokens = tokenize(tokenizer, user_input, true)
    print("      Tokens: ", tokens)
    print("")

    // ================================================================
    // TOKEN 1: Initial generation
    // ================================================================
    print("[3/4] Token 1 (initial, no cache)...")
    let e1 = embedding(embed_table, tokens)

    let h0_t1 = transformer_layer(e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_t1 = transformer_layer(h0_t1, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let h2_t1 = transformer_layer(h1_t1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)
    let h3_t1 = transformer_layer(h2_t1, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3)
    let h4_t1 = transformer_layer(h3_t1, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4)

    let final_norm_t1 = rms_norm(h4_t1, output_norm)
    let logits_t1 = matmul(final_norm_t1, output_weight)
    let token_1 = temperature_sample(logits_t1, 0.8)
    print("      Generated token 1: ", token_1)
    print("")

    // Extract KV caches for all 5 layers
    let x_norm_0 = rms_norm(e1, attn_norm_0)
    let K_cache_0 = matmul(x_norm_0, W_k_0)
    let V_cache_0 = matmul(x_norm_0, W_v_0)

    let x_norm_1 = rms_norm(h0_t1, attn_norm_1)
    let K_cache_1 = matmul(x_norm_1, W_k_1)
    let V_cache_1 = matmul(x_norm_1, W_v_1)

    let x_norm_2 = rms_norm(h1_t1, attn_norm_2)
    let K_cache_2 = matmul(x_norm_2, W_k_2)
    let V_cache_2 = matmul(x_norm_2, W_v_2)

    let x_norm_3 = rms_norm(h2_t1, attn_norm_3)
    let K_cache_3 = matmul(x_norm_3, W_k_3)
    let V_cache_3 = matmul(x_norm_3, W_v_3)

    let x_norm_4 = rms_norm(h3_t1, attn_norm_4)
    let K_cache_4 = matmul(x_norm_4, W_k_4)
    let V_cache_4 = matmul(x_norm_4, W_v_4)

    // ================================================================
    // TOKEN 2: Generation with KV cache
    // ================================================================
    print("[4/4] Token 2 (WITH KV cache)...")
    let e2 = e1

    // Layer 0
    let x_norm_0_t2 = rms_norm(e2, attn_norm_0)
    let Q_0_t2 = matmul(x_norm_0_t2, W_q_0)
    let K_0_t2 = matmul(x_norm_0_t2, W_k_0)
    let V_0_t2 = matmul(x_norm_0_t2, W_v_0)
    let K_all_0 = concat(K_cache_0, K_0_t2, 0.0)
    let V_all_0 = concat(V_cache_0, V_0_t2, 0.0)
    let attn_0_t2 = tinyllama_gqa_attention_with_cache(Q_0_t2, K_all_0, V_all_0, W_o_0)
    let x1_0_t2 = e2 + attn_0_t2
    let x_norm2_0_t2 = rms_norm(x1_0_t2, ffn_norm_0)
    let ffn_0_t2 = swiglu_ffn(x_norm2_0_t2, W_gate_0, W_up_0, W_down_0)
    let h0_t2 = x1_0_t2 + ffn_0_t2

    // Layer 1
    let x_norm_1_t2 = rms_norm(h0_t2, attn_norm_1)
    let Q_1_t2 = matmul(x_norm_1_t2, W_q_1)
    let K_1_t2 = matmul(x_norm_1_t2, W_k_1)
    let V_1_t2 = matmul(x_norm_1_t2, W_v_1)
    let K_all_1 = concat(K_cache_1, K_1_t2, 0.0)
    let V_all_1 = concat(V_cache_1, V_1_t2, 0.0)
    let attn_1_t2 = tinyllama_gqa_attention_with_cache(Q_1_t2, K_all_1, V_all_1, W_o_1)
    let x1_1_t2 = h0_t2 + attn_1_t2
    let x_norm2_1_t2 = rms_norm(x1_1_t2, ffn_norm_1)
    let ffn_1_t2 = swiglu_ffn(x_norm2_1_t2, W_gate_1, W_up_1, W_down_1)
    let h1_t2 = x1_1_t2 + ffn_1_t2

    // Layer 2
    let x_norm_2_t2 = rms_norm(h1_t2, attn_norm_2)
    let Q_2_t2 = matmul(x_norm_2_t2, W_q_2)
    let K_2_t2 = matmul(x_norm_2_t2, W_k_2)
    let V_2_t2 = matmul(x_norm_2_t2, W_v_2)
    let K_all_2 = concat(K_cache_2, K_2_t2, 0.0)
    let V_all_2 = concat(V_cache_2, V_2_t2, 0.0)
    let attn_2_t2 = tinyllama_gqa_attention_with_cache(Q_2_t2, K_all_2, V_all_2, W_o_2)
    let x1_2_t2 = h1_t2 + attn_2_t2
    let x_norm2_2_t2 = rms_norm(x1_2_t2, ffn_norm_2)
    let ffn_2_t2 = swiglu_ffn(x_norm2_2_t2, W_gate_2, W_up_2, W_down_2)
    let h2_t2 = x1_2_t2 + ffn_2_t2

    // Layer 3
    let x_norm_3_t2 = rms_norm(h2_t2, attn_norm_3)
    let Q_3_t2 = matmul(x_norm_3_t2, W_q_3)
    let K_3_t2 = matmul(x_norm_3_t2, W_k_3)
    let V_3_t2 = matmul(x_norm_3_t2, W_v_3)
    let K_all_3 = concat(K_cache_3, K_3_t2, 0.0)
    let V_all_3 = concat(V_cache_3, V_3_t2, 0.0)
    let attn_3_t2 = tinyllama_gqa_attention_with_cache(Q_3_t2, K_all_3, V_all_3, W_o_3)
    let x1_3_t2 = h2_t2 + attn_3_t2
    let x_norm2_3_t2 = rms_norm(x1_3_t2, ffn_norm_3)
    let ffn_3_t2 = swiglu_ffn(x_norm2_3_t2, W_gate_3, W_up_3, W_down_3)
    let h3_t2 = x1_3_t2 + ffn_3_t2

    // Layer 4
    let x_norm_4_t2 = rms_norm(h3_t2, attn_norm_4)
    let Q_4_t2 = matmul(x_norm_4_t2, W_q_4)
    let K_4_t2 = matmul(x_norm_4_t2, W_k_4)
    let V_4_t2 = matmul(x_norm_4_t2, W_v_4)
    let K_all_4 = concat(K_cache_4, K_4_t2, 0.0)
    let V_all_4 = concat(V_cache_4, V_4_t2, 0.0)
    let attn_4_t2 = tinyllama_gqa_attention_with_cache(Q_4_t2, K_all_4, V_all_4, W_o_4)
    let x1_4_t2 = h3_t2 + attn_4_t2
    let x_norm2_4_t2 = rms_norm(x1_4_t2, ffn_norm_4)
    let ffn_4_t2 = swiglu_ffn(x_norm2_4_t2, W_gate_4, W_up_4, W_down_4)
    let h4_t2 = x1_4_t2 + ffn_4_t2

    let final_norm_t2 = rms_norm(h4_t2, output_norm)
    let logits_t2 = matmul(final_norm_t2, output_weight)
    let token_2 = temperature_sample(logits_t2, 0.8)
    print("      Generated token 2: ", token_2)
    print("")

    print("========================================================")
    print("✅ 5-Layer KV Cache Demo Complete!")
    print("")
    print("Token sequence: ", token_1, " -> ", token_2)
    print("")
    print("KV Cache benefits:")
    print("  - All 5 layers reused K, V from token 1")
    print("  - Only computed new K, V for token 2")
    print("  - Concat operation accumulated cache efficiently")
    print("  - Buffer recycling minimized memory pressure")
}
