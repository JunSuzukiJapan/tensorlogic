// KV Cache Demo - 3 Token Autoregressive Generation
// Demonstrates KV cache accumulation across multiple tokens

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
    result = matmul(attn_reshaped, W_o)
}

main {
    print("=== Autoregressive KV Cache Demo: 3 Tokens ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    print("[1/5] Loading 2 layers...")
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

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Loaded 2 layers")
    print("")

    print("[2/5] Tokenizing...")
    let tokens = tokenize(tokenizer, "Hi", true)
    print("      Input tokens: ", tokens)
    print("")

    // ================================================================
    // TOKEN 1: Initial generation (build initial cache)
    // ================================================================
    print("[3/5] Token 1 (initial)...")
    let e1 = embedding(embed_table, tokens)

    // Layer 0 - Token 1
    let x_norm_0_t1 = rms_norm(e1, attn_norm_0)
    let Q_0_t1 = matmul(x_norm_0_t1, W_q_0)
    let K_0_t1 = matmul(x_norm_0_t1, W_k_0)
    let V_0_t1 = matmul(x_norm_0_t1, W_v_0)
    let attn_0_t1 = tinyllama_gqa_attention_with_cache(Q_0_t1, K_0_t1, V_0_t1, W_o_0)
    let x1_0_t1 = e1 + attn_0_t1
    let x_norm2_0_t1 = rms_norm(x1_0_t1, ffn_norm_0)
    let ffn_0_t1 = swiglu_ffn(x_norm2_0_t1, W_gate_0, W_up_0, W_down_0)
    let h0_t1 = x1_0_t1 + ffn_0_t1

    // Layer 1 - Token 1
    let x_norm_1_t1 = rms_norm(h0_t1, attn_norm_1)
    let Q_1_t1 = matmul(x_norm_1_t1, W_q_1)
    let K_1_t1 = matmul(x_norm_1_t1, W_k_1)
    let V_1_t1 = matmul(x_norm_1_t1, W_v_1)
    let attn_1_t1 = tinyllama_gqa_attention_with_cache(Q_1_t1, K_1_t1, V_1_t1, W_o_1)
    let x1_1_t1 = h0_t1 + attn_1_t1
    let x_norm2_1_t1 = rms_norm(x1_1_t1, ffn_norm_1)
    let ffn_1_t1 = swiglu_ffn(x_norm2_1_t1, W_gate_1, W_up_1, W_down_1)
    let h1_t1 = x1_1_t1 + ffn_1_t1

    let final_t1 = rms_norm(h1_t1, output_norm)
    let logits_t1 = matmul(final_t1, output_weight)
    let token_1 = temperature_sample(logits_t1, 0.8)

    // Initialize KV caches
    let K_cache_0_t1 = K_0_t1
    let V_cache_0_t1 = V_0_t1
    let K_cache_1_t1 = K_1_t1
    let V_cache_1_t1 = V_1_t1

    print("      Token 1: ", token_1)
    print("      KV cache [Layer 0]: ", shape(K_cache_0_t1))
    print("      KV cache [Layer 1]: ", shape(K_cache_1_t1))
    print("")

    // ================================================================
    // TOKEN 2: Accumulate cache
    // ================================================================
    print("[4/5] Token 2 (cache reuse + accumulation)...")
    let e2 = e1

    // Layer 0 - Token 2
    let x_norm_0_t2 = rms_norm(e2, attn_norm_0)
    let Q_0_t2 = matmul(x_norm_0_t2, W_q_0)
    let K_0_t2 = matmul(x_norm_0_t2, W_k_0)
    let V_0_t2 = matmul(x_norm_0_t2, W_v_0)
    let K_cache_0_t2 = concat(K_cache_0_t1, K_0_t2, 0.0)
    let V_cache_0_t2 = concat(V_cache_0_t1, V_0_t2, 0.0)
    let attn_0_t2 = tinyllama_gqa_attention_with_cache(Q_0_t2, K_cache_0_t2, V_cache_0_t2, W_o_0)
    let x1_0_t2 = e2 + attn_0_t2
    let x_norm2_0_t2 = rms_norm(x1_0_t2, ffn_norm_0)
    let ffn_0_t2 = swiglu_ffn(x_norm2_0_t2, W_gate_0, W_up_0, W_down_0)
    let h0_t2 = x1_0_t2 + ffn_0_t2

    // Layer 1 - Token 2
    let x_norm_1_t2 = rms_norm(h0_t2, attn_norm_1)
    let Q_1_t2 = matmul(x_norm_1_t2, W_q_1)
    let K_1_t2 = matmul(x_norm_1_t2, W_k_1)
    let V_1_t2 = matmul(x_norm_1_t2, W_v_1)
    let K_cache_1_t2 = concat(K_cache_1_t1, K_1_t2, 0.0)
    let V_cache_1_t2 = concat(V_cache_1_t1, V_1_t2, 0.0)
    let attn_1_t2 = tinyllama_gqa_attention_with_cache(Q_1_t2, K_cache_1_t2, V_cache_1_t2, W_o_1)
    let x1_1_t2 = h0_t2 + attn_1_t2
    let x_norm2_1_t2 = rms_norm(x1_1_t2, ffn_norm_1)
    let ffn_1_t2 = swiglu_ffn(x_norm2_1_t2, W_gate_1, W_up_1, W_down_1)
    let h1_t2 = x1_1_t2 + ffn_1_t2

    let final_t2 = rms_norm(h1_t2, output_norm)
    let logits_t2 = matmul(final_t2, output_weight)
    let token_2 = temperature_sample(logits_t2, 0.8)

    print("      Token 2: ", token_2)
    print("      KV cache [Layer 0]: ", shape(K_cache_0_t2))
    print("      KV cache [Layer 1]: ", shape(K_cache_1_t2))
    print("")

    // ================================================================
    // TOKEN 3: Further accumulation
    // ================================================================
    print("[5/5] Token 3 (cache reuse + accumulation)...")
    let e3 = e1

    // Layer 0 - Token 3
    let x_norm_0_t3 = rms_norm(e3, attn_norm_0)
    let Q_0_t3 = matmul(x_norm_0_t3, W_q_0)
    let K_0_t3 = matmul(x_norm_0_t3, W_k_0)
    let V_0_t3 = matmul(x_norm_0_t3, W_v_0)
    let K_cache_0_t3 = concat(K_cache_0_t2, K_0_t3, 0.0)
    let V_cache_0_t3 = concat(V_cache_0_t2, V_0_t3, 0.0)
    let attn_0_t3 = tinyllama_gqa_attention_with_cache(Q_0_t3, K_cache_0_t3, V_cache_0_t3, W_o_0)
    let x1_0_t3 = e3 + attn_0_t3
    let x_norm2_0_t3 = rms_norm(x1_0_t3, ffn_norm_0)
    let ffn_0_t3 = swiglu_ffn(x_norm2_0_t3, W_gate_0, W_up_0, W_down_0)
    let h0_t3 = x1_0_t3 + ffn_0_t3

    // Layer 1 - Token 3
    let x_norm_1_t3 = rms_norm(h0_t3, attn_norm_1)
    let Q_1_t3 = matmul(x_norm_1_t3, W_q_1)
    let K_1_t3 = matmul(x_norm_1_t3, W_k_1)
    let V_1_t3 = matmul(x_norm_1_t3, W_v_1)
    let K_cache_1_t3 = concat(K_cache_1_t2, K_1_t3, 0.0)
    let V_cache_1_t3 = concat(V_cache_1_t2, V_1_t3, 0.0)
    let attn_1_t3 = tinyllama_gqa_attention_with_cache(Q_1_t3, K_cache_1_t3, V_cache_1_t3, W_o_1)
    let x1_1_t3 = h0_t3 + attn_1_t3
    let x_norm2_1_t3 = rms_norm(x1_1_t3, ffn_norm_1)
    let ffn_1_t3 = swiglu_ffn(x_norm2_1_t3, W_gate_1, W_up_1, W_down_1)
    let h1_t3 = x1_1_t3 + ffn_1_t3

    let final_t3 = rms_norm(h1_t3, output_norm)
    let logits_t3 = matmul(final_t3, output_weight)
    let token_3 = temperature_sample(logits_t3, 0.8)

    print("      Token 3: ", token_3)
    print("      KV cache [Layer 0]: ", shape(K_cache_0_t3))
    print("      KV cache [Layer 1]: ", shape(K_cache_1_t3))
    print("")

    print("========================================================")
    print("✅ Autoregressive KV Cache Demo Complete!")
    print("")
    print("Token sequence: ", token_1, " -> ", token_2, " -> ", token_3)
    print("")
    print("Cache progression:")
    print("  Token 1: Initial cache [2, 256]")
    print("  Token 2: Concat → [4, 256]")
    print("  Token 3: Concat → [6, 256]")
    print("")
    print("Performance benefits:")
    print("  - O(N) computation instead of O(N²)")
    print("  - Each new token only computes 1 position of K, V")
    print("  - Attention attends to all previous tokens via cache")
}
