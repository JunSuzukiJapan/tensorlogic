// KV Cache Demo - 2 Token Generation with Cache Reuse
// Demonstrates concat-based KV cache accumulation

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

    print("      [Attention] Q seq_len: ", seq_len_f, ", KV cache len: ", cache_len_f)

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
    result = x1 + ffn_out
}

main {
    print("=== KV Cache Demo (2 Tokens with Cache Reuse) ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Load 2 layers for quick testing
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

    // Tokenize
    print("[2/5] Tokenizing...")
    let user_input = "Hi"
    let tokens = tokenize(tokenizer, user_input, true)
    print("      Tokens: ", tokens)
    print("")

    // ================================================================
    // TOKEN 1: Initial generation (no cache)
    // ================================================================
    print("[3/5] Generating Token 1 (no cache)...")
    let e1 = embedding(embed_table, tokens)
    print("      Embedding shape: ", shape(e1))

    // Layer 0 - Token 1
    let h0_t1 = transformer_layer(e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)

    // Layer 1 - Token 1
    let h1_t1 = transformer_layer(h0_t1, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)

    let final_norm_t1 = rms_norm(h1_t1, output_norm)
    let logits_t1 = matmul(final_norm_t1, output_weight)
    let token_1 = temperature_sample(logits_t1, 0.0)

    print("      Generated token 1: ", token_1)
    print("")

    // ================================================================
    // Extract KV cache from Token 1
    // ================================================================
    print("[4/5] Extracting KV cache from Token 1...")

    // For simplicity, we'll compute K and V again for the first token
    // In a real implementation, these would be saved during forward pass
    let x_norm1_layer0 = rms_norm(e1, attn_norm_0)
    let K_cache_0_t1 = matmul(x_norm1_layer0, W_k_0)
    let V_cache_0_t1 = matmul(x_norm1_layer0, W_v_0)

    let x_norm1_layer1 = rms_norm(h0_t1, attn_norm_1)
    let K_cache_1_t1 = matmul(x_norm1_layer1, W_k_1)
    let V_cache_1_t1 = matmul(x_norm1_layer1, W_v_1)

    print("      Layer 0 K cache: ", shape(K_cache_0_t1))
    print("      Layer 0 V cache: ", shape(V_cache_0_t1))
    print("      Layer 1 K cache: ", shape(K_cache_1_t1))
    print("      Layer 1 V cache: ", shape(V_cache_1_t1))
    print("")

    // ================================================================
    // TOKEN 2: Generation with KV cache reuse
    // ================================================================
    print("[5/5] Generating Token 2 (WITH KV cache reuse)...")

    // For demo purposes, use the same input embedding
    // In a real implementation, this would be the embedding of token_1
    let e2 = e1
    print("      Token 2 embedding shape: ", shape(e2))

    // Layer 0 - Token 2 with cache
    print("    [Layer 0]")
    let x_norm1_layer0_t2 = rms_norm(e2, attn_norm_0)
    let Q_0_t2 = matmul(x_norm1_layer0_t2, W_q_0)
    let K_0_t2 = matmul(x_norm1_layer0_t2, W_k_0)
    let V_0_t2 = matmul(x_norm1_layer0_t2, W_v_0)

    // Concatenate with cache
    let K_cache_0_all = concat(K_cache_0_t1, K_0_t2, 0.0)
    let V_cache_0_all = concat(V_cache_0_t1, V_0_t2, 0.0)

    print("      After concat - K cache: ", shape(K_cache_0_all))
    print("      After concat - V cache: ", shape(V_cache_0_all))

    // Attention with full cache
    let attn_out_0_t2 = tinyllama_gqa_attention_with_cache(Q_0_t2, K_cache_0_all, V_cache_0_all, W_o_0)
    let x1_0_t2 = e2 + attn_out_0_t2
    let x_norm2_layer0_t2 = rms_norm(x1_0_t2, ffn_norm_0)
    let ffn_out_0_t2 = swiglu_ffn(x_norm2_layer0_t2, W_gate_0, W_up_0, W_down_0)
    let h0_t2 = x1_0_t2 + ffn_out_0_t2

    // Layer 1 - Token 2 with cache
    print("    [Layer 1]")
    let x_norm1_layer1_t2 = rms_norm(h0_t2, attn_norm_1)
    let Q_1_t2 = matmul(x_norm1_layer1_t2, W_q_1)
    let K_1_t2 = matmul(x_norm1_layer1_t2, W_k_1)
    let V_1_t2 = matmul(x_norm1_layer1_t2, W_v_1)

    // Concatenate with cache
    let K_cache_1_all = concat(K_cache_1_t1, K_1_t2, 0.0)
    let V_cache_1_all = concat(V_cache_1_t1, V_1_t2, 0.0)

    print("      After concat - K cache: ", shape(K_cache_1_all))
    print("      After concat - V cache: ", shape(V_cache_1_all))

    // Attention with full cache
    let attn_out_1_t2 = tinyllama_gqa_attention_with_cache(Q_1_t2, K_cache_1_all, V_cache_1_all, W_o_1)
    let x1_1_t2 = h0_t2 + attn_out_1_t2
    let x_norm2_layer1_t2 = rms_norm(x1_1_t2, ffn_norm_1)
    let ffn_out_1_t2 = swiglu_ffn(x_norm2_layer1_t2, W_gate_1, W_up_1, W_down_1)
    let h1_t2 = x1_1_t2 + ffn_out_1_t2

    let final_norm_t2 = rms_norm(h1_t2, output_norm)
    let logits_t2 = matmul(final_norm_t2, output_weight)
    let token_2 = temperature_sample(logits_t2, 0.0)

    print("      Generated token 2: ", token_2)
    print("")

    print("========================================================")
    print("")
    print("✅ KV Cache Demo Complete!")
    print("")
    print("Token sequence: ", token_1, " -> ", token_2)
    print("")
    print("Performance benefit:")
    print("  - Token 1: Compute K, V for all positions")
    print("  - Token 2: Compute ONLY new K, V, concat with cache")
    print("  - Attention: Reuse cached K, V from previous tokens")
    print("")
    print("Memory efficiency:")
    print("  - Without cache: O(seq_len²) computation")
    print("  - With cache: O(seq_len) incremental computation")
    print("")
}
