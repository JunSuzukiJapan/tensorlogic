// Test attention scores with 38 tokens (the working case)

fn gqa_attention_debug(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    print("  [GQA] seq_len:", seq_len)

    // Reshape to heads
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    print("  [GQA] Q_heads sum:", sum(Q_heads), " min:", min(Q_heads), " max:", max(Q_heads))

    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    // Apply RoPE to Q and K
    let Q_rope = rope(Q_heads)
    print("  [GQA] Q_rope sum:", sum(Q_rope), " min:", min(Q_rope), " max:", max(Q_rope))

    let K_rope = rope(K_heads)
    print("  [GQA] K_rope sum:", sum(K_rope), " min:", min(K_rope), " max:", max(K_rope))

    // GQA: Expand K, V from 4 to 32 heads
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    // Attention computation
    print("  [GQA] Computing scores: einsum(ihd,jhd->ihj)...")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("  [GQA] Attention scores sum:", sum(scores), " min:", min(scores), " max:", max(scores))

    let scaled_scores = scores * 0.125
    print("  [GQA] Scaled scores sum:", sum(scaled_scores), " min:", min(scaled_scores), " max:", max(scaled_scores))

    let attn_weights = softmax(scaled_scores, 2)
    print("  [GQA] Attention weights sum:", sum(attn_weights), " min:", min(attn_weights), " max:", max(attn_weights))

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("  [GQA] Attention output sum:", sum(attn_output), " min:", min(attn_output), " max:", max(attn_output))

    // Flatten and project
    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    linear(attn_reshaped, W_o)
}

main {
    print("=== Attention Scores Test: 38 Tokens ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let L0_attn_norm = model.blk[0].attn_norm.weight
    let L0_W_q = model.blk[0].attn_q.weight
    let L0_W_k = model.blk[0].attn_k.weight
    let L0_W_v = model.blk[0].attn_v.weight
    let L0_W_o = model.blk[0].attn_output.weight

    // 38-token prompt (the working case)
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("[0] Token count:", len(tokens))

    let emb = embedding(emb_weight, tokens)
    print("Embedding sum:", sum(emb))
    print("  min:", min(emb), " max:", max(emb))
    print("")

    // Process Layer 0 with detailed attention tracking
    print("=== Layer 0 Attention Details ===")
    let normed = rms_norm(emb, L0_attn_norm)
    let Q = linear(normed, L0_W_q)
    let K = linear(normed, L0_W_k)
    let V = linear(normed, L0_W_v)

    let attn_output = gqa_attention_debug(Q, K, V, L0_W_o)
    print("Attention final output sum:", sum(attn_output))
    print("")

    print("=== Expected: Scores should NOT be inf for 38 tokens ===")
    print("=== Test Complete ===")
}
