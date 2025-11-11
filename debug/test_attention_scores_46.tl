// Detailed test of Attention scores calculation with 46 tokens

main {
    print("=== Attention Scores Detailed Test (38 Tokens) ===")
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

    // 38 tokens
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("[0] Token count:", len(tokens))
    print("")

    // Get embeddings and Q, K projections
    let emb = embedding(emb_weight, tokens)
    let normed = rms_norm(emb, L0_attn_norm)
    let Q = linear(normed, L0_W_q)
    let K = linear(normed, L0_W_k)

    print("[1] Q and K projections")
    print("  Q sum:", sum(Q), " min:", min(Q), " max:", max(Q))
    print("  K sum:", sum(K), " min:", min(K), " max:", max(K))
    print("")

    // Reshape to heads
    let seq_len_tensor = shape(Q)
    let seq_len = seq_len_tensor[0]
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])

    print("[2] Reshaped to heads")
    print("  Q_heads sum:", sum(Q_heads), " min:", min(Q_heads), " max:", max(Q_heads))
    print("  K_heads sum:", sum(K_heads), " min:", min(K_heads), " max:", max(K_heads))
    print("")

    // Apply RoPE
    print("[3] Applying RoPE")
    let Q_rope = rope(Q_heads)
    print("  Q_rope sum:", sum(Q_rope), " min:", min(Q_rope), " max:", max(Q_rope))

    let K_rope = rope(K_heads)
    print("  K_rope sum:", sum(K_rope), " min:", min(K_rope), " max:", max(K_rope))
    print("")

    // GQA: Expand K from 4 to 32 heads
    print("[4] GQA expansion")
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    print("  K_with_group sum:", sum(K_with_group), " min:", min(K_with_group), " max:", max(K_with_group))

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    print("  K_broadcast sum:", sum(K_broadcast), " min:", min(K_broadcast), " max:", max(K_broadcast))

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    print("  K_expanded sum:", sum(K_expanded), " min:", min(K_expanded), " max:", max(K_expanded))
    print("")

    // Attention computation
    print("[5] Attention scores computation")
    print("  Computing einsum('ihd,jhd->ihj', Q_rope, K_expanded)...")
    print("  Q_rope shape: [seq_len, 32, 64]")
    print("  K_expanded shape: [seq_len, 32, 64]")
    print("  Expected output shape: [seq_len, 32, seq_len]")
    print("")

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("  Attention scores computed!")
    print("  scores sum:", sum(scores))
    print("  scores min:", min(scores), " max:", max(scores))
    print("")

    print("=== Test Complete ===")
}
