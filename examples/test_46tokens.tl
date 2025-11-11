// Test attention components with 46 tokens

main {
    print("=== Testing 46 Token Processing ===")
    print("")

    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Model loaded")

    print("[2] Creating 46 token input...")
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    let token_count = len(tokens)
    print("    ✓ Tokens:", token_count)

    let dummy_emb = embedding(emb_weight, tokens)
    print("    ✓ Created embeddings")

    print("[3] Testing causal mask...")
    let mask_2d = causal_mask(token_count)
    print("    ✓ Created causal mask")
    let mask_shape = shape(mask_2d)
    print("    Mask shape:", mask_shape)

    print("[4] Getting layer 0 weights...")
    let W_q = model.blk[0].attn_q.weight
    let W_k = model.blk[0].attn_k.weight
    let W_v = model.blk[0].attn_v.weight
    print("    ✓ Weights loaded")

    print("[5] Testing QKV projections...")
    let Q = linear(dummy_emb, W_q)
    print("    ✓ Q projection done")
    let K = linear(dummy_emb, W_k)
    print("    ✓ K projection done")
    let V = linear(dummy_emb, W_v)
    print("    ✓ V projection done")

    print("[6] Reshaping to heads...")
    let Q_heads = reshape(Q, [token_count, 32, 64])
    print("    ✓ Q reshaped")
    let K_heads = reshape(K, [token_count, 4, 64])
    print("    ✓ K reshaped")

    print("[7] Testing RoPE...")
    let Q_rope = rope(Q_heads)
    print("    ✓ Q RoPE done")
    let K_rope = rope(K_heads)
    print("    ✓ K RoPE done")

    print("[8] Testing GQA expansion...")
    let K_with_group = reshape(K_rope, [token_count, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [token_count, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [token_count, 32, 64])
    print("    ✓ GQA expansion done")

    print("[9] Testing einsum for attention scores...")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("    ✓ Einsum done")
    let scores_shape = shape(scores)
    print("    Scores shape:", scores_shape)

    print("[10] Testing mask application...")
    let mask_3d = reshape(mask_2d, [token_count, 1, token_count])
    let mask_broadcast = broadcast_to(mask_3d, [token_count, 32, token_count])
    let masked_scores = apply_attention_mask(scores, mask_broadcast)
    print("    ✓ Mask applied")

    print("[11] Testing softmax...")
    let attn_weights = softmax(masked_scores, 2)
    print("    ✓ Softmax done")

    print("")
    print("=== All 46-token tests passed! ===")
}
