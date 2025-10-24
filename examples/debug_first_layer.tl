// Minimal test to debug first layer with real weights
// Focus on embedding → layer 0 → logits path

fn simple_attention(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    // Get sequence length from input shape [seq, d_model]
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    // QKV projections
    let Q = matmul(x, W_q)
    let K = matmul(x, W_k)
    let V = matmul(x, W_v)

    print("    Q shape:", shape(Q))
    print("    K shape:", shape(K))
    print("    V shape:", shape(V))

    // Reshape to heads
    let Q_heads = reshape(Q, [seq_len, 32.0, 64.0])
    let K_heads = reshape(K, [seq_len, 4.0, 64.0])
    let V_heads = reshape(V, [seq_len, 4.0, 64.0])

    // Expand KV heads for GQA
    let K_with_group = reshape(K_heads, [seq_len, 4.0, 1.0, 64.0])
    let V_with_group = reshape(V_heads, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [seq_len, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len, 32.0, 64.0])

    print("    Q_heads shape:", shape(Q_heads))
    print("    K_expanded shape:", shape(K_expanded))
    print("    V_expanded shape:", shape(V_expanded))

    // Attention
    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    print("    scores shape:", shape(scores))

    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("    attn_output shape:", shape(attn_output))

    // Output projection
    let attn_reshaped = reshape(attn_output, [seq_len, 2048.0])
    print("    attn_reshaped shape:", shape(attn_reshaped))

    result := matmul(attn_reshaped, W_o)
}

main {
    print("=== First Layer Debug Test ===")
    print("")

    // Load model and tokenizer
    print("[1/5] Loading model...")
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("      ✓ Loaded")

    print("[2/5] Loading tokenizer...")
    let tokenizer_path = "/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Loaded")

    // Get layer 0 weights
    print("[3/5] Loading layer 0 weights...")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    let W_k = get_tensor(model, "blk.0.attn_k.weight")
    let W_v = get_tensor(model, "blk.0.attn_v.weight")
    let W_o = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")

    print("      W_q shape:", shape(W_q))
    print("      W_k shape:", shape(W_k))
    print("      W_v shape:", shape(W_v))
    print("      W_o shape:", shape(W_o))
    print("      attn_norm shape:", shape(attn_norm))

    // Tokenize
    print("[4/5] Tokenizing...")
    let prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nHello\n<|im_start|>assistant\n"
    let token_ids = tokenize(tokenizer, prompt, true)
    print("      Tokens:", token_ids)

    // Embedding
    let token_embd_weight = get_tensor(model, "token_embd.weight")
    let embeddings = embedding(token_embd_weight, token_ids)
    print("      Embedding shape:", shape(embeddings))

    // Run first layer attention
    print("[5/5] Running layer 0 attention...")
    let x_norm = rms_norm(embeddings, attn_norm)
    print("  Normalized input shape:", shape(x_norm))

    let attn_out = simple_attention(x_norm, W_q, W_k, W_v, W_o)
    print("  Attention output shape:", shape(attn_out))

    // Residual
    let h = embeddings + attn_out
    print("  After residual shape:", shape(h))

    print("")
    print("✅ First layer attention complete")
    print("")
    print("This confirms:")
    print("  - Model weights load correctly")
    print("  - Embeddings work")
    print("  - Attention shapes are correct")
    print("  - No crashes in forward pass")
}
