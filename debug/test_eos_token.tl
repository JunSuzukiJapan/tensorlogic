// Test: Check if EOS token causes computation issues

main {
    print("=== Testing EOS Token in Sequence ===")
    print("")

    // Load model and tokenizer
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Test 1: Check what tokens are in the 46-token prompt
    print("[1] Analyzing 46-token prompt")
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("    Token count:", len(tokens))
    print("    Token IDs:", tokens)
    print("")

    // Try to detokenize to see the text
    let decoded_text = detokenize(tokenizer, tokens, false)
    print("    Decoded text:", decoded_text)
    print("")

    // Test 2: Process the actual 46-token prompt through model
    print("[2] Processing 46-token prompt through model")
    let emb_46 = embedding(emb_weight, tokens)
    print("    Embedding shape:", shape(emb_46))

    let normed_46 = rms_norm(emb_46, output_norm_weight)
    let logits_46 = linear(normed_46, output_weight)

    let max_46 = max(logits_46)
    let min_46 = min(logits_46)
    print("    Logit range: min=", min_46, ", max=", max_46)
    print("")

    // Test 3: Now try with 45 tokens (one less)
    print("[3] Testing with 45 tokens")
    let short_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + "Hello! Tell me a short fun fact about." + "</s>\n<|assistant|>\n"
    let tokens_45 = tokenize(tokenizer, short_prompt, false)
    print("    Token count:", len(tokens_45))

    let emb_45 = embedding(emb_weight, tokens_45)
    let normed_45 = rms_norm(emb_45, output_norm_weight)
    let logits_45 = linear(normed_45, output_weight)

    let max_45 = max(logits_45)
    let min_45 = min(logits_45)
    print("    Logit range: min=", min_45, ", max=", max_45)
    print("")

    // Test 4: Check if it's related to transformer layer processing
    print("[4] Testing 46 tokens through ONE transformer layer")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    let W_k = get_tensor(model, "blk.0.attn_k.weight")
    let W_v = get_tensor(model, "blk.0.attn_v.weight")
    let W_o = get_tensor(model, "blk.0.attn_output.weight")

    let x_norm = rms_norm(emb_46, attn_norm)
    print("    After attn_norm: min=", min(x_norm), " max=", max(x_norm))

    let Q = linear(x_norm, W_q)
    print("    After Q projection: min=", min(Q), " max=", max(Q))

    let K = linear(x_norm, W_k)
    let V = linear(x_norm, W_v)

    // Check if reshape works
    let seq_len_tensor = shape(Q)
    let seq_len = seq_len_tensor[0]
    print("    seq_len:", seq_len)

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    print("    After reshape Q: min=", min(Q_heads), " max=", max(Q_heads))

    // Apply RoPE
    print("    Applying RoPE to Q...")
    let Q_rope = rope(Q_heads)
    print("    After RoPE Q: min=", min(Q_rope), " max=", max(Q_rope))

    let K_heads = reshape(K, [seq_len, 4, 64])
    let K_rope = rope(K_heads)
    print("    After RoPE K: min=", min(K_rope), " max=", max(K_rope))

    let V_heads = reshape(V, [seq_len, 4, 64])
    print("    V_heads (no RoPE): min=", min(V_heads), " max=", max(V_heads))
    print("")

    // Try attention computation
    print("[5] Testing attention with 46 tokens")
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    // Attention scores
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("    Attention scores: min=", min(scores), " max=", max(scores))

    let scaled_scores = scores * 0.125
    print("    Scaled scores: min=", min(scaled_scores), " max=", max(scaled_scores))
    print("")

    print("=== Test Complete ===")
}
