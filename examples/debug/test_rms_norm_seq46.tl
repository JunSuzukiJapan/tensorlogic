// Test RMS norm at sequence length 46 to isolate NaN issue
// This is a minimal reproducer for the bug

main {
    print("Testing RMS norm at sequence length 46")
    print("========================================")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Same prompt as before
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\\n" + system_prompt + "</s>\\n<|user|>\\n" + user_message + "</s>\\n<|assistant|>\\n"
    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)

    print("Prompt tokens:", len(prompt_tokens))
    print("")

    // Add 10 dummy tokens to reach sequence length 46
    let test_tokens = prompt_tokens
    for i in range_i(10) {
        test_tokens = append(test_tokens, 100 + i)  // Arbitrary token IDs
    }

    let seq_len = len(test_tokens)
    print("Test sequence length:", seq_len)
    print("")

    // Get embedding
    let emb_weight = get_tensor(model, "token_embd.weight")
    let embeddings = embedding(emb_weight, test_tokens)

    print("Embedding shape:", shape(embeddings))
    let emb_sum = sum(embeddings)
    print("Embedding sum:", emb_sum)
    print("")

    // Get Layer 0 attention norm weight
    let attn_norm = model.blk[0].attn_norm.weight
    print("Attention norm weight shape:", shape(attn_norm))
    let norm_weight_sum = sum(attn_norm)
    print("Attention norm weight sum:", norm_weight_sum)
    print("")

    // Apply RMS norm
    print("Applying RMS norm...")
    let normalized = rms_norm(embeddings, attn_norm)

    print("After RMS norm shape:", shape(normalized))
    let norm_sum = sum(normalized)
    print("After RMS norm sum:", norm_sum)

    if is_nan(norm_sum) {
        print("")
        print("❌ BUG REPRODUCED: RMS norm output is NaN at sequence length", seq_len)

        // Additional debug info
        print("")
        print("Debug info:")
        print("  Input min:", min(embeddings))
        print("  Input max:", max(embeddings))
        print("  Weight min:", min(attn_norm))
        print("  Weight max:", max(attn_norm))
    } else {
        print("")
        print("✅ RMS norm works fine at sequence length", seq_len)
    }
}
