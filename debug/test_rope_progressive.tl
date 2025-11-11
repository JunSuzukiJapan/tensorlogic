// Test: Progressive token count to find where RoPE starts exploding

main {
    print("================================================================================")
    print("RoPE Progressive Token Count Test")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")

    // Test with different phrases to get different token counts
    let test_cases = [
        "Hello",                                    // ~1 token
        "Hello world",                              // ~2 tokens
        "Hello world today",                        // ~3 tokens
        "Hello world today is nice",               // ~5 tokens
        "The quick brown fox jumps over",          // ~6 tokens
        "The quick brown fox jumps over the lazy dog",  // ~9 tokens
        "The quick brown fox jumps over the lazy dog on a sunny day"  // ~14 tokens
    ]

    let i = 0
    while i < len(test_cases) {
        let text = test_cases[i]
        let tokens = tokenize(tokenizer, text, false)
        let num_tokens = len(tokens)

        let e = embedding(embed_table, tokens)
        let x_norm = rms_norm(e, attn_norm_0)
        let Q = linear(x_norm, W_q_0)
        let Q_heads = reshape(Q, [num_tokens, 32.0, 64.0])

        let before = sum(Q_heads)
        let Q_rope = rope(Q_heads)
        let after = sum(Q_rope)
        let ratio = after / before

        print("Tokens:", num_tokens, " | Before:", before, " | After:", after, " | Ratio:", ratio)

        i = i + 1
    }

    print("")
    print("================================================================================")
    print("Expected: Ratio should stay close to 1.0, not explode")
    print("================================================================================")
}
