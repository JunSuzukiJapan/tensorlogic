// Test: RoPE with 39 tokens (same as test_one_layer.tl)

main {
    print("================================================================================")
    print("RoPE Test with 39 Tokens")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    // Get weights
    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")

    // Same prompt as test_one_layer.tl
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let chat_prompt = "<|system|>\\n" + system_prompt + "</s>\\n<|user|>\\n" + user_message + "</s>\\n<|assistant|>\\n"

    // With BOS token (39 tokens total)
    let tokens = tokenize(tokenizer, chat_prompt, true)
    let num_tokens = len(tokens)
    print("Tokens:", num_tokens)
    print("")

    let e = embedding(embed_table, tokens)
    print("Embedding sum:", sum(e))
    print("")

    let x_norm = rms_norm(e, attn_norm_0)
    print("After RMS norm sum:", sum(x_norm))
    print("")

    let Q = linear(x_norm, W_q_0)
    print("Q shape:", shape(Q))
    print("Q sum:", sum(Q))
    print("")

    // Reshape to [39, 32, 64] (39 tokens with BOS)
    let Q_heads = reshape(Q, [39.0, 32.0, 64.0])
    print("Q_heads shape:", shape(Q_heads))
    print("Q_heads sum (before RoPE):", sum(Q_heads))
    print("")

    // Apply RoPE
    print("Applying RoPE...")
    let Q_rope = rope(Q_heads)
    print("Q_rope shape:", shape(Q_rope))
    print("Q_rope sum (after RoPE):", sum(Q_rope))
    print("")

    print("================================================================================")
    print("Result:")
    print("  Q sum before RoPE:", sum(Q_heads))
    print("  Q sum after RoPE:", sum(Q_rope))
    print("  Amplification ratio:", sum(Q_rope) / sum(Q_heads))
    print("================================================================================")
}
