// Test: RoPE with ~5 tokens

main {
    print("================================================================================")
    print("RoPE Test with ~5 Tokens")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")

    let tokens = tokenize(tokenizer, "Hello world today is", false)
    print("Tokens:", tokens)
    print("Number of tokens:", len(tokens))
    print("")

    let e = embedding(embed_table, tokens)
    let x_norm = rms_norm(e, attn_norm_0)
    let Q = linear(x_norm, W_q_0)

    let num_tokens = len(tokens)
    let Q_heads = reshape(Q, [num_tokens, 32.0, 64.0])

    print("Q_heads sum (before RoPE):", sum(Q_heads))
    let Q_rope = rope(Q_heads)
    print("Q_rope sum (after RoPE):", sum(Q_rope))
    print("")

    print("================================================================================")
    print("Result:")
    print("  Before RoPE:", sum(Q_heads))
    print("  After RoPE:", sum(Q_rope))
    print("  Ratio:", sum(Q_rope) / sum(Q_heads))
    print("================================================================================")
}
