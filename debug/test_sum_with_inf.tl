// Test: Check sum() behavior with inf values

main {
    print("================================================================================")
    print("Test sum() with inf values")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let tokens = tokenize(tokenizer, "Hello", false)
    let e = embedding(embed_table, tokens)

    // Get first layer weights to do minimal computation
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")

    // Simple forward pass
    let Q = linear(e, W_q_0)

    print("Q shape:", shape(Q))
    print("Q sum:", sum(Q))
    print("")

    // Apply final norm and linear (like logits computation)
    let normed = rms_norm(Q, output_norm)
    print("After RMS norm sum:", sum(normed))

    let logits = linear(normed, output_weight)
    print("Logits sum:", sum(logits))
    print("Logits shape:", shape(logits))
    print("")

    // Try to sample from it
    print("Attempting temperature_sample...")
    let sampled = temperature_sample(logits, 0.7)
    print("Sampled token:", sampled)

    print("================================================================================")
}
