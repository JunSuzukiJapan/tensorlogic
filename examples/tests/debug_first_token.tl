// Debug: Check hidden states before final linear layer
main {
    print("=== Debug: Single Layer + Final Linear ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Simple single token: BOS
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input:", bos_str)
    print("Tokens:", tokens)

    // Embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let e = embedding(embed_table, tokens)
    print("\nEmbedding shape:", shape(e))

    // Skip transformer layers for now, just apply final norm + linear
    print("\n=== Skip all layers, test final norm + linear directly ===")

    let output_norm = get_tensor(model, "output_norm.weight")
    let normed = rms_norm(e, output_norm)
    print("After output_norm:", shape(normed))

    let output_weight = get_tensor(model, "output.weight")
    let logits = linear(normed, output_weight)
    print("Logits shape:", shape(logits))

    let predicted = temperature_sample(logits, 0.0)
    print("\nPredicted token (greedy):", predicted)
    print("\n=== If this gives a reasonable prediction,")
    print("    then the issue is in the transformer layers ===")
}
