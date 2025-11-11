// Minimal test to debug zero logits issue
// Tests: embedding → rms_norm → linear chain

main {
    print("=== Testing logits generation ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight

    print("1. Model loaded successfully")
    print("   tok_embd shape: {}", shape(tok_embd))
    print("   output_norm shape: {}", shape(output_norm))

    // Simple prompt: just "Hello"
    let tokens = tokenizer.tokenize("Hello", true)
    print("")
    print("2. Tokenized 'Hello'")

    // Embedding
    let x = embedding(tok_embd, tokens)
    print("   embedding shape: {}", shape(x))

    // RMS norm
    let normed = rms_norm(x, output_norm)
    print("   after rms_norm shape: {}", shape(normed))

    // Linear projection (weight tying)
    let logits = linear(normed, tok_embd)
    print("   logits shape: {}", shape(logits))

    // Sample to see what token we get
    let token_id = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, token_id, false)

    print("")
    print("3. Generated token: '{}' (id={})", text, token_id)
    print("")
    print("=== Test complete ===")
}
