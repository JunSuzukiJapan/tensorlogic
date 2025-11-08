// Test minimal prefill to isolate the issue

main {
    print("=== Minimal Prefill Test ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight

    // Simple prompt
    let tokens = tokenizer.tokenize("Hello!", true)
    let x = embedding(tok_embd, tokens)

    print("1. Embedding shape: {}", shape(x))

    // Skip transformer layers - direct to output
    let final_norm = rms_norm(x, output_norm)
    print("2. After rms_norm shape: {}", shape(final_norm))

    // Extract last token
    let last_token = slice_last(final_norm, 0)
    print("3. Last token shape: {}", shape(last_token))

    let last_token_2d = reshape(last_token, [1.0, 2048.0])
    print("4. Reshaped shape: {}", shape(last_token_2d))

    // Project to vocabulary
    let logits = linear(last_token_2d, tok_embd)
    print("5. Logits shape: {}", shape(logits))

    // Sample
    let token_id = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, token_id, false)

    print("")
    print("Generated: '{}' (id={}))", text, token_id)
    print("")

    if token_id == 0 {
        print("❌ FAILED: Zero token (zero logits)")
    } else {
        print("✓ SUCCESS: Non-zero token")
    }
}
