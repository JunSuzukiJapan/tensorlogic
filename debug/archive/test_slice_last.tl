// Test slice_last function implementation

main {
    print("=== Testing slice_last function ===")
    print("")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight

    // Create a multi-token prompt to get shape [seq_len, hidden]
    let tokens = tokenizer.tokenize("Hello world!", true)
    let x = embedding(tok_embd, tokens)

    print("1. Original embedding shape: {}", shape(x))
    let x_shape = shape(x)
    let seq_len = x_shape[0]
    print("   seq_len = {}", seq_len)

    // Apply RMS norm (maintains shape)
    let normed = rms_norm(x, output_norm)
    print("")
    print("2. After rms_norm shape: {}", shape(normed))

    // Extract last token using slice_last
    let last_token = slice_last(normed, 0)
    print("")
    print("3. After slice_last(normed, 0) shape: {}", shape(last_token))
    let last_shape = shape(last_token)
    print("   Expected: [2048], Got: {}", last_shape)

    // Reshape to 2D for linear (linear expects [batch, features])
    let last_token_2d = reshape(last_token, [1.0, 2048.0])
    print("")
    print("4. After reshape to 2D: {}", shape(last_token_2d))

    // Project to vocabulary
    let logits = linear(last_token_2d, tok_embd)
    print("")
    print("5. Logits shape: {}", shape(logits))
    let logits_shape = shape(logits)
    print("   Expected: [1, 32000], Got: {}", logits_shape)

    // Sample to verify non-zero logits
    let token_id = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, token_id, false)

    print("")
    print("6. Sampled token: '{}' (id={})", text, token_id)
    print("")

    if token_id == 0 {
        print("❌ FAILED: Got zero token (indicates zero logits)")
    } else {
        print("✓ SUCCESS: Non-zero token generated")
    }

    print("")
    print("=== Test complete ===")
}
