// Test tokenizer functionality

main {
    print("=== Tokenizer Test ===")
    print("")

    // Load tokenizer
    print("[1/4] Loading tokenizer...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // Test encoding
    print("[2/4] Testing tokenization...")
    let text = "Hello, how are you today?"
    print("      Input text: \"", text, "\"")

    let token_ids = tokenize(tokenizer, text, true)
    print("      Token IDs:", token_ids)
    print("")

    // Test decoding
    print("[3/4] Testing detokenization...")
    let decoded_text = detokenize(tokenizer, token_ids, true)
    print("      Decoded text: \"", decoded_text, "\"")
    print("")

    // Test round-trip
    print("[4/4] Verifying round-trip...")
    if text == decoded_text {
        print("      ✓ Round-trip successful!")
    } else {
        print("      ✗ Round-trip failed")
        print("        Original:", text)
        print("        Decoded: ", decoded_text)
    }
    print("")

    print("=== Additional Tests ===")
    print("")

    // Test multiple sentences
    let multi_text = "The quick brown fox jumps over the lazy dog. This is a test."
    print("Multi-sentence text:", multi_text)
    let multi_ids = tokenize(tokenizer, multi_text, true)
    print("Decoded:", detokenize(tokenizer, multi_ids, true))
    print("")

    print("✅ All tokenizer tests complete!")
}
