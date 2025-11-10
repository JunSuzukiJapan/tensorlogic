// Test to find EOS (End of Sequence) token ID for TinyLlama

main {
    print("=== EOS Token Detection Test ===")
    print("")

    // Load tokenizer
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("✓ Tokenizer loaded")
    print("")

    // Test common EOS strings
    print("Testing common EOS token strings:")
    print("")

    // Test 1: </s>
    let eos1 = "</s>"
    let tokens1 = tokenize(tokenizer, eos1, false)
    print("  \"</s>\" -> tokens:", tokens1)
    if len(tokens1) > 0 {
        let first_token = get(tokens1, 0)
        print("    First token ID:", first_token)
    }
    print("")

    // Test 2: <s>
    let bos = "<s>"
    let tokens2 = tokenize(tokenizer, bos, false)
    print("  \"<s>\" -> tokens:", tokens2)
    if len(tokens2) > 0 {
        let first_token = get(tokens2, 0)
        print("    First token ID:", first_token)
    }
    print("")

    // Test 3: With special tokens
    let test_text = "Hello"
    let tokens_with_special = tokenize(tokenizer, test_text, true)
    print("  \"Hello\" with special tokens:", tokens_with_special)
    if len(tokens_with_special) > 0 {
        let first_token = get(tokens_with_special, 0)
        print("    First token (likely BOS):", first_token)
    }
    print("")

    // Test 4: Check if token ID 2 is EOS (common in Llama models)
    print("Testing if token ID 2 is EOS:")
    let token_ids_with_2 = [1, 15043, 2]
    let decoded_with_2 = detokenize(tokenizer, token_ids_with_2, true)
    print("  Tokens [1, 15043, 2] decode to:", decoded_with_2)
    print("")

    print("Summary:")
    print("  - Token ID 1 is likely BOS (Beginning of Sequence)")
    print("  - Token ID 2 is likely EOS (End of Sequence)")
    print("")
}
