// Simple test demonstrating TokenIds type
// Note: This is a demonstration - actual tokenization would require a tokenizer file

main {
    print("=== TokenIds Type Demo ===")
    print("")

    // Show what TokenIds would look like
    print("TokenIds is a dedicated type for storing token IDs")
    print("It stores Vec<u32> internally - proper integer array for token IDs")
    print("")

    print("Example usage:")
    print("  let tokenizer = load_tokenizer('path/to/tokenizer.json')")
    print("  let tokens = tokenize(tokenizer, 'Hello world', false)")
    print("  // tokens is now TokenIds type, e.g. TokenIds([15496, 995])")
    print("  let text = detokenize(tokenizer, tokens, false)")
    print("  // text is now 'Hello world'")
    print("")

    print("Benefits over f16 tensor:")
    print("  1. Semantically correct - token IDs are integers")
    print("  2. No precision loss from float conversion")
    print("  3. Efficient memory usage")
    print("  4. Clear intent in code")
}
