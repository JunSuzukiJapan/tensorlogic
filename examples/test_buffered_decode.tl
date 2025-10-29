// Test buffered decoding

main {
    print("=== Testing Buffered Decoding ===")

    let home = env("HOME")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    // Initialize empty token array
    let generated_tokens = tokenizer.tokenize("", false)
    print("Initialized empty tokens")

    // Append a few test tokens
    let token1 = 15043  // "Hello"
    generated_tokens = generated_tokens.append_token(token1)
    let text1 = tokenizer.detokenize(generated_tokens, false)
    print("After token 1: {}", text1)

    let token2 = 29991  // " world"
    generated_tokens = generated_tokens.append_token(token2)
    let text2 = tokenizer.detokenize(generated_tokens, false)
    print("After token 2: {}", text2)

    print("=== Test Complete ===")
}
