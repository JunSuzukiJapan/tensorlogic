// Test token decoding to understand what tokens are being generated

main {
    print("================================================================================")
    print("Testing Token Decoding")
    print("================================================================================")
    print("")

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Test the tokens that were generated in chat demo
    print("[Test 1] Decode tokens from recent chat demo output:")

    // From the debug output: #1: token_id=29918
    let text1 = tokenize(tokenizer, "", false)
    let tokens1 = append(text1, 29918)
    let decoded1 = detokenize(tokenizer, tokens1, false)
    print("  Token 29918 alone:", decoded1)

    let tokens2 = append(text1, 29871)
    let decoded2 = detokenize(tokenizer, tokens2, false)
    print("  Token 29871 alone:", decoded2)

    let tokens3 = append(text1, 13)
    let decoded3 = detokenize(tokenizer, tokens3, false)
    print("  Token 13 alone:", decoded3)
    print("")

    // Test a known English phrase
    print("[Test 2] Encode and decode a known phrase:")
    let test_phrase = "Hello, how are you?"
    let encoded = tokenize(tokenizer, test_phrase, false)
    print("  Original:", test_phrase)
    print("  Token IDs:", encoded)
    let decoded = detokenize(tokenizer, encoded, false)
    print("  Decoded:", decoded)
    print("")

    // Test special tokens
    print("[Test 3] Special tokens:")
    let bos = tokenize(tokenizer, "", true)  // with BOS
    print("  BOS token IDs:", bos)

    print("")
    print("================================================================================")
    print("Test completed")
    print("================================================================================")
}
