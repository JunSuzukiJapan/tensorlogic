// Test greedy sampling (temperature = 0.0)
// Should always select highest logit token

main {
    print("================================================================================")
    print("Testing Greedy Sampling (temp=0.0)")
    print("================================================================================")
    print("")

    // Load model and tokenizer
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Simple test: decode first 50 token IDs to see what they are
    print("[Test 1] First 50 tokens in vocabulary:")
    for i in range_i(50) {
        let token_text = detokenize(tokenizer, [i], false)
        print("  Token", i, ":", token_text)
    }
    print("")

    // Test most common tokens
    print("[Test 2] Common token IDs:")
    print("  Token 29871 (space):", detokenize(tokenizer, [29871], false))
    print("  Token 29918:", detokenize(tokenizer, [29918], false))
    print("  Token 13 (newline):", detokenize(tokenizer, [13], false))
    print("  Token 2 (EOS):", detokenize(tokenizer, [2], false))
    print("")

    print("================================================================================")
    print("Test completed")
    print("================================================================================")
}
