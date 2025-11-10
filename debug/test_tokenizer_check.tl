// Tokenizer verification - why 39 tokens instead of 38?

main {
    print("================================================================================")
    print("🔍 Tokenizer Verification - Token Count Investigation")
    print("================================================================================")
    print("")

    // Load tokenizer
    let tokenizer_path = "/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json"
    print("[1/3] Loading tokenizer...")
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // Create exact same prompt as HuggingFace
    print("[2/3] Creating prompt...")
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"

    let prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("      Prompt (raw):")
    print(prompt)
    print("")

    print("      Prompt (escaped):")
    print("      <|system|>\\nYou are a friendly chatbot.</s>\\n<|user|>\\nHello! How are you?</s>\\n<|assistant|>\\n")
    print("")

    // Tokenize
    print("[3/3] Tokenizing...")
    let tokens = tokenize(tokenizer, prompt, false)
    let num_tokens = len(tokens)

    print("      Number of tokens:", num_tokens)
    print("")

    print("      Token IDs:")
    print("      ", tokens)
    print("")

    // Compare with HuggingFace reference
    print("================================================================================")
    print("📊 Comparison with HuggingFace Reference")
    print("================================================================================")
    print("")

    print("  HuggingFace tokens: 38")
    print("  TensorLogic tokens:", num_tokens)
    print("")

    if num_tokens == 38 {
        print("  ✅ Token count matches!")
    }

    if num_tokens == 39 {
        print("  ❌ TensorLogic has 1 extra token")
        print("  Investigation needed:")
        print("    - Check if BOS token is added")
        print("    - Check tokenizer configuration")
        print("    - Compare token IDs with reference")
    }

    if num_tokens != 38 && num_tokens != 39 {
        print("  ❌ Unexpected token count:", num_tokens)
    }

    print("")
    print("  HuggingFace reference tokens:")
    print("  [529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 19780, 13563, 7451, 29889, 2, 29871, 13, 29966, 29989, 1792, 29989, 29958, 13, 10994, 29991, 1128, 526, 366, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13]")
    print("")

    print("  TensorLogic tokens:")
    print("  ", tokens)
    print("")

    print("================================================================================")
    print("✅ Tokenizer verification complete")
    print("================================================================================")
}
