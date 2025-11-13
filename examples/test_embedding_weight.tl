// Embedding weight検証テスト

main {
    print("================================================================================")
    print("Embedding Weight検証テスト")
    print("================================================================================")
    print("")

    // Load model
    print("[1] Loading f16 model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("    Embedding weight shape:", shape(emb_weight))
    print("")

    // Tokenize test to get TokenIds type
    print("[2] Loading tokenizer...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("    ✓ Loaded")
    print("")

    print("[3] Testing BOS token (ID=1) via single-token prompt...")
    // Use a prompt that tokenizes to just BOS
    let bos_only = tokenize(tokenizer, "", true)  // Empty string + BOS = just BOS
    print("    BOS-only tokens:", bos_only)
    print("    Num tokens:", len(bos_only))
    let bos_emb = embedding(emb_weight, bos_only)
    print("    BOS embedding shape:", shape(bos_emb))
    let bos_sum = sum(bos_emb)
    print("    BOS embedding sum:", bos_sum)
    print("")

    // Full prompt test
    print("[4] Full prompt embedding test...")
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello! How are you?</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, true)  // With BOS
    print("    Num tokens:", len(tokens))
    let full_emb = embedding(emb_weight, tokens)
    let full_sum = sum(full_emb)
    print("    Full embedding sum:", full_sum)
    print("    Candle reference:  3.6829965")
    let diff = full_sum - 3.6829965
    print("    Difference:", diff)
    print("")

    print("================================================================================")
    print("✅ Test Complete")
    print("================================================================================")
}
