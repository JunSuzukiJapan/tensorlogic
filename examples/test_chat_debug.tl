// Minimal test to debug chat demo issue

main {
    print("=== Chat Demo Debug Test ===")
    print("")

    // Step 1: Load model
    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("    ✓ Model loaded")

    // Step 2: Load tokenizer
    print("[2] Loading tokenizer...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("    ✓ Tokenizer loaded")

    // Step 3: Get embedding weight
    print("[3] Getting embedding weight...")
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Embedding weight loaded")

    // Step 4: Tokenize a simple prompt
    print("[4] Tokenizing prompt...")
    let prompt = "<|system|>\nYou are helpful.</s>\n<|user|>\nHi</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    let token_count = len(tokens)
    print("    ✓ Tokens:", token_count)

    // Step 5: Try embedding
    print("[5] Testing embedding with", token_count, "tokens...")
    let embeddings = embedding(emb_weight, tokens)
    print("    ✓ Embeddings computed")

    // Step 6: Check embedding shape
    let emb_shape = shape(embeddings)
    print("    Shape:", emb_shape)

    print("")
    print("=== All tests passed! ===")
}
