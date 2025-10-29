// Simple embedding test to debug hang issue

main {
    print("=== Testing Embedding Function ===")

    // Load model
    print("[1/3] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model_f32(model_path)
    print("      ✓ Model loaded")

    // Get embedding table
    print("[2/3] Getting embedding table...")
    let embed_table = model.token_embd.weight
    print("      ✓ Got embedding table")

    // Test embedding with tokenizer
    print("[3/3] Testing embedding lookup...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let test_text = "Hello"
    print("      Test text:", test_text)

    let test_tokens = tokenize(tokenizer, test_text, false)
    print("      Tokens:", test_tokens)

    print("      Calling embedding()...")
    let result = embedding(embed_table, test_tokens)
    print("      ✓ Embedding successful!")
    // Note: shape() function calls to_vec() which panics for GPU tensors
    // print("      Result shape:", shape(result))

    print("=== Test Complete ===")
}
