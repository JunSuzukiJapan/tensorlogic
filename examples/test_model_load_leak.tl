// Test model loading for memory leaks

main {
    print("=== Model Load Memory Test ===")
    print("")

    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("    ✓ Model loaded")

    print("[2] Getting one tensor...")
    let emb = get_tensor(model, "token_embd.weight")
    print("    ✓ Got embedding tensor:", shape(emb))

    print("")
    print("=== Test Complete ===")
}
