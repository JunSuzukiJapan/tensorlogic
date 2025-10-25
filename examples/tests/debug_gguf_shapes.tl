main {
    print("=== Debug GGUF Shape Loading ===")

    // This will trigger GGUF loading with debug info
    // We need to check the GGUF loader code to see what's happening

    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"

    print("Loading model (check console for GGUF loader debug output)...")
    let model = load_model(model_path)

    print("\nChecking actual tensor shapes:")
    let wk = get_tensor(model, "blk.0.attn_k.weight")
    print("blk.0.attn_k.weight shape:", shape(wk))

    let wgate = get_tensor(model, "blk.0.ffn_gate.weight")
    print("blk.0.ffn_gate.weight shape:", shape(wgate))
}
