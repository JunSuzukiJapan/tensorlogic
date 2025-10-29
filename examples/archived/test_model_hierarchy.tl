// Test new object-oriented model access

main {
    print("=== Testing Object-Oriented Model Access ===")

    print("[1/3] Loading model...")
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    print("      ✓ Model loaded")

    print("[2/3] Testing property access...")
    let tok_embd = model.token_embd
    print("      model.token_embd =", tok_embd)

    print("[3/3] Testing layer access...")
    let layers = model.blk
    print("      model.blk =", layers)

    print("      Accessing layer 0...")
    let L0 = model.blk[0]
    print("      model.blk[0] =", L0)

    print("      Accessing attn_norm feature...")
    let attn_norm = model.blk[0].attn_norm
    print("      model.blk[0].attn_norm =", attn_norm)

    print("      Accessing weight tensor...")
    let weight = model.blk[0].attn_norm.weight
    print("      model.blk[0].attn_norm.weight =", weight)

    print("")
    print("=== Test Complete ===")
    print("✓ Object-oriented access working!")
    print("✓ Can assign intermediate objects to variables!")
    print("✓ No more string-based path resolution!")
}
