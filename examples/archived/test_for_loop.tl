// Simple test for FOR loop with array indexing
// Demonstrates: for i in range(N) { model.blk[i].weight }

main {
    print("")
    print("=== Testing FOR Loop with Array Index ===")
    print("")

    // Load model
    print("[1/2] Loading model...")
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    print("      ✓ Model loaded")
    print("")

    // Test: Load weights in a for loop
    print("[2/2] Testing FOR loop: for i in range(5) { model.blk[i].weight }")
    print("")

    for i in range(5) {
        print("  [Loop iteration", i, "]")

        // Key test: Can we access model.blk[i] where i is a loop variable?
        let attn_norm = model.blk[i].attn_norm.weight
        let attn_q = model.blk[i].attn_q.weight
        let ffn_gate = model.blk[i].ffn_gate.weight

        print("    ✓ Loaded layer", i, "weights dynamically!")
    }

    print("")
    print("=== SUCCESS! ===")
    print("")
    print("💡 Achievement:")
    print("   - FOR loop variable 'i' works in array indexing")
    print("   - model.blk[i].weight syntax is fully functional")
    print("   - Dynamic layer loading is now possible!")
    print("")
    print("🚀 Next Step:")
    print("   - Add array/list type to TensorLogic")
    print("   - Store layer weights in arrays: weights[i] = model.blk[i]")
    print("   - Process all 22 layers in a loop!")
    print("")
}
