// Minimal reproduction of the original chat_demo_optimized.tl segfault
// This test recreates the exact scenario that caused the segfault

main {
    print("=== Minimal Segfault Reproduction Test ===")
    print("")

    // Step 1: Load model
    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    let embed_table = get_tensor(model, "token_embd.weight")
    print("    ✓ Model loaded")
    print("")

    // Step 2: Get a sample tensor (simulating Q, K, V projections)
    print("[2] Creating sample tensor...")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    print("    W_q shape:", shape(W_q))

    // Create a simple tensor to work with
    let test_input = slice(embed_table, 0.0, 0.0, 5.0)  // Get first 5 rows
    print("    Test input shape:", shape(test_input))
    print("")

    // Step 3: Extract seq_len from shape (Integer value)
    print("[3] Extracting seq_len from shape...")
    let Q_shape = shape(test_input)
    let seq_len = Q_shape[0]
    print("    seq_len =", seq_len, "(type: Integer)")
    print("")

    // Step 4: Try to create mixed-type array [seq_len, 32.0, 64.0]
    // This is EXACTLY what the original script did
    print("[4] Creating mixed-type array [seq_len, 32.0, 64.0]...")
    print("    Expected: Either error message OR segfault")
    print("")

    let mixed_shape = [seq_len, 32.0, 64.0]

    print("    ❌ ERROR: Should not reach here!")
    print("    mixed_shape =", mixed_shape)
    print("")

    // Step 5: If we somehow got here, try reshape
    print("[5] Attempting reshape...")
    let reshaped = reshape(test_input, mixed_shape)
    print("    Reshaped shape:", shape(reshaped))
    print("    ❌ ERROR: Should definitely not reach here!")
}
