// Test: FloatArray in reshape operations
// Check if floating point literals cause issues

main {
    print("=== FloatArray Reshape Test ===")
    print("")

    // Test 1: Pure IntArray (should work)
    print("Test 1: Pure IntArray [2, 3, 4]")
    let int_shape = [2, 3, 4]
    print("  Value:", int_shape)
    print("")

    // Test 2: Pure FloatArray (might cause issues)
    print("Test 2: Pure FloatArray [2.0, 3.0, 4.0]")
    let float_shape = [2.0, 3.0, 4.0]
    print("  Value:", float_shape)
    print("")

    // Test 3: Skip mixed type test (causes error)
    // print("Test 3: Mixed [10, 32.0, 64.0]")
    // let seq_len = 10
    // let mixed_shape = [seq_len, 32.0, 64.0]
    // print("  Value:", mixed_shape)
    // print("")

    // Test 4: Load model and get a real tensor to test FloatArray reshape
    print("Test 4: Load model tensor and reshape with FloatArray")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    let embed_table = get_tensor(model, "token_embd.weight")
    print("  Loaded embed_table, shape:", shape(embed_table))

    // Reshape the full table: 32000 * 2048 = 65,536,000 elements
    // Let's reshape to [2000, 16, 2048] = 65,536,000
    print("  Trying reshape with FloatArray [2000.0, 16.0, 2048.0]...")
    let reshaped = reshape(embed_table, [2000.0, 16.0, 2048.0])
    print("  Reshaped shape:", shape(reshaped))
    print("  ✅ Success! FloatArray works in reshape")
    print("")

    // Test 5: Now test the problematic mixed-type case
    print("Test 5: Test mixed-type array (Integer + Float)")
    let tensor_shape = shape(reshaped)
    let dim0 = tensor_shape[0]
    print("  dim0 from shape():", dim0, "(type: Integer)")

    print("  Attempting mixed array [dim0, 32.0, 64.0]...")
    print("  (Expected: 'mixed types' error)")
    let mixed_array = [dim0, 32.0, 64.0]
    print("  If you see this, mixed arrays work:", mixed_array)
    print("")

    print("=== All tests passed! ===")
}
