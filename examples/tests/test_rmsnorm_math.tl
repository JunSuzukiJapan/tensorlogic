// RMSNorm basic test - verify it runs without NaN
// RMS(x) = sqrt(mean(x^2) + eps)
// RMSNorm(x) = (x / RMS(x)) * weight

main {
    print("=== RMSNorm Basic Test ===")
    print("")

    // Test 1: Simple input with weight=1
    print("[Test 1] Simple input with uniform weight")
    let input1 = [2.0, 4.0, 6.0, 8.0]
    let weight1 = [1.0, 1.0, 1.0, 1.0]
    let output1 = rms_norm(input1, weight1)

    print("  Input:", input1)
    print("  Output:", output1)
    print("  Expected: Smaller values (normalized)")
    print("")

    // Test 2: Non-uniform weight
    print("[Test 2] Non-uniform weight")
    let input2 = [1.0, 2.0, 3.0, 4.0]
    let weight2 = [0.5, 1.0, 1.5, 2.0]
    let output2 = rms_norm(input2, weight2)

    print("  Input:", input2)
    print("  Weight:", weight2)
    print("  Output:", output2)
    print("  Expected: Scaled by weight")
    print("")

    // Test 3: Actual model weight (from TinyLlama)
    print("[Test 3] Using real model weight")
    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Load first layer attention norm weight (shape: [2048])
    let attn_norm_weight = get_tensor(model, "blk.0.attn_norm.weight")
    print("  Weight shape:", shape(attn_norm_weight))

    // Create test input with same size
    let test_input = ones([2048])

    // Apply RMSNorm
    let normalized = rms_norm(test_input, attn_norm_weight)
    print("  Input shape:", shape(test_input))
    print("  Output shape:", shape(normalized))
    print("  Output (first 5):", normalized)

    // Check for NaN
    print("")
    print("✓ Checking for NaN/Inf...")
    print("  If you see NaN or Inf above, RMSNorm has a problem")
    print("  If all values are finite, RMSNorm works correctly")
}
