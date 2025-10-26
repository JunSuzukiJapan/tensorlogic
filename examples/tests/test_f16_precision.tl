// Test f16 precision impact on basic operations
//
// TensorLogic uses f16 throughout, while llama.cpp uses mixed f32/f16.
// This test checks if f16 precision causes significant differences.

main {
    print("=== f16 Precision Impact Test ===\n")

    // Test 1: Simple operations precision
    print("[Test 1] Basic Arithmetic Precision")

    let a = 1.0 / 3.0
    let b = 0.333333
    print("  1.0 / 3.0 = ", a)
    print("  Expected ~0.333...")
    print("")

    // Test 2: Softmax precision
    print("[Test 2] Softmax Precision")
    let logits = tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    let probs = softmax(logits, 0)
    print("  Input logits:  ", logits)
    print("  Softmax probs: ", probs)
    print("  Sum of probs:  ", sum(probs))
    print("")

    // Test 3: RMS Norm precision
    print("[Test 3] RMS Norm Precision")
    let x = tensor([1.0, 2.0, 3.0, 4.0])
    let weight = tensor([1.0, 1.0, 1.0, 1.0])
    let normed = rms_norm(x, weight)
    print("  Input:  ", x)
    print("  RMS Normed: ", normed)
    print("")

    // Test 4: Large value handling
    print("[Test 4] Large Value Handling")
    let large = tensor([100.0, 200.0, 300.0])
    let large_sum = sum(large)
    let large_mean = mean(large)
    print("  Large values: ", large)
    print("  Sum:  ", large_sum)
    print("  Mean: ", large_mean)
    print("")

    // Test 5: Small value handling
    print("[Test 5] Small Value Handling")
    let small = tensor([0.001, 0.002, 0.003])
    let small_sum = sum(small)
    let small_mean = mean(small)
    print("  Small values: ", small)
    print("  Sum:  ", small_sum)
    print("  Mean: ", small_mean)
    print("")

    // Test 6: Accumulation precision
    print("[Test 6] Accumulation Precision (1000 additions)")
    let ones = ones([1000.0])
    let accumulated = sum(ones)
    print("  Sum of 1000 ones: ", accumulated)
    print("  Expected: 1000.0")
    print("  Error: ", accumulated - 1000.0)
    print("")

    print("✅ Precision tests complete")
    print("\n📝 Note: f16 has ~3-4 decimal digits of precision")
    print("   Accumulated errors may appear in deep computations")
}
