// Simple test to debug silu function
// Test with small tensor to see if sigmoid and multiplication work

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Simple SiLU Test")
    print("================================================================================")
    print("")

    // Create a simple test tensor
    let test_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    let test_tensor = tensor(test_data, [1, 5])

    print("Step 1: Input tensor")
    print("  Shape:", shape(test_tensor))
    print("  Sum:", sum(test_tensor))
    print("")

    print("Step 2: Compute sigmoid(x)")
    let sigmoid_result = sigmoid(test_tensor)
    print("  Sigmoid sum:", sum(sigmoid_result))
    print("")

    print("Step 3: Compute x * sigmoid(x)")
    let silu_result = silu(test_tensor)
    print("  SiLU sum:", sum(silu_result))
    print("")

    // Expected results:
    // sigmoid(-2) ≈ 0.119, sigmoid(-1) ≈ 0.269, sigmoid(0) = 0.5
    // sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881
    //
    // silu(-2) = -2 * 0.119 ≈ -0.238
    // silu(-1) = -1 * 0.269 ≈ -0.269
    // silu(0) = 0
    // silu(1) = 1 * 0.731 ≈ 0.731
    // silu(2) = 2 * 0.881 ≈ 1.762
    //
    // Expected sum: -0.238 + -0.269 + 0 + 0.731 + 1.762 ≈ 1.986

    print("Expected SiLU sum: ~1.99")
    print("")
    print("================================================================================")
    print("✅ Simple test complete")
    print("================================================================================")
}
