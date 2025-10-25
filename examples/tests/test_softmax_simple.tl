// Simple Softmax test - sum should equal 1.0

main {
    print("=== Softmax Test: Normalization Property ===")

    // Test 1: Simple vector
    let input1 = [1.0, 2.0, 3.0, 4.0]
    let probs1 = softmax(input1)
    let sum1 = sum(probs1)

    print("Input:", input1)
    print("Output:", probs1)
    print("Sum:", sum1)
    print("Expected: Monotonically increasing, sum=1.0")
    print("")

    // Test 2: Uniform input (all same)
    let input2 = [2.0, 2.0, 2.0, 2.0]
    let probs2 = softmax(input2)
    let sum2 = sum(probs2)

    print("Uniform input:", input2)
    print("Output:", probs2)
    print("Sum:", sum2)
    print("Expected: All equal ~0.25, sum=1.0")
}
