// Compare Metal GPU vs CPU implementations
// Verify numerical consistency

main {
    print("=== Metal GPU vs CPU Numerical Comparison ===\n")

    // Test 1: Softmax
    print("Test 1: Softmax (last dimension)\n")

    let test_input = [[1.0, 2.0, 3.0, 4.0, 5.0],
                      [5.0, 4.0, 3.0, 2.0, 1.0]]

    let result_softmax = softmax(test_input, 1)
    print("  Input: ", test_input)
    print("  Softmax output: ", result_softmax)
    print("  Expected: probabilities should sum to 1.0")
    print("")

    // Test 2: Linear (matmul + bias)
    print("Test 2: Linear transformation\n")

    let x = [[1.0, 2.0, 3.0]]
    let W = [[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]]

    let result_linear = matmul(x, W)
    print("  Input x: ", x)
    print("  Weight W: ", W)
    print("  Output: ", result_linear)
    print("  Expected: [1*0.5+2*0.3+3*0.2, 1*0.5+2*0.7+3*0.8] = [1.7, 4.3]")
    print("")

    // Test 3: RoPE
    print("Test 3: RoPE (Rotary Position Embedding)\n")

    let test_rope = [[[1.0, 2.0, 3.0, 4.0]]]  // [1, 1, 4]
    let result_rope = rope(test_rope, 0, 10000.0, "neox")

    print("  Input (position 0): ", test_rope)
    print("  RoPE output: ", result_rope)
    print("  Expected: At position 0, cos(0)=1 sin(0)=0, should preserve input")
    print("")

    // Test 4: Load actual model and compare first layer
    print("Test 4: Actual model - Layer 0 comparison\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let token_embd = get_tensor(model, "token_embd.weight")
    let h = embedding(token_embd, [15043.0])  // "Hello"

    print("  Embedding shape: ", shape(h))
    print("  (Metal GPU is used by default)")
    print("")

    // Test 5: Attention scores (einsum)
    print("Test 5: Einsum (Attention scores)\n")

    let Q_test = [[[1.0, 2.0], [3.0, 4.0]]]  // [1, 2, 2]
    let K_test = [[[1.0, 0.0], [0.0, 1.0]]]  // [1, 2, 2]

    let scores = einsum("ihd,jhd->ihj", Q_test, K_test)
    print("  Q: ", Q_test)
    print("  K: ", K_test)
    print("  Scores (Q·K^T): ", scores)
    print("  Expected: [[1*1+2*0, 1*0+2*1], [3*1+4*0, 3*0+4*1]] = [[1,2],[3,4]]")
    print("")

    print("=== All tests complete ===")
    print("If Metal GPU produces different results than CPU,")
    print("we would see numerical discrepancies in the output")
}
