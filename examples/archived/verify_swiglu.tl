// Verify SwiGLU FFN implementation

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    x * sig
}

main {
    print("=== SwiGLU FFN Verification ===")
    print("")

    // Test SiLU activation
    print("[Test 1] SiLU (Swish) activation")
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    let x_tensor = reshape(x, [1.0, 5.0])

    let silu_out = silu(x_tensor)
    print("  Input:", x)
    print("  SiLU output:", silu_out)

    // Manual calculation for silu(x) = x * sigmoid(x):
    // silu(-2) = -2 * sigmoid(-2) = -2 * 0.119 = -0.238
    // silu(-1) = -1 * sigmoid(-1) = -1 * 0.269 = -0.269
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.0
    // silu(1) = 1 * sigmoid(1) = 1 * 0.731 = 0.731
    // silu(2) = 2 * sigmoid(2) = 2 * 0.881 = 1.762
    print("  Expected: [-0.238, -0.269, 0.000, 0.731, 1.762]")
    print("")

    // Test element-wise multiplication
    print("[Test 2] Element-wise multiplication")
    let a = [1.0, 2.0, 3.0, 4.0]
    let b = [5.0, 6.0, 7.0, 8.0]
    let a_tensor = reshape(a, [1.0, 4.0])
    let b_tensor = reshape(b, [1.0, 4.0])

    let product = a_tensor * b_tensor
    print("  A:", a)
    print("  B:", b)
    print("  A * B:", product)
    print("  Expected: [5.0, 12.0, 21.0, 32.0]")
    print("")

    // Test full SwiGLU pattern: silu(gate) * up
    print("[Test 3] SwiGLU pattern: silu(gate) * up")
    let gate_vals = [1.0, 2.0, 3.0]
    let up_vals = [4.0, 5.0, 6.0]
    let gate_tensor = reshape(gate_vals, [1.0, 3.0])
    let up_tensor = reshape(up_vals, [1.0, 3.0])

    let gate_act = silu(gate_tensor)
    print("  Gate:", gate_vals)
    print("  Gate after SiLU:", gate_act)

    let swiglu_out = gate_act * up_tensor
    print("  Up:", up_vals)
    print("  SwiGLU output:", swiglu_out)

    // Manual calculation:
    // silu([1, 2, 3]) = [0.731, 1.762, 2.858]
    // [0.731, 1.762, 2.858] * [4, 5, 6] = [2.924, 8.810, 17.148]
    print("  Expected: [2.924, 8.810, 17.148]")
    print("")

    print("=== Verification Complete ===")
    print("")
    print("If outputs match expected values, SwiGLU is correct.")
}
