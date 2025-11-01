// ============================================================================
// Chat Demo Function Tests
// ============================================================================
//
// Tests for the core functions used in chat_full_22layers_f16.tl
//

// SiLU activation function (used in SwiGLU)
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

// Test 1: RoPE (Rotary Position Embedding) preserves shape
test rope_shape {
    print("Test: RoPE preserves tensor shape")

    // Create simple test input: [seq_len=1, n_heads=1, head_dim=4]
    tensor test_input: float16[1, 1, 4] = [[[3.0, 4.0, 5.0, 12.0]]]

    let result = rope(test_input, 0.0)
    let result_shape = shape(result)

    // Verify shape is preserved
    // Note: Checking via print since assert() is not implemented
    print("  Input shape: [1, 1, 4]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], ", ", result_shape[2], "]")
    print("  ✓ RoPE executed successfully")
}

// Test 2: RMS Norm produces normalized output
test rms_norm_shape {
    print("Test: RMS Norm shape preservation")

    tensor x: float16[1, 4] = [[2.0, 4.0, 6.0, 8.0]]
    tensor weight: float16[4] = [1.0, 1.0, 1.0, 1.0]

    let result = rms_norm(x, weight)
    let result_shape = shape(result)

    print("  Input shape: [1, 4]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ RMS Norm executed successfully")
}

// Test 3: Softmax shape preservation
test softmax_shape {
    print("Test: Softmax shape preservation")

    tensor logits: float16[1, 4] = [[1.0, 2.0, 3.0, 4.0]]
    let probs = softmax(logits)
    let probs_shape = shape(probs)

    print("  Input shape: [1, 4]")
    print("  Output shape: [", probs_shape[0], ", ", probs_shape[1], "]")
    print("  ✓ Softmax executed successfully")
}

// Test 4: Linear layer (uses transposed matmul internally)
test linear_shape {
    print("Test: Linear layer shape")

    // linear() does: x @ weight.T
    // x: [2, 3], weight: [4, 3] -> output: [2, 4]
    tensor x: float16[2, 3] = [[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]
    tensor weight: float16[4, 3] = [[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]]

    let result = linear(x, weight)
    let result_shape = shape(result)

    print("  Input shape: [2, 3]")
    print("  Weight shape: [4, 3]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ Linear executed successfully")
}

// Test 5: Reshape operation
test reshape_shape {
    print("Test: Reshape operation")

    tensor x: float16[2, 3] = [[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]
    let x_shape = shape(x)

    let reshaped = reshape(x, [3.0, 2.0])
    let new_shape = shape(reshaped)

    print("  Original shape: [", x_shape[0], ", ", x_shape[1], "]")
    print("  Reshaped to: [", new_shape[0], ", ", new_shape[1], "]")
    print("  ✓ Reshape executed successfully")
}

// Test 6: SiLU activation
test silu_shape {
    print("Test: SiLU activation")

    tensor x: float16[1, 5] = [[-2.0, -1.0, 0.0, 1.0, 2.0]]
    let result = silu(x)
    let result_shape = shape(result)

    print("  Input shape: [1, 5]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ SiLU executed successfully")
}

// Test 7: Attention output shape
test attention_shape {
    print("Test: Attention mechanism shape")

    // Q, K, V: [seq_len=2, n_embd=8]
    // W_o: [n_embd=8, n_embd=8]
    tensor q: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor k: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor v: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor w_o: float16[8, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    let result = attention_with_cache(q, k, v, w_o)
    let result_shape = shape(result)

    print("  Input Q/K/V shape: [2, 8]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ Attention executed successfully")
}
