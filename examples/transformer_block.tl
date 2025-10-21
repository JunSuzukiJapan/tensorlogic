// Simplified Transformer Block
// Demonstrates: Self-Attention + Layer Norm + Feed-Forward + Residual

main {
    // Input sequence: [seq_len, d_model] = [2, 4]
    tensor X: float16[2, 4] = [[1.0, 0.5, 0.0, 0.5],
                                [0.0, 1.0, 0.5, 0.0]]

    print("=== Transformer Block ===")
    print("Input X:", X)

    // === Self-Attention (simplified) ===
    // In real implementation: Q = X @ W_q, K = X @ W_k, V = X @ W_v
    // Here we use X directly as Q, K, V for simplicity

    // Compute attention scores: X @ X^T
    tensor X_T: float16[4, 2] = transpose(X)
    tensor scores: float16[2, 2] = X @ X_T

    print("Attention scores (X @ X^T):", scores)

    // Scale by sqrt(d_model)
    tensor d_model: float16[1] = [4.0]
    tensor sqrt_d: float16[1] = sqrt(d_model)
    tensor scale: float16[1] = [1.0] / sqrt_d

    // Apply softmax to first row (demonstration)
    tensor scores_0: float16[2] = [scores[0, 0], scores[0, 1]]
    tensor scores_1: float16[2] = [scores[1, 0], scores[1, 1]]

    tensor scaled_0: float16[2] = scores_0 * scale
    tensor scaled_1: float16[2] = scores_1 * scale

    tensor attn_weights_0: float16[2] = softmax(scaled_0)
    tensor attn_weights_1: float16[2] = softmax(scaled_1)

    print("Attention weights:")
    print("  Position 0:", attn_weights_0)
    print("  Position 1:", attn_weights_1)

    // Simplified attention output (using original X as values)
    // In reality: attn_output = attention_weights @ V
    tensor attn_out: float16[2, 4] = X  // Placeholder for demonstration

    // === Residual Connection + Layer Norm ===
    tensor residual_1: float16[2, 4] = X + attn_out

    print("After residual (X + attention):", residual_1)

    // Layer Normalization (simplified - normalize each position)
    // For demonstration, we'll show the concept
    tensor pos_0: float16[4] = [residual_1[0, 0], residual_1[0, 1],
                                 residual_1[0, 2], residual_1[0, 3]]

    tensor mean_0: float16[1] = mean(pos_0)

    print("Mean of position 0:", mean_0)
    print("Note: Full layer norm would compute variance and normalize")

    // === Feed-Forward Network ===
    // FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
    // Simplified: just apply a linear transformation

    tensor W1: float16[4, 4] learnable = [[0.5, 0.5, 0.0, 0.0],
                                           [0.0, 0.5, 0.5, 0.0],
                                           [0.0, 0.0, 0.5, 0.5],
                                           [0.5, 0.0, 0.0, 0.5]]

    // Apply FFN to normalized output (using residual_1 for demo)
    tensor ffn_out: float16[2, 4] = residual_1 @ W1

    print("FFN output:", ffn_out)

    // Apply ReLU activation
    tensor ffn_activated: float16[2, 4] = relu(ffn_out)

    print("FFN after ReLU:", ffn_activated)

    // === Final Residual Connection ===
    tensor output: float16[2, 4] = residual_1 + ffn_activated

    print("Final Transformer block output:", output)
    print("Note: This is a simplified demonstration")
    print("      Real Transformer uses proper Q/K/V projections,")
    print("      multi-head attention, and full layer normalization")
}
