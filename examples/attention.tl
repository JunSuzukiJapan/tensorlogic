// Scaled Dot-Product Attention
// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

// Input dimensions
// seq_len = 3 (sequence length)
// d_k = 2 (key/query dimension)
// d_v = 2 (value dimension)

main {
    // Query matrix: [seq_len, d_k] = [3, 2]
    tensor Q: float16[3, 2] = [[1.0, 0.0],
                                [0.0, 1.0],
                                [1.0, 1.0]]

    // Key matrix: [seq_len, d_k] = [3, 2]
    tensor K: float16[3, 2] = [[1.0, 0.0],
                                [0.0, 1.0],
                                [0.5, 0.5]]

    // Value matrix: [seq_len, d_v] = [3, 2]
    tensor V: float16[3, 2] = [[2.0, 0.0],
                                [0.0, 2.0],
                                [1.0, 1.0]]

    print("=== Scaled Dot-Product Attention ===")
    print("Query (Q):", Q)
    print("Key (K):", K)
    print("Value (V):", V)

    // Step 1: Compute Q @ K^T
    // Result: [seq_len, seq_len] = [3, 3]
    tensor K_T: float16[2, 3] = transpose(K)
    tensor scores: float16[3, 3] = Q @ K_T

    print("Q @ K^T (attention scores):", scores)

    // Step 2: Scale by sqrt(d_k)
    tensor d_k: float16[1] = [2.0]
    tensor sqrt_d_k: float16[1] = sqrt(d_k)
    tensor scale: float16[1] = [1.0] / sqrt_d_k

    // Broadcast scale to match scores shape
    // Manual broadcasting: multiply each element
    tensor scores_row0: float16[3] = [scores[0, 0], scores[0, 1], scores[0, 2]]
    tensor scores_row1: float16[3] = [scores[1, 0], scores[1, 1], scores[1, 2]]
    tensor scores_row2: float16[3] = [scores[2, 0], scores[2, 1], scores[2, 2]]

    // Scale each row (simplified for demonstration)
    tensor scaled_0: float16[3] = scores_row0 * scale
    tensor scaled_1: float16[3] = scores_row1 * scale
    tensor scaled_2: float16[3] = scores_row2 * scale

    print("Scaled scores (Q @ K^T / sqrt(d_k)):")
    print("  Row 0:", scaled_0)
    print("  Row 1:", scaled_1)
    print("  Row 2:", scaled_2)

    // Step 3: Apply softmax to get attention weights
    // For simplicity, demonstrate with one row
    tensor weights_0: float16[3] = softmax(scaled_0)
    tensor weights_1: float16[3] = softmax(scaled_1)
    tensor weights_2: float16[3] = softmax(scaled_2)

    print("Attention weights (after softmax):")
    print("  Row 0:", weights_0)
    print("  Row 1:", weights_1)
    print("  Row 2:", weights_2)

    // Step 4: Weighted sum of values
    // For demonstration, compute output for first query position
    // output[0] = weights_0[0] * V[0] + weights_0[1] * V[1] + weights_0[2] * V[2]

    tensor V_0: float16[2] = [V[0, 0], V[0, 1]]
    tensor V_1: float16[2] = [V[1, 0], V[1, 1]]
    tensor V_2: float16[2] = [V[2, 0], V[2, 1]]

    // Weighted contributions (simplified scalar multiplication)
    tensor w0_0: float16[1] = [weights_0[0]]
    tensor w0_1: float16[1] = [weights_0[1]]
    tensor w0_2: float16[1] = [weights_0[2]]

    tensor contrib_0: float16[2] = V_0 * w0_0
    tensor contrib_1: float16[2] = V_1 * w0_1
    tensor contrib_2: float16[2] = V_2 * w0_2

    tensor output_0: float16[2] = contrib_0 + contrib_1 + contrib_2

    print("Attention output for position 0:", output_0)
    print("Note: This demonstrates the attention mechanism")
    print("      Full implementation would process all positions")
}
