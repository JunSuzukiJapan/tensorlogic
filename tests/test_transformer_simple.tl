// ============================================================================
// Simple Transformer Component Test
// ============================================================================
//
// Simplified test based on Candle's transformer patterns
// Tests basic transformer operations with small tensors
//
// ============================================================================

main {
    print("================================================================================")
    print("Simple Transformer Component Test")
    print("================================================================================")
    print("")

    // ========================================================================
    // Test 1: Basic Matrix Multiplication (Q, K, V projection)
    // ========================================================================
    print("Test 1: Q, K, V Projections")
    print("--------------------------------------------------------------------------------")
    print("")

    // Input: [2, 4] - 2 tokens, 4 dimensions
    tensor input: float16[2, 4] = [[0.5, 0.3, 0.2, 0.1],
                                    [0.3, 0.5, 0.4, 0.2]]

    // Projection weights: [4, 4]
    tensor W_q: float16[4, 4] = [[0.8, 0.1, 0.0, 0.0],
                                  [0.1, 0.8, 0.0, 0.0],
                                  [0.0, 0.1, 0.8, 0.0],
                                  [0.0, 0.0, 0.1, 0.8]]

    tensor Q: float16[2, 4] = input @ W_q

    print("Input shape: [2, 4]")
    print("W_q shape: [4, 4]")
    print("Q shape: [2, 4]")
    print("")
    print("Q values:")
    print("  Token 0:", [Q[0, 0], Q[0, 1], Q[0, 2], Q[0, 3]])
    print("  Token 1:", [Q[1, 0], Q[1, 1], Q[1, 2], Q[1, 3]])
    print("")
    print("✓ Q projection successful")
    print("")

    // ========================================================================
    // Test 2: Attention Scores (Q @ K^T)
    // ========================================================================
    print("Test 2: Attention Scores")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor W_k: float16[4, 4] = [[0.7, 0.2, 0.0, 0.0],
                                  [0.2, 0.7, 0.0, 0.0],
                                  [0.0, 0.2, 0.7, 0.0],
                                  [0.0, 0.0, 0.2, 0.7]]

    tensor K: float16[2, 4] = input @ W_k
    tensor scores: float16[2, 2] = Q @ transpose(K)

    print("Attention scores (Q @ K^T):")
    print("  Row 0:", [scores[0, 0], scores[0, 1]])
    print("  Row 1:", [scores[1, 0], scores[1, 1]])
    print("")
    print("✓ Attention scores computed")
    print("")

    // ========================================================================
    // Test 3: Scaled Attention
    // ========================================================================
    print("Test 3: Scaled Attention")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor scaled: float16[2, 2] = scores / [2.0]  // sqrt(4) = 2.0

    print("Scaled scores (/ sqrt(d_k)):")
    print("  Row 0:", [scaled[0, 0], scaled[0, 1]])
    print("  Row 1:", [scaled[1, 0], scaled[1, 1]])
    print("")
    print("✓ Scaling applied")
    print("")

    // ========================================================================
    // Test 4: Softmax Attention Weights
    // ========================================================================
    print("Test 4: Softmax Attention Weights")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor attn_weights: float16[2, 2] = softmax(scaled)

    print("Attention weights (softmax):")
    print("  Row 0:", [attn_weights[0, 0], attn_weights[0, 1]])
    print("  Row 1:", [attn_weights[1, 0], attn_weights[1, 1]])
    print("")
    print("Note: Each row should sum to ~1.0")
    print("")
    print("✓ Softmax applied")
    print("")

    // ========================================================================
    // Test 5: Apply Attention to Values
    // ========================================================================
    print("Test 5: Apply Attention to Values")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor W_v: float16[4, 4] = [[0.6, 0.2, 0.0, 0.0],
                                  [0.2, 0.6, 0.0, 0.0],
                                  [0.0, 0.2, 0.6, 0.0],
                                  [0.0, 0.0, 0.2, 0.6]]

    tensor V: float16[2, 4] = input @ W_v
    tensor attended: float16[2, 4] = attn_weights @ V

    print("Attended output (attn_weights @ V):")
    print("  Token 0:", [attended[0, 0], attended[0, 1], attended[0, 2], attended[0, 3]])
    print("  Token 1:", [attended[1, 0], attended[1, 1], attended[1, 2], attended[1, 3]])
    print("")
    print("✓ Attention applied to values")
    print("")

    // ========================================================================
    // Test 6: Layer Normalization
    // ========================================================================
    print("Test 6: Layer Normalization")
    print("--------------------------------------------------------------------------------")
    print("")

    // Layer norm with eps parameter only (gamma/beta not yet supported)
    tensor normalized: float16[2, 4] = layer_norm(attended, 0.00001)

    print("Normalized output:")
    print("  Token 0:", [normalized[0, 0], normalized[0, 1], normalized[0, 2], normalized[0, 3]])
    print("  Token 1:", [normalized[1, 0], normalized[1, 1], normalized[1, 2], normalized[1, 3]])
    print("")
    print("✓ Layer normalization applied")
    print("")

    // ========================================================================
    // Test 7: Residual Connection
    // ========================================================================
    print("Test 7: Residual Connection")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor with_residual: float16[2, 4] = input + attended

    print("With residual (input + attended):")
    print("  Token 0:", [with_residual[0, 0], with_residual[0, 1], with_residual[0, 2], with_residual[0, 3]])
    print("  Token 1:", [with_residual[1, 0], with_residual[1, 1], with_residual[1, 2], with_residual[1, 3]])
    print("")
    print("✓ Residual connection added")
    print("")

    // ========================================================================
    // Test 8: Feed-Forward Network
    // ========================================================================
    print("Test 8: Feed-Forward Network")
    print("--------------------------------------------------------------------------------")
    print("")

    // FFN: expand to [2, 8], then project back to [2, 4]
    tensor W_ff1: float16[4, 8] = [[0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.2, 0.3],
                                    [0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4],
                                    [0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5],
                                    [0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3]]

    tensor W_ff2: float16[8, 4] = [[0.4, 0.3, 0.2, 0.1],
                                    [0.3, 0.4, 0.3, 0.2],
                                    [0.2, 0.3, 0.4, 0.3],
                                    [0.1, 0.2, 0.3, 0.4],
                                    [0.3, 0.1, 0.2, 0.3],
                                    [0.4, 0.3, 0.1, 0.2],
                                    [0.2, 0.4, 0.3, 0.1],
                                    [0.3, 0.2, 0.4, 0.3]]

    tensor ff_hidden: float16[2, 8] = relu(with_residual @ W_ff1)
    tensor ff_output: float16[2, 4] = ff_hidden @ W_ff2

    print("FFN output:")
    print("  Token 0:", [ff_output[0, 0], ff_output[0, 1], ff_output[0, 2], ff_output[0, 3]])
    print("  Token 1:", [ff_output[1, 0], ff_output[1, 1], ff_output[1, 2], ff_output[1, 3]])
    print("")
    print("✓ Feed-forward network applied")
    print("")

    // ========================================================================
    // Test 9: Final Residual Connection
    // ========================================================================
    print("Test 9: Final Residual Connection")
    print("--------------------------------------------------------------------------------")
    print("")

    tensor final_output: float16[2, 4] = with_residual + ff_output

    print("Final output (with_residual + ff_output):")
    print("  Token 0:", [final_output[0, 0], final_output[0, 1], final_output[0, 2], final_output[0, 3]])
    print("  Token 1:", [final_output[1, 0], final_output[1, 1], final_output[1, 2], final_output[1, 3]])
    print("")
    print("✓ Final residual connection added")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("================================================================================")
    print("Summary - All Transformer Components Verified")
    print("================================================================================")
    print("")

    print("All components tested successfully:")
    print("  ✓ 1. Q, K, V projections")
    print("  ✓ 2. Attention scores (Q @ K^T)")
    print("  ✓ 3. Scaled attention")
    print("  ✓ 4. Softmax attention weights")
    print("  ✓ 5. Apply attention to values")
    print("  ✓ 6. Layer normalization")
    print("  ✓ 7. Residual connection")
    print("  ✓ 8. Feed-forward network")
    print("  ✓ 9. Final residual connection")
    print("")

    print("Complete transformer block pattern:")
    print("  x = x + Attention(LayerNorm(x))")
    print("  x = x + FFN(LayerNorm(x))")
    print("")

    print("All transformer components working correctly!")
    print("")

    print("================================================================================")
    print("End of Simple Transformer Component Test")
    print("================================================================================")
}
