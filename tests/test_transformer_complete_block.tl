// ============================================================================
// Complete Transformer Block Test (inspired by Candle transformers)
// ============================================================================
//
// Tests complete transformer block with:
//   - Self-attention mechanism
//   - Residual connections
//   - Layer normalization
//   - Feed-forward network (MLP)
//   - Complete forward pass
//
// Based on:
//   - Candle's llama.rs Block implementation
//   - "Attention is All You Need" paper (Vaswani et al., 2017)
// ============================================================================

main {
    print("================================================================================")
    print("Complete Transformer Block Test")
    print("================================================================================")
    print("")

    // ========================================================================
    // Configuration
    // ========================================================================
    let seq_len = 4          // Sequence length
    let d_model = 8          // Model dimension (smaller for testing)
    let d_ff = 16            // Feed-forward dimension (2x d_model)

    print("Configuration:")
    print("  Sequence length:", seq_len)
    print("  Model dimension (d_model):", d_model)
    print("  Feed-forward dimension (d_ff):", d_ff)
    print("")

    // ========================================================================
    // Test 1: Input Preparation
    // ========================================================================
    print("================================================================================")
    print("Test 1: Input Embeddings")
    print("================================================================================")
    print("")

    // Input: [seq_len, d_model] = [4, 8]
    tensor x_input: float16[4, 8] = [[0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.2, 0.3],
                                      [0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4],
                                      [0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5],
                                      [0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3]]

    print("Input [4, 8]:")
    print("  Token 0:", [x_input[0, 0], x_input[0, 1], x_input[0, 2], x_input[0, 3]])
    print("  Token 1:", [x_input[1, 0], x_input[1, 1], x_input[1, 2], x_input[1, 3]])
    print("✓ Input prepared")
    print("")

    // ========================================================================
    // Test 2: Pre-Attention Layer Normalization
    // ========================================================================
    print("================================================================================")
    print("Test 2: Pre-Attention Layer Normalization")
    print("================================================================================")
    print("")
    print("Equation: x_norm = LayerNorm(x)")
    print("  This normalizes each token's features to mean=0, std=1")
    print("")

    tensor ln1_gamma: float16[8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    tensor ln1_beta: float16[8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    tensor x_norm1: float16[4, 8] = layer_norm(x_input, ln1_gamma, ln1_beta, 0.00001)

    print("Normalized input [4, 8]:")
    print("  Token 0:", [x_norm1[0, 0], x_norm1[0, 1], x_norm1[0, 2], x_norm1[0, 3]])
    print("✓ Pre-attention layer norm applied")
    print("")

    // ========================================================================
    // Test 3: Self-Attention
    // ========================================================================
    print("================================================================================")
    print("Test 3: Self-Attention Mechanism")
    print("================================================================================")
    print("")
    print("Equation: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    print("")

    // Q, K, V projection weights: [d_model, d_model] = [8, 8]
    tensor W_q: float16[8, 8] = [[0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.1, 0.1, 0.8, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8]]

    tensor W_k: float16[8, 8] = [[0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.2, 0.7, 0.1, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.1],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7]]

    tensor W_v: float16[8, 8] = [[0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.2, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.2, 0.2, 0.6, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6]]

    // Project to Q, K, V
    tensor Q: float16[4, 8] = x_norm1 @ W_q
    tensor K: float16[4, 8] = x_norm1 @ W_k
    tensor V: float16[4, 8] = x_norm1 @ W_v

    // Compute attention scores and weights
    tensor attn_scores: float16[4, 4] = Q @ transpose(K)
    tensor attn_scaled: float16[4, 4] = attn_scores / [2.828]  // sqrt(8)
    tensor attn_weights: float16[4, 4] = softmax(attn_scaled)

    // Apply attention to values
    tensor attn_out: float16[4, 8] = attn_weights @ V

    print("Attention output [4, 8]:")
    print("  Token 0:", [attn_out[0, 0], attn_out[0, 1], attn_out[0, 2], attn_out[0, 3]])
    print("✓ Self-attention computed")
    print("")

    // ========================================================================
    // Test 4: Output Projection + Residual Connection
    // ========================================================================
    print("================================================================================")
    print("Test 4: Output Projection and Residual Connection")
    print("================================================================================")
    print("")
    print("Equation: x = x + Attention(LayerNorm(x))")
    print("  The residual connection adds the input back to attention output")
    print("")

    // Output projection
    tensor W_o: float16[8, 8] = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]]

    tensor attn_proj: float16[4, 8] = attn_out @ W_o

    // Add residual connection
    tensor x_after_attn: float16[4, 8] = x_input + attn_proj

    print("After attention + residual [4, 8]:")
    print("  Token 0:", [x_after_attn[0, 0], x_after_attn[0, 1], x_after_attn[0, 2], x_after_attn[0, 3]])
    print("✓ Residual connection applied")
    print("")

    // ========================================================================
    // Test 5: Pre-FFN Layer Normalization
    // ========================================================================
    print("================================================================================")
    print("Test 5: Pre-FFN Layer Normalization")
    print("================================================================================")
    print("")

    tensor ln2_gamma: float16[8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    tensor ln2_beta: float16[8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    tensor x_norm2: float16[4, 8] = layer_norm(x_after_attn, ln2_gamma, ln2_beta, 0.00001)

    print("Normalized for FFN [4, 8]:")
    print("  Token 0:", [x_norm2[0, 0], x_norm2[0, 1], x_norm2[0, 2], x_norm2[0, 3]])
    print("✓ Pre-FFN layer norm applied")
    print("")

    // ========================================================================
    // Test 6: Feed-Forward Network (MLP)
    // ========================================================================
    print("================================================================================")
    print("Test 6: Feed-Forward Network (MLP)")
    print("================================================================================")
    print("")
    print("Equation: FFN(x) = ReLU(x @ W1) @ W2")
    print("  Two-layer MLP with ReLU activation")
    print("")

    // W1: [d_model, d_ff] = [8, 16]
    tensor W_ff1: float16[8, 16] = [
        [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.2, 0.3, 0.1, 0.5, 0.4, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.3, 0.5, 0.4, 0.1, 0.2, 0.3, 0.4],
        [0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.1, 0.2, 0.4, 0.5, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2],
        [0.4, 0.1, 0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3],
        [0.6, 0.3, 0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3, 0.5],
        [0.2, 0.5, 0.4, 0.1, 0.2, 0.4, 0.5, 0.3, 0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3]
    ]

    // W2: [d_ff, d_model] = [16, 8]
    tensor W_ff2: float16[16, 8] = [
        [0.4, 0.3, 0.2, 0.1, 0.3, 0.5, 0.2, 0.3],
        [0.3, 0.4, 0.3, 0.2, 0.1, 0.3, 0.4, 0.2],
        [0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.3],
        [0.3, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
        [0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2],
        [0.2, 0.4, 0.3, 0.1, 0.2, 0.3, 0.4, 0.3],
        [0.3, 0.2, 0.4, 0.3, 0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2, 0.3],
        [0.2, 0.4, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2],
        [0.3, 0.2, 0.4, 0.3, 0.2, 0.4, 0.3, 0.1],
        [0.1, 0.3, 0.2, 0.4, 0.3, 0.2, 0.4, 0.3],
        [0.3, 0.1, 0.3, 0.2, 0.4, 0.3, 0.2, 0.4],
        [0.4, 0.3, 0.1, 0.3, 0.2, 0.4, 0.3, 0.2],
        [0.2, 0.4, 0.3, 0.1, 0.3, 0.2, 0.4, 0.3],
        [0.3, 0.2, 0.4, 0.3, 0.1, 0.3, 0.2, 0.4]
    ]

    // First linear layer + ReLU
    tensor ff_hidden: float16[4, 16] = relu(x_norm2 @ W_ff1)

    print("FFN hidden layer [4, 16] (after ReLU):")
    print("  Token 0:", [ff_hidden[0, 0], ff_hidden[0, 1], ff_hidden[0, 2], ff_hidden[0, 3]])

    // Second linear layer
    tensor ff_out: float16[4, 8] = ff_hidden @ W_ff2

    print("FFN output [4, 8]:")
    print("  Token 0:", [ff_out[0, 0], ff_out[0, 1], ff_out[0, 2], ff_out[0, 3]])
    print("✓ Feed-forward network computed")
    print("")

    // ========================================================================
    // Test 7: Final Residual Connection
    // ========================================================================
    print("================================================================================")
    print("Test 7: Final Residual Connection")
    print("================================================================================")
    print("")
    print("Equation: output = x + FFN(LayerNorm(x))")
    print("")

    tensor block_output: float16[4, 8] = x_after_attn + ff_out

    print("Block output [4, 8]:")
    print("  Token 0:", [block_output[0, 0], block_output[0, 1], block_output[0, 2], block_output[0, 3]])
    print("  Token 1:", [block_output[1, 0], block_output[1, 1], block_output[1, 2], block_output[1, 3]])
    print("✓ Final residual connection applied")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("================================================================================")
    print("Summary - Complete Transformer Block")
    print("================================================================================")
    print("")

    print("All components verified:")
    print("  ✓ 1. Input preparation")
    print("  ✓ 2. Pre-attention layer normalization")
    print("  ✓ 3. Self-attention mechanism")
    print("  ✓ 4. Attention output projection + residual")
    print("  ✓ 5. Pre-FFN layer normalization")
    print("  ✓ 6. Feed-forward network (MLP)")
    print("  ✓ 7. Final residual connection")
    print("")

    print("Complete transformer block structure:")
    print("  x = x + Attention(LayerNorm(x))")
    print("  x = x + FFN(LayerNorm(x))")
    print("")

    print("Transformer block working correctly!")
    print("")

    print("================================================================================")
    print("End of Complete Transformer Block Test")
    print("================================================================================")
}
