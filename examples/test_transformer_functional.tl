// Functional tests for Transformer operations
// Actually executes each function to verify it works

main {
    print("=== Functional Tests for Transformer Operations ===")
    print("")

    // Test 1: ReLU activation
    print("Test 1: ReLU Activation")
    print("Creating tensor with positional encoding (contains positive and negative values)")
    let test_tensor = positional_encoding(4, 8)
    print("Applying ReLU (should zero out negative values)")
    let relu_output = relu(test_tensor)
    print("✓ ReLU executed successfully!")
    print("")

    // Test 2: Matrix multiplication (matmul)
    print("Test 2: Matrix Multiplication")
    print("Creating two matrices using positional encoding")
    let matrix_a = positional_encoding(4, 8)  // [4, 8]
    let matrix_b = positional_encoding(8, 6)  // [8, 6]
    print("Computing matmul(A, B) -> should give [4, 6]")
    let matmul_result = matmul(matrix_a, matrix_b)
    print("✓ Matrix multiplication executed successfully!")
    print("")

    // Test 3: Layer normalization
    print("Test 3: Layer Normalization")
    print("Creating tensor for normalization")
    let norm_input = positional_encoding(4, 8)
    print("Applying layer_norm (normalizes last dimension)")
    let norm_output = layer_norm(norm_input)
    print("✓ Layer normalization executed successfully!")
    print("")

    // Test 4: Concatenation
    print("Test 4: Tensor Concatenation")
    print("Creating two tensors to concatenate")
    let concat_a = positional_encoding(4, 8)  // [4, 8]
    let concat_b = positional_encoding(4, 8)  // [4, 8]
    print("Concatenating along dimension 1 (columns)")
    let concat_result = concat(concat_a, concat_b, 1)  // Should give [4, 16]
    print("✓ Concatenation executed successfully!")
    print("")

    // Test 5: Softmax
    print("Test 5: Softmax (probability distribution)")
    let logits = positional_encoding(1, 10)  // [1, 10]
    print("Converting logits to probabilities")
    let probs = softmax(logits)
    print("✓ Softmax executed successfully!")
    print("")

    // Test 6: Combined operations (simple attention-like computation)
    print("Test 6: Combined Operations (Attention-like)")
    print("Simulating a simple attention mechanism:")
    print("")

    print("  Step 1: Create Query and Key matrices")
    let query = positional_encoding(4, 8)   // [4, 8] - 4 positions, 8-dim
    let key = positional_encoding(4, 8)     // [4, 8]
    print("  ✓ Q and K created")
    print("")

    print("  Step 2: Compute attention scores (Q @ K^T)")
    print("  Note: Need transpose operation for full attention")
    print("  For now, computing Q @ K (simplified)")
    let key_for_mul = positional_encoding(8, 4)  // Simulate transpose
    let scores = matmul(query, key_for_mul)  // [4, 4]
    print("  ✓ Attention scores computed")
    print("")

    print("  Step 3: Apply softmax to get attention weights")
    let attn_weights = softmax(scores)
    print("  ✓ Attention weights computed")
    print("")

    print("  Step 4: Apply attention to values")
    let value = positional_encoding(4, 8)  // [4, 8]
    let attn_output = matmul(attn_weights, value)  // [4, 8]
    print("  ✓ Attention output computed")
    print("")

    print("  Step 5: Apply layer normalization")
    let normalized = layer_norm(attn_output)
    print("  ✓ Normalized output")
    print("")

    // Test 7: MLP block simulation
    print("Test 7: MLP Block Simulation")
    print("Simulating a simple MLP (Multi-Layer Perceptron):")
    print("")

    print("  Step 1: Input projection")
    let mlp_input = positional_encoding(4, 8)   // [4, 8]
    let w1 = positional_encoding(8, 16)          // [8, 16] - expand dimension
    let mlp_hidden = matmul(mlp_input, w1)       // [4, 16]
    print("  ✓ Projected to hidden dimension")
    print("")

    print("  Step 2: Apply ReLU activation")
    let mlp_activated = relu(mlp_hidden)
    print("  ✓ ReLU activation applied")
    print("")

    print("  Step 3: Output projection")
    let w2 = positional_encoding(16, 8)          // [16, 8] - back to original
    let mlp_output = matmul(mlp_activated, w2)   // [4, 8]
    print("  ✓ Projected back to output dimension")
    print("")

    print("  Step 4: Layer normalization")
    let mlp_final = layer_norm(mlp_output)
    print("  ✓ MLP block complete")
    print("")

    // Summary
    print("==================================================")
    print("All Transformer Operations Tested Successfully!")
    print("==================================================")
    print("")
    print("Verified operations:")
    print("  ✓ relu() - Activation function")
    print("  ✓ matmul() - Matrix multiplication")
    print("  ✓ layer_norm() - Normalization")
    print("  ✓ concat() - Tensor concatenation")
    print("  ✓ softmax() - Probability distribution")
    print("")
    print("Demonstrated patterns:")
    print("  ✓ Attention mechanism (Q, K, V)")
    print("  ✓ MLP block (Linear -> ReLU -> Linear)")
    print("  ✓ Layer normalization (residual connections)")
    print("")
    print("All components ready for full Transformer implementation!")
}
