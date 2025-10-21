// Test new features: attention mask, batch norm, dropout

main {
    print("Testing New Features")
    print("===================")

    // Test 1: Causal Mask
    print("\n1. Causal Mask Test")
    tensor mask: float16[3, 3] = causal_mask(3)
    print("Causal mask [3,3]:")
    print("  Row 0:", [mask[0, 0], mask[0, 1], mask[0, 2]])
    print("  Row 1:", [mask[1, 0], mask[1, 1], mask[1, 2]])
    print("  Row 2:", [mask[2, 0], mask[2, 1], mask[2, 2]])

    // Test 2: Apply Mask
    print("\n2. Apply Mask Test")
    tensor scores: float16[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
    tensor m: float16[2, 2] = [[1.0, 0.0], [1.0, 1.0]]
    tensor masked: float16[2, 2] = apply_mask(scores, m)
    print("Masked scores:")
    print("  Row 0:", [masked[0, 0], masked[0, 1]])
    print("  Row 1:", [masked[1, 0], masked[1, 1]])

    // Test 3: Batch Norm
    print("\n3. Batch Normalization Test")
    tensor x: float16[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor gamma: float16[3] = [1.0, 1.0, 1.0]
    tensor beta: float16[3] = [0.0, 0.0, 0.0]
    tensor normalized: float16[2, 3] = batch_norm(x, gamma, beta, 0.00001)
    print("Normalized:")
    print("  Row 0:", [normalized[0, 0], normalized[0, 1], normalized[0, 2]])
    print("  Row 1:", [normalized[1, 0], normalized[1, 1], normalized[1, 2]])

    // Test 4: Dropout (inference mode)
    print("\n4. Dropout Test (inference)")
    tensor data: float16[4] = [1.0, 2.0, 3.0, 4.0]
    tensor dropped: float16[4] = dropout(data, 0.5, false)
    print("Dropout (inference):", dropped)

    print("\nAll tests completed!")
}
