// Test slice_last operation

main {
    print("================================================================================")
    print("Testing slice_last operation")
    print("================================================================================")
    print("")

    // Test 1: 2D tensor
    print("[Test 1] 2D tensor [3, 4]:")
    let x = tensor([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0],
                     [9.0, 10.0, 11.0, 12.0]])
    print("  Input:", x)
    print("  Shape:", shape(x))

    let last = slice_last(x, 0)
    print("  slice_last(axis=0):", last)
    print("  Output shape:", shape(last))
    print("  Expected: [9.0, 10.0, 11.0, 12.0]")
    print("")

    // Test 2: 3D tensor
    print("[Test 2] 3D tensor [2, 3, 4]:")
    let y = ones([2, 3, 4])
    let y_scaled = mul(y, 5.0)
    print("  Input shape:", shape(y_scaled))

    let last2 = slice_last(y_scaled, 0)
    print("  slice_last(axis=0) shape:", shape(last2))
    print("  Min:", min(last2))
    print("  Max:", max(last2))
    print("  Expected: all 5.0")
    print("")

    // Test 3: Real-world case - transformer output
    print("[Test 3] Transformer-like tensor [46, 2048]:")
    let transformer_out = ones([46, 2048])
    let transformer_scaled = mul(transformer_out, 3.5)
    print("  Input shape:", shape(transformer_scaled))

    let last_token = slice_last(transformer_scaled, 0)
    print("  slice_last(axis=0) shape:", shape(last_token))
    print("  Min:", min(last_token))
    print("  Max:", max(last_token))
    print("  Expected: all 3.5")
    print("")

    print("================================================================================")
    print("All tests completed")
    print("================================================================================")
}
