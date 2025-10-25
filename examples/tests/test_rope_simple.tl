// Simple RoPE test - position 0 should preserve input (no rotation)

main {
    print("=== RoPE Test: Position 0 (Identity) ===")

    // Create simple input: [seq=1, heads=1, dim=8]
    let input = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    let tensor_in = reshape(input, [1, 1, 8])

    print("Input shape:", shape(tensor_in))
    print("Input values:", tensor_in)

    // Apply RoPE at position 0 (should be identity: cos(0)=1, sin(0)=0)
    let output = rope(tensor_in, 0)

    print("Output shape:", shape(output))
    print("Output values:", output)
    print("Expected: Same as input (no rotation at position 0)")
}
