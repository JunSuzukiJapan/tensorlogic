// Simple RoPE test
main {
    print("Testing RoPE implementation...")

    // Create test tensor: [seq_len=2, n_heads=2, head_dim=4]
    // Shape: [2, 2, 4] = 16 elements
    let input = ones([2, 2, 4])
    print("Input shape:", shape(input))
    print("Input:", input)

    print("")
    print("Applying RoPE...")
    let output = rope(input)

    print("Output shape:", shape(output))
    print("Output:", output)

    print("")
    print("✓ RoPE test complete!")
}
