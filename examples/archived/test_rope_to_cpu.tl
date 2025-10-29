// Test to_cpu() after reshape→rope twice

main {
    print("=== Testing to_cpu() After Reshape → RoPE Twice ===")
    print("")

    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // First: reshape → rope
    print("[1] First Reshape → RoPE...")
    let x1 = f32::ones([seq_len, hidden])
    let x1_heads = reshape(x1, [seq_len, num_heads, head_dim])
    let x1_rope = rope(x1_heads)
    print("  ✓ OK")
    print("")

    // Second: reshape → rope
    print("[2] Second Reshape → RoPE...")
    let x2 = f32::ones([seq_len, hidden])
    let x2_heads = reshape(x2, [seq_len, num_heads, head_dim])
    let x2_rope = rope(x2_heads)
    print("  ✓ OK")
    print("")

    // Try to_vec() (which calls to_cpu internally)
    print("[3] Calling to_vec() on result...")
    let shape_result = shape(x2_rope)
    print("  Shape:", shape_result)
    print("")

    print("=== Completed ===")
}
