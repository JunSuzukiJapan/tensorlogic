// Test GQA expansion: 4 heads → 32 heads (8x replication)
main {
    print("=== Testing GQA Expansion ===")
    print("")

    // Create simple test input: [2, 4, 64] (seq_len=2, n_kv_heads=4, head_dim=64)
    // Fill with distinctive values so we can track which head is which
    let seq_len = 2.0
    let n_kv_heads = 4.0
    let head_dim = 64.0

    // Create input with pattern: head_idx * 1000 + dim_idx
    // Head 0: all 0s, Head 1: all 1000s, Head 2: all 2000s, Head 3: all 3000s
    tensor K_input: float16[2, 4, 64] = [
        // Sequence position 0
        [[0.0; 64],    // Head 0
         [1000.0; 64], // Head 1
         [2000.0; 64], // Head 2
         [3000.0; 64]], // Head 3
        // Sequence position 1
        [[0.0; 64],
         [1000.0; 64],
         [2000.0; 64],
         [3000.0; 64]]
    ]

    print("[1] Input K shape:", shape(K_input))
    print("    Input K[0, 0, 0]:", K_input[0, 0, 0], "(Head 0)")
    print("    Input K[0, 1, 0]:", K_input[0, 1, 0], "(Head 1)")
    print("    Input K[0, 2, 0]:", K_input[0, 2, 0], "(Head 2)")
    print("    Input K[0, 3, 0]:", K_input[0, 3, 0], "(Head 3)")
    print("")

    // GQA expansion: 4 → 32 heads
    let shape_with_group = [seq_len, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_input, shape_with_group)
    print("[2] After reshape to [seq, 4, 1, 64]:", shape(K_with_group))

    let shape_broadcast = [seq_len, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    print("[3] After broadcast to [seq, 4, 8, 64]:", shape(K_broadcast))

    let shape_expanded = [seq_len, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    print("[4] After reshape to [seq, 32, 64]:", shape(K_expanded))
    print("")

    // Verify expansion pattern
    print("=== Verifying Expansion Pattern ===")
    print("Expected: Heads 0-7 should all be 0.0 (from KV head 0)")
    print("Expected: Heads 8-15 should all be 1000.0 (from KV head 1)")
    print("Expected: Heads 16-23 should all be 2000.0 (from KV head 2)")
    print("Expected: Heads 24-31 should all be 3000.0 (from KV head 3)")
    print("")

    print("Actual values at seq_pos=0, dim=0:")
    print("  Head 0:", K_expanded[0, 0, 0], "(should be 0.0)")
    print("  Head 7:", K_expanded[0, 7, 0], "(should be 0.0)")
    print("  Head 8:", K_expanded[0, 8, 0], "(should be 1000.0)")
    print("  Head 15:", K_expanded[0, 15, 0], "(should be 1000.0)")
    print("  Head 16:", K_expanded[0, 16, 0], "(should be 2000.0)")
    print("  Head 23:", K_expanded[0, 23, 0], "(should be 2000.0)")
    print("  Head 24:", K_expanded[0, 24, 0], "(should be 3000.0)")
    print("  Head 31:", K_expanded[0, 31, 0], "(should be 3000.0)")
    print("")

    print("=== Test Complete ===")
}
