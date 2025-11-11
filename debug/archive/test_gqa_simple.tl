// Simplified GQA expansion test
main {
    print("=== Testing GQA Expansion (Simplified) ===")
    print("")

    // Create small test: [1, 2, 4] (seq=1, kv_heads=2, head_dim=4)
    // Expand to: [1, 4, 4] (seq=1, q_heads=4, head_dim=4)
    tensor K_input: float16[1, 2, 4] = [
        [[1.0, 2.0, 3.0, 4.0],      // KV Head 0
         [10.0, 20.0, 30.0, 40.0]]  // KV Head 1
    ]

    print("[1] Input K shape:", shape(K_input))
    print("    K[0, 0, :] (Head 0):", [K_input[0, 0, 0], K_input[0, 0, 1], K_input[0, 0, 2], K_input[0, 0, 3]])
    print("    K[0, 1, :] (Head 1):", [K_input[0, 1, 0], K_input[0, 1, 1], K_input[0, 1, 2], K_input[0, 1, 3]])
    print("")

    // GQA expansion: 2 KV heads → 4 Q heads (2x replication)
    let seq_len = 1.0
    let shape_with_group = [seq_len, 2.0, 1.0, 4.0]
    let K_with_group = reshape(K_input, shape_with_group)
    print("[2] Reshape to [1, 2, 1, 4]:", shape(K_with_group))

    let shape_broadcast = [seq_len, 2.0, 2.0, 4.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    print("[3] Broadcast to [1, 2, 2, 4]:", shape(K_broadcast))

    let shape_expanded = [seq_len, 4.0, 4.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    print("[4] Reshape to [1, 4, 4]:", shape(K_expanded))
    print("")

    // Verify expansion
    print("=== Verifying Expansion ===")
    print("Expected pattern:")
    print("  Heads 0-1 should match KV Head 0 (values 1,2,3,4)")
    print("  Heads 2-3 should match KV Head 1 (values 10,20,30,40)")
    print("")

    print("Actual values:")
    print("  Head 0:", [K_expanded[0, 0, 0], K_expanded[0, 0, 1], K_expanded[0, 0, 2], K_expanded[0, 0, 3]])
    print("  Head 1:", [K_expanded[0, 1, 0], K_expanded[0, 1, 1], K_expanded[0, 1, 2], K_expanded[0, 1, 3]])
    print("  Head 2:", [K_expanded[0, 2, 0], K_expanded[0, 2, 1], K_expanded[0, 2, 2], K_expanded[0, 2, 3]])
    print("  Head 3:", [K_expanded[0, 3, 0], K_expanded[0, 3, 1], K_expanded[0, 3, 2], K_expanded[0, 3, 3]])
    print("")

    // Check correctness
    let head0_correct = K_expanded[0, 0, 0] == 1.0
    let head1_correct = K_expanded[0, 1, 0] == 1.0
    let head2_correct = K_expanded[0, 2, 0] == 10.0
    let head3_correct = K_expanded[0, 3, 0] == 10.0

    if head0_correct {
        if head1_correct {
            if head2_correct {
                if head3_correct {
                    print("✅ GQA Expansion is CORRECT!")
                }
            }
        }
    }

    if !head0_correct {
        print("❌ Head 0 is WRONG!")
    }
    if !head1_correct {
        print("❌ Head 1 is WRONG!")
    }
    if !head2_correct {
        print("❌ Head 2 is WRONG!")
    }
    if !head3_correct {
        print("❌ Head 3 is WRONG!")
    }

    print("")
    print("=== Test Complete ===")
}
