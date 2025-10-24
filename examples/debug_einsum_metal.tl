// Debug script to compare CPU vs Metal einsum implementations
// Tests with small known inputs to identify output correctness issues

main {
    print("=== Einsum Metal vs CPU Debug Test ===")
    print("")

    // Test 1: Simple 2x2x2 einsum for "ihd,jhd->ihj"
    print("[Test 1] einsum ihd,jhd->ihj with 2x2x2 inputs")
    print("")

    // Create simple test inputs: [I=2, H=2, D=2]
    // Q[0,0,:] = [1.0, 0.0]
    // Q[0,1,:] = [0.0, 1.0]
    // Q[1,0,:] = [1.0, 1.0]
    // Q[1,1,:] = [0.5, 0.5]
    let Q_data = [
        1.0, 0.0,  // i=0, h=0
        0.0, 1.0,  // i=0, h=1
        1.0, 1.0,  // i=1, h=0
        0.5, 0.5   // i=1, h=1
    ]

    let Q = reshape(Q_data, [2.0, 2.0, 2.0])  // [I, H, D]
    let K = Q  // Use same tensor for K

    print("Input Q shape: [2, 2, 2]")
    print("Input K shape: [2, 2, 2]")
    print("Q data:", Q_data)
    print("")

    // Compute scores = einsum("ihd,jhd->ihj", Q, K)
    // Expected output shape: [2, 2, 2] (I, H, J)
    //
    // Manual calculation for verification:
    // scores[0,0,0] = Q[0,0,:] · K[0,0,:] = [1,0]·[1,0] = 1.0
    // scores[0,0,1] = Q[0,0,:] · K[1,0,:] = [1,0]·[1,1] = 1.0
    // scores[0,1,0] = Q[0,1,:] · K[0,1,:] = [0,1]·[0,1] = 1.0
    // scores[0,1,1] = Q[0,1,:] · K[1,1,:] = [0,1]·[0.5,0.5] = 0.5
    // scores[1,0,0] = Q[1,0,:] · K[0,0,:] = [1,1]·[1,0] = 1.0
    // scores[1,0,1] = Q[1,0,:] · K[1,0,:] = [1,1]·[1,1] = 2.0
    // scores[1,1,0] = Q[1,1,:] · K[0,1,:] = [0.5,0.5]·[0,1] = 0.5
    // scores[1,1,1] = Q[1,1,:] · K[1,1,:] = [0.5,0.5]·[0.5,0.5] = 0.5

    let scores = einsum("ihd,jhd->ihj", Q, K)

    print("Output shape:", shape(scores))
    print("Output scores:", scores)
    print("")
    print("Expected values:")
    print("  scores[0,0,:] = [1.0, 1.0]")
    print("  scores[0,1,:] = [1.0, 0.5]")
    print("  scores[1,0,:] = [1.0, 2.0]")
    print("  scores[1,1,:] = [0.5, 0.5]")
    print("")

    // Test 2: Check second einsum pattern "ihj,jhd->ihd"
    print("[Test 2] einsum ihj,jhd->ihd with 2x2x2 inputs")
    print("")

    // Use scores as attention weights (already [I, H, J])
    let attn_weights = scores
    let V = Q  // Use Q data as V for simplicity: [J, H, D]

    print("Attention weights shape: [2, 2, 2]")
    print("V shape: [2, 2, 2]")
    print("")

    // Compute output = einsum("ihj,jhd->ihd", attn_weights, V)
    // Expected output shape: [2, 2, 2] (I, H, D)
    let output = einsum("ihj,jhd->ihd", attn_weights, V)

    print("Output shape:", shape(output))
    print("Output values:", output)
    print("")

    // Test 3: Realistic attention scenario with known output
    print("[Test 3] Identity-like attention test")
    print("")

    // Create inputs where we expect specific output
    let identity_q = [
        1.0, 0.0,
        0.0, 1.0
    ]
    let Q_id = reshape(identity_q, [2.0, 1.0, 2.0])  // [2, 1, 2]
    let K_id = Q_id

    let scores_id = einsum("ihd,jhd->ihj", Q_id, K_id)
    print("Identity Q·K^T shape:", shape(scores_id))
    print("Identity scores:", scores_id)
    print("Expected: [[1.0, 0.0], [0.0, 1.0]] (identity matrix)")
    print("")

    print("=== Debug Test Complete ===")
    print("")
    print("If Metal einsum is working correctly, the output should match expected values.")
    print("Any mismatch indicates a bug in index calculation or memory layout.")
}
