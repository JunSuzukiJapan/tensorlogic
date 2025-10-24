// Verify basic operations with known inputs/outputs

main {
    print("=== Operation Verification Tests ===")
    print("")

    // Test 1: RMSNorm with known values
    print("[Test 1] RMSNorm verification")
    let x = [2.0, 4.0, 6.0, 8.0]
    let x_tensor = reshape(x, [1.0, 4.0])  // [1, 4]
    let weight = [1.0, 1.0, 1.0, 1.0]
    let weight_tensor = reshape(weight, [4.0])  // [4]

    let normalized = rms_norm(x_tensor, weight_tensor)
    print("  Input:", x)
    print("  Output:", normalized)
    print("  Expected: values scaled by RMS normalization")

    // Manual calculation:
    // sq_sum = 4 + 16 + 36 + 64 = 120
    // mean_sq = 120 / 4 = 30
    // rms = sqrt(30 + 1e-6) = 5.477
    // output[i] = x[i] / rms * weight[i]
    // output = [0.365, 0.730, 1.095, 1.460]
    print("")

    // Test 2: Softmax with known values
    print("[Test 2] Softmax verification")
    let logits = [1.0, 2.0, 3.0]
    let logits_tensor = reshape(logits, [1.0, 3.0])  // [1, 3]

    let probs = softmax(logits_tensor, 1)
    print("  Input:", logits)
    print("  Output:", probs)

    // Manual calculation:
    // max = 3.0
    // exp(1-3) = exp(-2) = 0.135
    // exp(2-3) = exp(-1) = 0.368
    // exp(3-3) = exp(0) = 1.0
    // sum = 1.503
    // probs = [0.090, 0.245, 0.665]
    print("  Expected: [0.090, 0.245, 0.665]")
    print("")

    // Test 3: Embedding lookup
    print("[Test 3] Embedding verification")
    let embed_table_data = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    ]
    let embed_table = reshape(embed_table_data, [4.0, 3.0])  // [vocab=4, embed=3]

    let token_ids = [0, 2, 1]  // Select rows 0, 2, 1
    let embedded = embedding(embed_table, token_ids)

    print("  Token IDs:", token_ids)
    print("  Embedded shape:", shape(embedded))
    print("  Expected shape: [3, 3]")
    print("  Expected values:")
    print("    Row 0 (token 0): [1.0, 2.0, 3.0]")
    print("    Row 1 (token 2): [7.0, 8.0, 9.0]")
    print("    Row 2 (token 1): [4.0, 5.0, 6.0]")
    print("")

    // Test 4: Matrix multiplication
    print("[Test 4] Matmul verification")
    let A = [
        1.0, 2.0,
        3.0, 4.0
    ]
    let B = [
        5.0, 6.0,
        7.0, 8.0
    ]
    let A_tensor = reshape(A, [2.0, 2.0])
    let B_tensor = reshape(B, [2.0, 2.0])

    let C = matmul(A_tensor, B_tensor)
    print("  A:", A)
    print("  B:", B)
    print("  C = A @ B:", C)

    // Manual calculation:
    // C[0,0] = 1*5 + 2*7 = 19
    // C[0,1] = 1*6 + 2*8 = 22
    // C[1,0] = 3*5 + 4*7 = 43
    // C[1,1] = 3*6 + 4*8 = 50
    print("  Expected: [19.0, 22.0, 43.0, 50.0]")
    print("")

    print("=== Verification Complete ===")
    print("")
    print("Compare outputs with expected values.")
    print("If any mismatch, that operation has a bug.")
}
