// Profile matrix multiplication performance

main {
    print("=== Matrix Multiplication Performance Test ===")
    print("")

    // Test typical transformer decode pattern: [1, 2048] @ [2048, 2048]
    let K = 2048
    let N = 2048

    print("Creating test matrices...")
    let A = ones_f32(1, K)
    let B = ones_f32(K, N)
    
    print("Test 1: Regular matmul")
    let start1 = 0
    let C1 = matmul(A, B)
    print("  Completed regular matmul")

    print("")
    print("Test 2: Fused transpose-matmul")
    let B_t = transpose(B)
    let C2 = matmul(A, B_t)
    print("  Completed fused transpose-matmul")

    print("")
    print("=== Test Complete ===")
}
