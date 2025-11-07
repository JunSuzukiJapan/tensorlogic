// Simple test to verify GPU matrix multiplication

main {
    print("=== Testing GPU Matrix Multiplication ===")
    print("")

    // Create two matrices
    let A = tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], float32)
    let B = tensor([[7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0]], float32)

    print("[1/2] Input matrices created")
    print("A shape:", shape(A))
    print("B shape:", shape(B))
    print("")

    print("[2/2] Computing A @ B on GPU...")
    let C = linear(A, B)

    print("Result shape:", shape(C))
    print("Result:", C)
    print("")
    print("=== Test Complete ===")
}
