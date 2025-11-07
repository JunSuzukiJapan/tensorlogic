// Test to verify actual tensor shapes vs shape() results

main {
    print("=== Testing Actual Tensor Shapes ===")
    print("")

    // Test 1: Create tensor and check if operations work
    print("Test 1: Create [2, 3] tensor and perform operations")
    let t1 = zeros([2.0, 3.0])
    print("  Created t1 = zeros([2, 3])")
    print("  shape(t1) = {}", shape(t1))

    // Try to create another tensor with same shape for addition
    let t2 = zeros([2.0, 3.0])
    print("  Created t2 = zeros([2, 3])")
    let t3 = t1 + t2
    print("  t3 = t1 + t2")
    print("  shape(t3) = {}", shape(t3))
    print("")

    // Test 2: Create large tensor like tok_embd
    print("Test 2: Create [32000, 2048] tensor (like tok_embd)")
    let big = zeros([32000.0, 2048.0])
    print("  Created big = zeros([32000, 2048])")
    print("  shape(big) = {}", shape(big))

    // Try reshape on it
    let reshaped = reshape(big, [2048.0, 32000.0])
    print("  reshaped = reshape(big, [2048, 32000])")
    print("  shape(reshaped) = {}", shape(reshaped))
    print("")

    // Test 3: Try operations that should fail if tensor is actually [2]
    print("Test 3: Operations that would fail if tensor is really [2]")
    let t_2_3 = zeros([2.0, 3.0])
    print("  Created t_2_3 = zeros([2, 3])")

    // This should work if t_2_3 is really [2, 3]
    let t_3_4 = zeros([3.0, 4.0])
    print("  Created t_3_4 = zeros([3, 4])")

    // Matrix multiply: [2, 3] @ [3, 4] = [2, 4]
    let result = matmul(t_2_3, t_3_4)
    print("  result = matmul(t_2_3, t_3_4)  # [2,3] @ [3,4] = [2,4]")
    print("  shape(result) = {}", shape(result))
    print("")

    print("=== Tests Complete ===")
    print("")
    print("INTERPRETATION:")
    print("  If all operations succeed, tensors have correct shapes")
    print("  If matmul fails, it means tensors don't have expected shapes")
}
