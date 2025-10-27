// Test for unsqueeze and squeeze operations

tensor a: float16[3] = [1.0, 2.0, 3.0]
tensor b: float16[2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

main {
    print("=== TensorLogic unsqueeze/squeeze Test ===")
    print()

    // Test 1: unsqueeze on 1D tensor
    print("Test 1: unsqueeze on 1D tensor")
    print("Original tensor a:", a)
    let a_unsqueeze_0 = a.unsqueeze(0)
    print("unsqueeze(a, 0) shape should be [1, 3]:", a_unsqueeze_0)

    let a_unsqueeze_1 = a.unsqueeze(1)
    print("unsqueeze(a, 1) shape should be [3, 1]:", a_unsqueeze_1)
    print()

    // Test 2: unsqueeze on 2D tensor
    print("Test 2: unsqueeze on 2D tensor")
    print("Original tensor b shape [2, 3]:", b)
    let b_unsqueeze_0 = b.unsqueeze(0)
    print("unsqueeze(b, 0) shape should be [1, 2, 3]:", b_unsqueeze_0)

    let b_unsqueeze_1 = b.unsqueeze(1)
    print("unsqueeze(b, 1) shape should be [2, 1, 3]:", b_unsqueeze_1)

    let b_unsqueeze_2 = b.unsqueeze(2)
    print("unsqueeze(b, 2) shape should be [2, 3, 1]:", b_unsqueeze_2)
    print()

    // Test 3: squeeze all dimensions - with method chaining!
    print("Test 3: squeeze all dimensions of size 1")
    let c_unsqueeze = a.unsqueeze(0).unsqueeze(2)
    print("Tensor with shape [1, 3, 1]:", c_unsqueeze)

    let c_squeezed = c_unsqueeze.squeeze()
    print("squeeze(c_unsqueeze) should be [3]:", c_squeezed)
    print()

    // Test 4: squeeze specific dimension
    print("Test 4: squeeze specific dimension")
    let d_squeezed = c_unsqueeze.squeeze(0)
    print("squeeze(c_unsqueeze, 0) should be [3, 1]:", d_squeezed)

    let d_squeezed_2 = c_unsqueeze.squeeze(2)
    print("squeeze(c_unsqueeze, 2) should be [1, 3]:", d_squeezed_2)
    print()

    // Test 5: round-trip (unsqueeze then squeeze) - with method chaining!
    print("Test 5: round-trip test")
    print("Original a:", a)
    let f = a.unsqueeze(0).squeeze(0)
    print("After unsqueeze(0).squeeze(0):", f)
    print("Should match original!")
    print()

    print("✅ All unsqueeze/squeeze tests complete!")
}
