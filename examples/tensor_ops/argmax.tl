// TensorLogic argmax/argmin Test
//
// Tests argmax and argmin operations

tensor a: float16[5] = [1.0, 5.0, 3.0, 2.0, 4.0]
tensor b: float16[2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
tensor c: float16[4] = [-5.0, -1.0, -3.0, -2.0]

main {
    print("=== TensorLogic argmax/argmin Test ===")
    print()

    // Test 1: Global argmax/argmin
    print("Test 1: Global argmax/argmin")
    print("Tensor a:", a)
    let max_idx = a.argmax()
    let min_idx = a.argmin()
    print("argmax (global):", max_idx)
    print("  Expected: 1 (value 5.0)")
    print("argmin (global):", min_idx)
    print("  Expected: 0 (value 1.0)")
    print()

    // Test 2: 2D tensor - argmax along dim 0
    print("Test 2: 2D tensor - argmax along dim 0")
    print("Tensor b shape [2, 3]:", b)
    let max_dim0 = b.argmax(0)
    print("argmax(dim=0):", max_dim0)
    print("  Expected: [1, 1, 1] (second row has larger values)")
    print()

    // Test 3: 2D tensor - argmax along dim 1
    print("Test 3: 2D tensor - argmax along dim 1")
    let max_dim1 = b.argmax(1)
    print("argmax(dim=1):", max_dim1)
    print("  Expected: [2, 2] (last column in each row)")
    print()

    // Test 4: argmin on same 2D tensor
    print("Test 4: argmin on 2D tensor")
    let min_dim0 = b.argmin(0)
    let min_dim1 = b.argmin(1)
    print("argmin(dim=0):", min_dim0)
    print("  Expected: [0, 0, 0] (first row has smaller values)")
    print("argmin(dim=1):", min_dim1)
    print("  Expected: [0, 0] (first column in each row)")
    print()

    // Test 5: keepdim parameter
    print("Test 5: keepdim parameter")
    let max_keepdim = b.argmax(1, true)
    print("argmax(dim=1, keepdim=true):", max_keepdim)
    print("  Shape should be [2, 1]")
    print()

    // Test 6: Negative values
    print("Test 6: Negative values")
    print("Tensor c:", c)
    let max_neg = c.argmax()
    let min_neg = c.argmin()
    print("argmax:", max_neg)
    print("  Expected: 1 (value -1.0 is max)")
    print("argmin:", min_neg)
    print("  Expected: 0 (value -5.0 is min)")
    print()

    print("✅ All argmax/argmin tests complete!")
}
