// Test Display implementation for small tensors

main {
    print("=== Testing Tensor Display Implementation ===")
    print("")

    // Create a small 1D tensor directly to test display
    print("Test 1: Direct tensor display")
    let dims_tensor = zeros([2.0])
    print("  dims_tensor: {}", dims_tensor)

    print("")
    print("Test 2: shape() result display")
    let test_tensor = zeros([32000.0, 2048.0])
    let shape_result = shape(test_tensor)
    print("  shape_result: {}", shape_result)

    print("")
    print("Test 3: Check shape() result properties")
    let shape_of_shape = shape(shape_result)
    print("  shape(shape_result): {}", shape_of_shape)

    print("")
    print("EXPECTED:")
    print("  - Test 1: Should show [0, 0] or similar data")
    print("  - Test 2: Should show [32000, 2048]")
    print("  - Test 3: Should show [2]")
}
