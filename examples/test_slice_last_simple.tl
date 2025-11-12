// Test slice_last() GPU implementation
// Simple test to verify correctness

main {
    print("=== Testing slice_last() ===")

    // Test 1: 2D tensor
    print("\n[Test 1] 2D tensor [3, 4]")
    let data_2d = vec(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    )
    let x_2d = from_vec(data_2d, shape(3, 4))
    print("  Input shape:", dims(x_2d))
    print("  Input data (first 4):", slice(data_2d, 0, 4))
    print("  Input data (last 4):", slice(data_2d, 8, 12))

    let last_2d = slice_last(x_2d, 0)
    print("  Output shape:", dims(last_2d))
    let result_2d = to_vec(last_2d)
    print("  Output data:", result_2d)
    print("  Expected: [9.0, 10.0, 11.0, 12.0]")

    // Test 2: 3D tensor
    print("\n[Test 2] 3D tensor [2, 3, 4]")
    let data_3d = vec(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        // Second slice (i=1)
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0
    )
    let x_3d = from_vec(data_3d, shape(2, 3, 4))
    print("  Input shape:", dims(x_3d))

    let last_3d = slice_last(x_3d, 0)
    print("  Output shape:", dims(last_3d))
    let result_3d = to_vec(last_3d)
    print("  Output data:", slice(result_3d, 0, 4))
    print("  Expected: [13.0, 14.0, 15.0, 16.0]")

    print("\n=== All tests complete ===")
}
