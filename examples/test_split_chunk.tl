// Test for split and chunk operations

tensor a: float16[6, 4] = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0
]

tensor b: float16[7, 4] = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0
]

main {
    print("=== TensorLogic split/chunk Test ===")
    print()

    // Test 1: split with even division
    print("Test 1: split with even division")
    print("Original tensor a shape [6, 4]:", a)
    let split1 = split(a, 2, 0)
    print("split(a, 2, 0) - first split [2, 4]:", split1)
    print("  Expected: First 2 rows [1-8]")
    print()

    // Test 2: split with uneven division
    print("Test 2: split with uneven division")
    print("Original tensor b shape [7, 4]:", b)
    let split2 = split(b, 3, 0)
    print("split(b, 3, 0) - first split [3, 4]:", split2)
    print("  Expected: First 3 rows [1-12]")
    print()

    // Test 3: chunk operation
    print("Test 3: chunk operation")
    let chunk1 = chunk(a, 3, 0)
    print("chunk(a, 3, 0) - first chunk [2, 4]:", chunk1)
    print("  Expected: First 2 rows [1-8]")
    print()

    // Test 4: split along different dimension
    print("Test 4: split along dim 1")
    let split3 = split(a, 2, 1)
    print("split(a, 2, 1) - first split [6, 2]:", split3)
    print("  Expected: First 2 columns of all rows")
    print()

    print("✅ All split/chunk tests complete!")
    print("Note: Only first split/chunk is returned (simplified implementation)")
}
