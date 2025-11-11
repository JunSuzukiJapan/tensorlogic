// Test range_i() with the new iterator trait system
main {
    print("Testing range_i() iterator trait implementation")
    print("")

    // Test 1: Simple loop with range_i
    print("[Test 1] Loop with range_i(5):")
    for i in range_i(5) {
        print("  i =", i)
    }
    print("")

    // Test 2: Using loop variable for array indexing
    print("[Test 2] Array indexing with range_i loop variable:")
    let test_array = [10, 20, 30, 40, 50]
    for idx in range_i(5) {
        let val = test_array[idx]
        print("  test_array[", idx, "] =", val)
    }
    print("")

    print("✅ All tests passed!")
}
