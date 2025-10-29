// Test shape() and indexing

main {
    print("=== Testing shape() and indexing ===")

    // Create simple tensor
    print("[1/2] Creating tensor...")
    let test_array = [1.0, 2.0, 3.0, 4.0]
    print("      Array:", test_array)

    // Get shape (should return [4])
    print("[2/2] Getting shape...")
    let s = shape(test_array)
    print("      Got shape tensor")

    // Try to index it
    print("      Accessing shape[0]...")
    let dim0 = s[0]
    print("      shape[0] =", dim0)

    print("=== Test Complete ===")
}
