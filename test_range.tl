// Test range() function

main {
    print("=== Testing zeros and range functions ===")
    print("")

    // Test zeros first (known to work)
    print("[Test 0] zeros([3.0]):")
    let z = zeros([3.0])
    let sz = shape(z)
    print("  Shape:", sz)
    print("")

    // Test 1: arange(n)
    print("[Test 1] arange(5):")
    let r1 = arange(5)
    let s1 = shape(r1)
    print("  Shape:", s1)
    print("  Expected: [5]")
    print("")

    // Test 2: arange(start, end)
    print("[Test 2] arange(10, 15):")
    let r2 = arange(10, 15)
    let s2 = shape(r2)
    print("  Shape:", s2)
    print("  Expected: [5]")
    print("")

    print("=== Tests complete ===")
}
