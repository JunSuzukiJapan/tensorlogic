// Test: Mixed-type array error propagation
// Verify that mixed Integer+Float arrays produce proper error messages

main {
    print("=== Testing Mixed-Type Array Error Propagation ===")
    print("")

    // Test 1: Simple Integer variable + Float literal
    print("Test 1: Integer variable + Float literals")
    let x = 10
    print("  Creating array [x, 32.0, 64.0] where x =", x)
    print("  Expected: 'Array contains mixed types' error")
    print("")

    let mixed = [x, 32.0, 64.0]

    print("  ERROR: Should not reach here!")
    print("  mixed =", mixed)
}
