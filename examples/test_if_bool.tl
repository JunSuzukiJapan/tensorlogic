// Test if statement with boolean variables

main {
    print("Test 1: If with boolean outside loop")
    let flag = true
    if flag {
        print("  ✓ Flag is true (outside loop)")
    } else {
        print("  ✗ Flag is false (outside loop)")
    }

    print("\nTest 2: If with boolean inside loop")
    let flag2 = true
    for i in range(3) {
        print("Iteration ", i, ", flag2 = ", flag2)
        if flag2 {
            print("  ✓ Flag2 is true (inside loop)")
        } else {
            print("  ✗ Flag2 is false (inside loop)")
        }
    }

    print("\nTest 3: If with literal true inside loop")
    for i in range(3) {
        print("Iteration ", i)
        if true {
            print("  ✓ Literal true works")
        }
    }

    print("\nAll tests completed")
}
