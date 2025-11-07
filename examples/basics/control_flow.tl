// Test control flow implementation

main {
    print("=== Control Flow Test ===")

    // Test 1: If-else
    print("\n1. If-else test:")
    let x = [5.0]

    if x[0] > 3.0 {
        print("  x[0] > 3.0: TRUE")
    } else {
        print("  x[0] > 3.0: FALSE")
    }

    // Test 2: For loop with range
    print("\n2. For loop with range(5):")
    for i in range(5) {
        print("  i =", i)
    }

    // Test 3: While loop
    print("\n3. While loop test:")
    let counter = [0.0]

    while counter[0] < 3.0 {
        print("  counter =", counter[0])
        counter = counter + [1.0]
    }

    print("\n=== All control flow tests passed! ===")
}
