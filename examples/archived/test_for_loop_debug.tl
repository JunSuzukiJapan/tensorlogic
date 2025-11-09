// Debug for loop variable usage

main {
    print("=== Testing for loop variable ===")
    print("")

    print("Test 1: Simple for loop without using i:")
    for i in range(3) {
        print("  Iteration")
    }
    print("")

    print("Test 2: Using i in print:")
    for i in range(3) {
        print("  i =", i)
    }
    print("")

    print("Test 3: Using i in calculation:")
    let result = 0.0
    for i in range(5) {
        result + i
    }
    print("  Sum:", result)
    print("")
}
