// Simple test to verify for loop functionality

main {
    print("Test 1: Basic for loop with range()")
    for i in range(5) {
        print("  i = ", i)
    }

    print("\nTest 2: Nested for loop")
    for i in range(3) {
        for j in range(2) {
            print("  (", i, ", ", j, ")")
        }
    }

    print("\nTest 3: For loop with break")
    for i in range(10) {
        print("  i = ", i)
        if i == 3 {
            print("  Breaking at i=3")
            break
        }
    }

    print("\nAll tests completed!")
}
