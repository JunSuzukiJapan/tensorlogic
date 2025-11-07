// Test break statement

main {
    print("=== Test 1: Simple break in while loop ===")

    let counter = 0
    while counter < 100 {
        print("Counter:", counter)
        counter = counter + 2

        if counter >= 10 {
            print("Breaking at counter =", counter)
            break
        }
    }

    print("Final counter:", counter)
    print("✅ Break worked correctly!")
    print("")

    print("=== Test 2: Break in nested loop ===")
    let i = 0
    let j = 0  // Declare j outside the loops

    while i < 5 {
        print("Outer loop i =", i)

        // Reset j for each outer loop iteration
        j = 0

        while j < 10 {
            print("  Inner loop j =", j)
            j = j + 1

            if j >= 3 {
                print("  Breaking inner loop")
                break
            }
        }

        i = i + 1
    }

    print("✅ Nested break test complete!")
}
