// Test variable assignment with = and =

main {
    print("Test 1: Assignment with = inside loop")
    let counter = 0
    for i in range(5) {
        counter = counter + 1
        print("  counter = ", counter)
    }
    print("Final counter:", counter)

    print("\nTest 2: Assignment with = inside loop")
    let counter2 = 0
    for i in range(5) {
        counter2 = counter2 + 1
        print("  counter2 = ", counter2)
    }
    print("Final counter2:", counter2)

    print("\nTest 3: Boolean assignment with =")
    let flag = true
    for i in range(3) {
        print("  Iteration ", i, ", flag = ", flag)
        if i == 1 {
            flag = false
            print("  Set flag to false")
        }
    }
    print("Final flag:", flag)

    print("\nAll tests completed!")
}
