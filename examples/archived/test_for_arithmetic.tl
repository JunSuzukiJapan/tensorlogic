// Test arithmetic in for loop

main {
    print("=== Testing arithmetic in for loop ===")
    print("")

    print("Test 1: Simple arithmetic:")
    for i in range(3) {
        let val = 20.0 + i
        print("  20.0 + i =", val)
    }
    print("")

    print("Test 2: f32::ones with arithmetic:")
    for i in range(3) {
        let val = 20.0 + i
        let pos = f32::ones([1]) * val
        print("  pos shape:", shape(pos))
    }
    print("")

    print("Test 3: Direct arithmetic in expression:")
    for i in range(3) {
        let pos = f32::ones([1]) * (20.0 + i)
        print("  pos shape:", shape(pos))
    }
    print("")
}
