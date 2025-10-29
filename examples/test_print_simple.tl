// Simple print test

main {
    // Test 1: Simple string
    print("Hello, world!")

    // Test 2: String variable
    let name = "TensorLogic"
    print("Welcome to {}", name)

    // Test 3: Simple mode without format
    print("A", "B", "C")

    // Test 4: Tensor
    let x = f32::ones([2.0])
    print("Tensor:", x)
}
