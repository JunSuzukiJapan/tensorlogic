// Test print format string support

main {
    // Test 1: Simple string printing
    print("Hello, world!")

    // Test 2: Format string with single value
    let x = f32::ones([2.0])
    print("Tensor x: {}", x)

    // Test 3: Format string with multiple values
    let a = f32::from_scalar(42.0)
    let b = f32::from_scalar(3.14)
    print("Values: a={}, b={}", a, b)

    // Test 4: Simple mode - multiple values without format
    print(a, b)

    // Test 5: Mixed types
    let name = "TensorLogic"
    print("Welcome to {}", name)
}
