// Minimal test to identify GPU memory leak

main {
    print("=== Memory Leak Test ===")
    print("")

    // Just create a small tensor and let it go out of scope
    let x = f32::ones([10, 10])
    print("Created tensor:", shape(x))

    print("")
    print("=== Test Complete ===")
}
