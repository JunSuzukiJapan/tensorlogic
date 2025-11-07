// Test to_vec() performance/hang

main {
    print("Testing to_vec() on large tensor...")

    // Create a large tensor similar to logits: [47, 32000]
    let t = zeros_f32([47.0, 32000.0])
    print("Created tensor shape [47, 32000]")

    // Try to convert to vec (this might hang)
    print("Calling temperature_sample...")
    let token = temperature_sample(t, 0.8)
    print("Got token:", token)

    print("Test complete!")
}
