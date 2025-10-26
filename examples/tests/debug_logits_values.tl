// Debug logits computation - check intermediate values

main {
    print("=== Logits Computation Debug ===\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("output_norm shape: ", shape(output_norm), "\n")
    print("output_weight shape: ", shape(output_weight), "\n\n")

    // Create a simple test hidden state: [1, 2048] with known values
    // Use values close to what a real model might output
    let test_hidden = ones([1.0, 2048.0]) * 0.5
    print("Test hidden state shape: ", shape(test_hidden), "\n")
    print("Test hidden first 5 values: should be [0.5, 0.5, 0.5, 0.5, 0.5]\n\n")

    // Apply RMS norm
    let normed = rms_norm(test_hidden, output_norm)
    print("After RMS norm shape: ", shape(normed), "\n")
    print("After RMS norm: check if values are reasonable\n\n")

    // Apply output projection
    let logits = linear(normed, output_weight)
    print("Logits shape: ", shape(logits), "\n")
    print("Logits: should have varying values, not all the same\n\n")

    print("Next: Test with actual model hidden state from token 'Hello' (15043)\n")
}
