// TensorLogic Generate Function Test
//
// Tests the basic generate() function with a model
// This demonstrates the API, though full inference is not yet implemented

main {
    print("=== TensorLogic Generate Function Test ===\n")

    // Get model path
    let home_dir = env("HOME")
    let model_path = home_dir + "/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"

    print("Loading model:", model_path)

    // Load the model
    let model = load_model(model_path)
    print("Model loaded successfully!")
    print()

    // Test generate function
    print("Testing generate() function...")
    let prompt = "Hello, how are you?"
    print("Prompt:", prompt)
    print()

    let response = generate(model, prompt, 50, 0.7)
    print("Response:", response)
    print()

    print("✅ Generate function test complete!")
    print()
    print("Note: Full transformer inference will be implemented in the future.")
    print("This includes tokenization, attention mechanisms, and proper sampling.")
}
