// Test new built-in functions

// Test env() function
let home = env("HOME")

main {
    print("Home directory:", home)

    // Test load_model() function
    let model_path = home + "/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"
    print("Loading model from:", model_path)

    let model = load_model(model_path)
    print("Model loaded successfully!")

    // Test input() function
    print("What is your name?")
    let name = input()
    print("Hello,", name, "!")

    print("Test completed!")
}
