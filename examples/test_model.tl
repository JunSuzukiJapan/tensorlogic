// Test model loading

main {
    print("Testing load_model():")
    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    print("Model path:", model_path)
    
    let model = load_model(model_path)
    print("Model loaded successfully!")
}
