// Simple Local LLM Chat Demo
// Uses TinyLlama model for conversational AI

main {
    print("=======================================================================")
    print("TensorLogic Local LLM Chat")
    print("=======================================================================")
    print("")
    
    // Load the model
    let home = env("HOME")
    let model_path = home + "/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"
    
    print("Loading TinyLlama model...")
    print("Model path:", model_path)
    let model = load_model(model_path)
    print("Model loaded successfully!")
    print("")
    
    print("Chat Commands:")
    print("  Type 'quit' or 'exit' to end the chat")
    print("  Type 'help' for this help message")
    print("")
    print("=======================================================================")
    print("")
    
    // Simple chat loop
    loop {
        // Get user input
        let user_input = input("You: ")
        
        // Check for exit commands
        let is_exit = user_input == "quit"
        let is_exit2 = user_input == "exit"
        let should_exit = is_exit || is_exit2
        
        if should_exit {
            print("")
            print("Goodbye! Thanks for chatting.")
            break
        }
        
        // Check for help
        let is_help = user_input == "help"
        if is_help {
            print("")
            print("Commands:")
            print("  quit/exit - End the chat")
            print("  help - Show this message")
            print("")
        }
        
        // For now, just echo back (actual LLM inference will be added later)
        print("AI: I received your message: \"" + user_input + "\"")
        print("    (Note: Full LLM inference not yet implemented)")
        print("")
    }
}
