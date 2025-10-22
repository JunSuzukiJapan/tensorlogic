// Test new built-in functions
// Note: Variables must be declared at top level, not inside main

main {
    // Test 1: env() function
    print("=== Testing env() ===")
    print("HOME directory:", env("HOME"))
    print("USER:", env("USER"))
    
    // Test 2: input() function  
    print("\n=== Testing input() ===")
    print("Enter your name:")
    print("You entered:", input())
    
    // Test 3: load_model() function
    print("\n=== Testing load_model() ===")
    print("Attempting to load TinyLlama model...")
    // Note: This will only work if you have downloaded the model
    // Run: cargo run --bin download_model -- --model tinyllama
    // load_model(env("HOME") + "/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    print("\n✅ All basic tests completed!")
}
