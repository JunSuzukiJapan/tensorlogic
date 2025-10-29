// Test new property access syntax for model weights

main {
    print("=== Testing new property access syntax ===")
    
    // Load model  
    let model = load_model_f32("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    // The old way (now removed):
    // let tok_emb = get_tensor(model, "token_embd.weight")
    
    // The new way:
    print("✓ Testing: model.token_embd.weight")
    let tok_emb = model.token_embd.weight
    print("✓ Success! Got token embedding tensor")
    
    print("\n=== Property access syntax working! ===")
}
