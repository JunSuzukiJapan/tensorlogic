// Test new property access syntax - demonstrate the improvement

main {
    print("=== Property Access Syntax Demo ===")
    print("")
    
    // Load model  
    let model = load_model_f32("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    print("Testing nested property access:")
    print("  model.blk.0.attn_norm.weight")
    let layer_0_attn_norm = model.blk.0.attn_norm.weight
    print("  ✓ Success!")
    print("")
    
    print("=== Property access is working! ===")
    print("No more get_tensor(model, string) calls!")
}
