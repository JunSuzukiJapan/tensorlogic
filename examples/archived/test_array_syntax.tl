// Test array index syntax for model weights: model.blk[0].attn_norm.weight

main {
    print("=== Testing Array Index Syntax ===")
    print("")
    
    // Load model  
    let model = load_model_f32("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    print("OLD (removed): get_tensor(model, \"blk.0.attn_norm.weight\")")
    print("NEW: model.blk[0].attn_norm.weight")
    print("")
    
    print("Testing: model.blk[0].attn_norm.weight")
    let layer_0_attn_norm = model.blk[0].attn_norm.weight
    print("✓ Success! Got layer 0 attention norm")
    print("")
    
    print("Testing: model.blk[1].attn_q.weight")
    let layer_1_q = model.blk[1].attn_q.weight
    print("✓ Success! Got layer 1 query weight")
    print("")
    
    print("=== Array index syntax working! ===")
}
