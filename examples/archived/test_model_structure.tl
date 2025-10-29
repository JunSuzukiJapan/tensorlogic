main {
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    print("Model structure test:")
    print("Has token_embd?", model.token_embd)
    print("Has blk.0?", model.blk.0)
}
