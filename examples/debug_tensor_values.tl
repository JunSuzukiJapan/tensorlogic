// Debug script to test field access
main {
    print("Loading model as f16...")
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    print("✓ Model loaded")

    print("Accessing token_embd...")
    let tok_embd = model.token_embd.weight
    print("✓ token_embd accessed")

    print("Getting shape...")
    let embd_shape = shape(tok_embd)
    print("✓ Shape obtained: [{}, {}]", embd_shape[0], embd_shape[1])
}
