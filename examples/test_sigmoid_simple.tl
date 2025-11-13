// Simple sigmoid test with -6208 value

main {
    print("Testing sigmoid(-6208)...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Create a simple 1x1 tensor with -6208
    let emb_weight = get_tensor(model, "token_embd.weight")

    // Get first embedding vector and scale it to get large negative values
    let tokens = [1, 2, 3, 4, 5]
    let emb = embedding(emb_weight, tokens)

    // Create large negative values by multiplication
    let large_neg = emb * (-10.0)

    print("Input sum:", sum(large_neg))

    let sig = sigmoid(large_neg)
    print("Sigmoid sum:", sum(sig))
    print("Sigmoid shape:", shape(sig))

    print("Test completed")
}
