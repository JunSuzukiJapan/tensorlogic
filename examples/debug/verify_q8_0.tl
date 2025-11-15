// Verify Q8_0 model against Candle reference
// Expected BOS sum: 0.0528666377

main {
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q8_0.gguf"
    let model = load_model(model_path)

    // Get embedding weight tensor
    let emb_weight = get_tensor(model, "token_embd.weight")

    // Get BOS token (ID=1) embedding
    let bos_emb = embedding(emb_weight, 1)

    // Calculate sum
    let bos_sum = sum(bos_emb)
    print("Q8_0 Model - BOS Token Sum:")
    print("  TensorLogic: " + str(bos_sum))
    print("  Candle Ref:  0.0528666377")

    let expected = 0.0528666377
    let diff = abs(bos_sum - expected)
    print("  Difference:  " + str(diff))

    let relative_error = diff / abs(expected) * 100.0
    print("  Relative error: " + str(relative_error) + "%")

    if diff < 0.00001 {
        print("  ✅ PASS: Excellent match!")
    } else {
        if diff < 0.0001 {
            print("  ⚠️  WARN: Good match")
        } else {
            print("  ❌ FAIL: Significant mismatch")
        }
    }
}
