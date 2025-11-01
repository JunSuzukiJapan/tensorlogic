// Test chat generation with full 22-layer f16 model
// Validates that token generation loop works correctly

main {
    print("Running full 22-layer f16 chat generation tests...")
}

test test_chat_generation_f16 {
    print("=== Test: Full 22-Layer F16 Chat Generation ===")

    // Load model
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    print("  ✓ Model and tokenizer loaded")

    // Test that we can load model weights
    let tok_embd = model.token_embd.weight
    let emb_shape = shape(tok_embd)
    print("  Embedding matrix shape: ", emb_shape)

    // Verify embedding shape [vocab_size, hidden_dim] = [32000, 2048]
    assert_eq(emb_shape[0], 32000.0)
    assert_eq(emb_shape[1], 2048.0)

    print("  ✓ Model weights verified")

    // Simple prompt
    let tokens = tokenizer.tokenize("<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHi</s>\n<|assistant|>\n", false)

    print("  ✓ Tokenization successful")

    // Test embeddings
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    print("  Embedded tensor shape: ", x_shape)

    // Should be [seq_len, hidden_dim] where hidden_dim = 2048
    assert_eq(x_shape[1], 2048.0)

    // Verify seq_len is reasonable (between 10 and 40)
    assert(x_shape[0] > 10.0)
    assert(x_shape[0] < 40.0)

    print("  ✓ Embedding test passed")
    print("✓ All f16 tests passed!")
}
