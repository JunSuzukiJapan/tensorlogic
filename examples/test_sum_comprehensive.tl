// Comprehensive test for sum() function at TL script level

main {
    print("================================================================================")
    print("🔍 Comprehensive sum() Function Test")
    print("================================================================================")
    print("")

    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("")

    // Create a base set of tokens for testing
    let test_prompt_short = "Hello"
    let test_tokens_short = tokenize(tokenizer, test_prompt_short, false)
    let test_emb_short = embedding(emb_weight, test_tokens_short)

    // Test 1: Sum of small positive values
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 1: Sum of small positive values")
    print("──────────────────────────────────────────────────────────────────────────")
    let emb1 = test_emb_short
    let sum1 = sum(emb1)
    print("  Embedding shape:", shape(emb1))
    print("  Sum:", sum1)
    print("  Expected: Small positive value")
    print("")

    // Test 2: Sum after sigmoid on small tensor
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 2: Sum of sigmoid on small tensor")
    print("──────────────────────────────────────────────────────────────────────────")
    let emb2 = test_emb_short
    let sig2 = sigmoid(emb2)
    print("  Input shape:", shape(emb2))
    print("  Input sum:", sum(emb2))
    print("  Sigmoid sum:", sum(sig2))
    print("  Expected: Value in range [0, size*seq_len]")
    print("")

    // Test 3: Sum after sigmoid on large negative values (FFN-like)
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 3: Sum of sigmoid on large negative values")
    print("──────────────────────────────────────────────────────────────────────────")
    let test_prompt_10 = "Hello world, how are you today? Great!"
    let test_tokens_10 = tokenize(tokenizer, test_prompt_10, false)
    let emb3 = embedding(emb_weight, test_tokens_10)
    let large_neg = emb3 * (-1000.0)
    print("  Large negative sum:", sum(large_neg))
    let sig3 = sigmoid(large_neg)
    print("  Sigmoid of large negative sum:", sum(sig3))
    print("  Expected: Small positive value close to 0")
    print("")

    // Test 4: Sum after linear transformation (FFN gate simulation)
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 4: Sum after linear transformation (FFN simulation)")
    print("──────────────────────────────────────────────────────────────────────────")
    let layer_idx = 0
    let W_gate = model.blk[layer_idx].ffn_gate.weight

    let emb4 = test_emb_short
    let ffn_norm = model.blk[layer_idx].ffn_norm.weight
    let x_norm = rms_norm(emb4, ffn_norm)

    print("  Input sum:", sum(x_norm))

    let gate = linear(x_norm, W_gate)
    print("  Gate sum:", sum(gate))
    print("  Gate shape:", shape(gate))

    let sig_gate = sigmoid(gate)
    print("  Sigmoid(gate) sum:", sum(sig_gate))
    print("  Expected: Finite positive value")
    print("")

    // Test 5: Sum of very small values (close to sigmoid output)
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 5: Sum of manually scaled small values")
    print("──────────────────────────────────────────────────────────────────────────")
    let emb5 = test_emb_short
    let tiny = emb5 * 0.0001
    print("  Tiny values sum:", sum(tiny))
    print("  Expected: Very small but finite")
    print("")

    // Test 6: Sum after multiple operations
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 6: Sum after sigmoid and multiplication")
    print("──────────────────────────────────────────────────────────────────────────")
    let emb6 = test_emb_short
    let neg6 = emb6 * (-100.0)
    let sig6 = sigmoid(neg6)
    let result6 = neg6 * sig6
    print("  Neg sum:", sum(neg6))
    print("  Sigmoid sum:", sum(sig6))
    print("  Neg * Sigmoid sum:", sum(result6))
    print("  Expected: All finite")
    print("")

    // Test 7: Large tensor sum (FFN size)
    print("──────────────────────────────────────────────────────────────────────────")
    print("TEST 7: Large tensor sum (46 tokens)")
    print("──────────────────────────────────────────────────────────────────────────")
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n"
    let tokens7 = tokenize(tokenizer, prompt, false)
    print("  Token count:", len(tokens7))
    let emb7 = embedding(emb_weight, tokens7)
    print("  Embedding sum:", sum(emb7))

    let x_norm7 = rms_norm(emb7, ffn_norm)
    let gate7 = linear(x_norm7, W_gate)
    print("  Gate sum:", sum(gate7))
    print("  Gate shape:", shape(gate7))

    let sig_gate7 = sigmoid(gate7)
    print("  Sigmoid(gate) sum:", sum(sig_gate7))
    print("  Expected: Finite positive value")
    print("")

    print("================================================================================")
    print("✅ Comprehensive sum() Test Completed")
    print("================================================================================")
}
