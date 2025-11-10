// Test forward pass with accumulated tokens
// Isolate where the computation breaks

// Simple transformer layer (minimal version)
fn simple_transformer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    attn_norm: float16[?]
) -> float16[?, ?] {
    let x_norm = rms_norm(x, attn_norm)
    print("        [DEBUG] x_norm sample:", x_norm[0, 0])
    let result = linear(x_norm, W_q)
    print("        [DEBUG] linear output sample:", result[0, 0])
    result
}

main {
    print("=== Forward Pass Accumulation Test ===")
    print("")

    // Load model
    print("[1/4] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Loaded")
    print("")

    // Get weights
    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Test 1: Initial prompt (39 tokens)
    print("[2/4] Testing initial prompt (39 tokens)...")
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello! How are you?</s>\n<|assistant|>\n"
    let tokens1 = tokenize(tokenizer, prompt, true)
    print("      Tokens:", len(tokens1))

    let e1 = embedding(embed_table, tokens1)
    print("      e1 shape:", shape(e1))
    print("      e1 sample:", e1[0, 0])

    let h1 = simple_transformer(e1, W_q_0, attn_norm_0)
    print("      h1 shape:", shape(h1))
    print("      h1 sample:", h1[0, 0])

    let logits1 = linear(h1, output_weight)
    print("      logits1 shape:", shape(logits1))
    print("      logits1[0] (first position):", logits1[0, 0], logits1[0, 1], logits1[0, 2])

    // Check LAST position (what matters for generation)
    let last_pos1 = 38
    print("      logits1[38] (last position):", logits1[last_pos1, 0], logits1[last_pos1, 1], logits1[last_pos1, 2])

    if logits1[last_pos1, 0] == 0.0 {
        print("      ❌ Last position logits are zero!")
    } else {
        print("      ✅ Forward pass 1 complete")
    }
    print("")

    // Test 2: Accumulated tokens (39 + 1 = 40 tokens)
    print("[3/4] Testing accumulated tokens (40 tokens)...")
    let new_token = 100
    let tokens2 = append(tokens1, new_token)
    print("      Tokens:", len(tokens2))

    let e2 = embedding(embed_table, tokens2)
    print("      e2 shape:", shape(e2))
    print("      e2 sample:", e2[0, 0])

    let h2 = simple_transformer(e2, W_q_0, attn_norm_0)
    print("      h2 shape:", shape(h2))
    print("      h2 sample:", h2[0, 0])

    let logits2 = linear(h2, output_weight)
    print("      logits2 shape:", shape(logits2))
    print("      logits2[0] (first position):", logits2[0, 0], logits2[0, 1], logits2[0, 2])

    // Check LAST position
    let last_pos2 = 39
    print("      logits2[39] (last position):", logits2[last_pos2, 0], logits2[last_pos2, 1], logits2[last_pos2, 2])

    if logits2[last_pos2, 0] == 0.0 {
        print("      ❌ BUG: Last position logits are zero!")
    } else {
        print("      ✅ Forward pass 2 complete")
    }
    print("")

    // Test 3: Third accumulation (40 + 1 = 41 tokens)
    print("[4/4] Testing second accumulation (41 tokens)...")
    let new_token2 = 200
    let tokens3 = append(tokens2, new_token2)
    print("      Tokens:", len(tokens3))

    let e3 = embedding(embed_table, tokens3)
    print("      e3 shape:", shape(e3))
    print("      e3 sample:", e3[0, 0])

    let h3 = simple_transformer(e3, W_q_0, attn_norm_0)
    print("      h3 shape:", shape(h3))
    print("      h3 sample:", h3[0, 0])

    let logits3 = linear(h3, output_weight)
    print("      logits3 shape:", shape(logits3))
    print("      logits3[0] (first position):", logits3[0, 0], logits3[0, 1], logits3[0, 2])

    // Check LAST position
    let last_pos3 = 40
    print("      logits3[40] (last position):", logits3[last_pos3, 0], logits3[last_pos3, 1], logits3[last_pos3, 2])

    if logits3[last_pos3, 0] == 0.0 {
        print("      ❌ BUG: Last position logits are zero!")
    } else {
        print("      ✅ Forward pass 3 complete")
    }
    print("")

    print("=== Test Complete ===")
}
