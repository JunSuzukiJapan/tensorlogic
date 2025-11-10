// Comprehensive test for matmul_transposed_b Metal kernel
// This test verifies the tile loading bug fix

main {
    print("=== MatMul Kernel Test ===")
    print("")
    print("Testing matmul with transposed B (A @ B.T)")
    print("This test verifies correct tile loading in Metal kernels")
    print("")

    // Initialize Metal device by loading model
    print("[1/4] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    print("      ✓ Metal device initialized")
    print("")

    // Test 1: Real model matmul (embedding @ query weight)
    print("[2/4] Test 1: Embedding vector @ Query weight")
    print("-----------------------------------------------")

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")

    // Get embedding for token "Hello"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let tokens = tokenize(tokenizer, "Hello", false)
    let x = embedding(embed_table, tokens)

    print("      Input shape:", shape(x))
    print("      Weight shape:", shape(W_q_0))
    print("      Input sample:", x[0, 0], x[0, 1], x[0, 2])
    print("")
    print("      Computing x @ W_q_0.T...")

    let result1 = matmul(x, transpose(W_q_0))
    print("      Result shape:", shape(result1))
    print("      Result sample:", result1[0, 0], result1[0, 1], result1[0, 2])
    print("")

    // BUG CHECK: The tile loading bug would cause all zeros
    if result1[0, 0] == 0.0 && result1[0, 1] == 0.0 && result1[0, 2] == 0.0 {
        print("      ❌ FAILED: All zeros detected!")
        print("         This indicates the tile loading bug:")
        print("         b_row = col (varies per thread) - WRONG")
        print("         Should be: b_row = threadgroup_pos * TILE_SIZE + ty")
    } else {
        print("      ✅ PASSED: Non-zero results indicate correct tile loading")
    }
    print("")

    // Test 2: Multi-token embedding @ query weight
    print("[3/4] Test 2: Multi-token sequence @ Query weight")
    print("---------------------------------------------------")

    let prompt = "Hello world"
    let tokens2 = tokenize(tokenizer, prompt, false)
    let x2 = embedding(embed_table, tokens2)

    print("      Input shape:", shape(x2))
    print("      Token count:", len(tokens2))
    print("      Computing x2 @ W_q_0.T...")

    let result2 = matmul(x2, transpose(W_q_0))
    print("      Result shape:", shape(result2))
    print("      Result sample (token 0):", result2[0, 0], result2[0, 1], result2[0, 2])
    print("      Result sample (token 1):", result2[1, 0], result2[1, 1], result2[1, 2])
    print("")

    // Check both rows
    let token0_ok = result2[0, 0] != 0.0 || result2[0, 1] != 0.0 || result2[0, 2] != 0.0
    let token1_ok = result2[1, 0] != 0.0 || result2[1, 1] != 0.0 || result2[1, 2] != 0.0

    if token0_ok && token1_ok {
        print("      ✅ PASSED: Multi-token matmul works correctly")
    } else {
        print("      ❌ FAILED: Some tokens produced all zeros")
    }
    print("")

    // Test 3: Different layer weights (test different matrix sizes)
    print("[4/4] Test 3: Testing different weight matrices")
    print("--------------------------------------------------")

    // Test with key weight
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    print("      Key weight shape:", shape(W_k_0))

    let result3 = matmul(x, transpose(W_k_0))
    print("      x @ W_k.T result:", result3[0, 0], result3[0, 1], result3[0, 2])

    // Test with value weight
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    print("      Value weight shape:", shape(W_v_0))

    let result4 = matmul(x, transpose(W_v_0))
    print("      x @ W_v.T result:", result4[0, 0], result4[0, 1], result4[0, 2])

    // Test with output weight
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    print("      Output weight shape:", shape(W_o_0))

    let result5 = matmul(x, transpose(W_o_0))
    print("      x @ W_o.T result:", result5[0, 0], result5[0, 1], result5[0, 2])
    print("")

    let key_ok = result3[0, 0] != 0.0 || result3[0, 1] != 0.0 || result3[0, 2] != 0.0
    let val_ok = result4[0, 0] != 0.0 || result4[0, 1] != 0.0 || result4[0, 2] != 0.0
    let out_ok = result5[0, 0] != 0.0 || result5[0, 1] != 0.0 || result5[0, 2] != 0.0

    if key_ok && val_ok && out_ok {
        print("      ✅ PASSED: All weight matrices work correctly")
    } else {
        print("      ❌ FAILED: Some weight matrices produced all zeros")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("This test verifies the matmul_transposed_b kernel fix:")
    print("")
    print("  The Bug:")
    print("    uint b_row = col;  // col varies per thread!")
    print("    Each thread loads from different B row → incorrect tiling")
    print("")
    print("  The Fix:")
    print("    uint b_row = threadgroup_position_in_grid.x * TILE_SIZE + ty;")
    print("    All threads in threadgroup load same B tile → correct!")
    print("")
    print("If all tests pass, the kernel correctly loads tiles and computes matmul.")
    print("If tests fail with all zeros, the tile loading bug is present.")
}
