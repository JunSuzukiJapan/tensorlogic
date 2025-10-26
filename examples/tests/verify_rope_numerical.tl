// RoPE Numerical Verification with Small Input
// Create a simple test case where we can manually verify the math

main {
    print("=== RoPE Numerical Verification ===\n")

    // Create a simple 2D input: [1, 2, 4]
    // (1 sequence, 2 heads, 4 dimensions)
    // Using small values we can verify manually

    // Head 0: [1.0, 2.0, 3.0, 4.0]
    // Head 1: [5.0, 6.0, 7.0, 8.0]

    // First, let's understand what rope() does
    // rope_base = 10000
    // For position = 0:
    //   freq_0 = 1.0 / 10000^(0/4) = 1.0
    //   freq_1 = 1.0 / 10000^(2/4) = 1.0 / 100 = 0.01
    //   theta_0 = position * freq_0 = 0 * 1.0 = 0
    //   theta_1 = position * freq_1 = 0 * 0.01 = 0
    //   cos(0) = 1.0, sin(0) = 0.0
    //
    // For each dimension pair (d0, d1):
    //   out_d0 = d0 * cos(theta) - d1 * sin(theta) = d0 * 1.0 - d1 * 0.0 = d0
    //   out_d1 = d0 * sin(theta) + d1 * cos(theta) = d0 * 0.0 + d1 * 1.0 = d1
    //
    // So at position=0, RoPE should preserve the input!

    // We need actual model to use rope() function
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Get Q weight to create a proper tensor
    let W_q = get_tensor(model, "blk.0.attn_q.weight")

    print("Creating test tensor...")
    // We'll create a tiny custom input for verification
    // Unfortunately TensorLogic doesn't have easy tensor creation from literals for 3D
    // So we'll use the embedding approach with known token

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Use token 0 (should be <unk> or similar)
    let tokens = tokenize(tokenizer, "a", false)
    print("Token: ", tokens)

    let embed_table = get_tensor(model, "token_embd.weight")
    let h = embedding(embed_table, tokens)  // [1, 2048]

    print("Embedding shape: ", shape(h))

    // Project to Q
    let Q = linear(h, W_q)  // [1, 2048]
    print("Q shape: ", shape(Q))

    // Reshape to heads
    let Q_heads = reshape(Q, [1.0, 32.0, 64.0])
    print("Q_heads shape: ", shape(Q_heads))

    // Apply RoPE at position 0
    print("\nApplying RoPE at position 0...")
    let Q_rope = rope(Q_heads)
    print("Q_rope shape: ", shape(Q_rope))

    print("\n📝 Verification:")
    print("At position=0, RoPE should approximately preserve input")
    print("because cos(0)=1 and sin(0)=0")
    print("")
    print("Expected: Q_rope ≈ Q_heads (within numerical precision)")
    print("")
    print("To numerically verify, we would need to:")
    print("1. Extract actual values from Q_heads")
    print("2. Extract actual values from Q_rope")
    print("3. Compare them element-wise")
    print("")
    print("⚠️  TensorLogic currently doesn't have a built-in way")
    print("    to extract/print individual tensor values in scripts")
    print("    This would require adding a debug function like:")
    print("    print_values(tensor, start, count)")
}
