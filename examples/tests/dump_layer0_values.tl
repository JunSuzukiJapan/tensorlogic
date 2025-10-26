// Dump numerical values from Layer 0 forward pass
// This helps compare with Python reference implementation

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

main {
    print("=== Layer 0 Forward Pass - Numerical Dump ===\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    // Token ID 15043 = "Hello"
    let tokens = tokenize(tokenizer, "Hello", false)
    print("Input token: ", tokens, "\n")

    // Get embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let h = embedding(embed_table, tokens)  // [1, 2048]

    print("Step 1: Embedding")
    print("  Shape: ", shape(h))
    // Note: TensorLogic doesn't have easy way to print 2D tensor values
    // We'll just verify shapes match expected
    print("")

    // Layer 0 weights
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")

    // Attention normalization
    print("Step 2: RMS Normalization (attn)")
    let h_norm = rms_norm(h, attn_norm_0)
    print("  Shape: ", shape(h_norm))
    print("")

    // Q, K, V projections
    print("Step 3: Q, K, V Projections")
    let Q = linear(h_norm, W_q_0)  // [1, 2048]
    let K = linear(h_norm, W_k_0)  // [1, 256]
    let V = linear(h_norm, W_v_0)  // [1, 256]

    print("  Q shape: ", shape(Q))
    print("  K shape: ", shape(K))
    print("  V shape: ", shape(V))
    print("")

    // Reshape to heads
    print("Step 4: Reshape to heads")
    let Q_heads = reshape(Q, [1.0, 32.0, 64.0])
    let K_heads = reshape(K, [1.0, 4.0, 64.0])
    let V_heads = reshape(V, [1.0, 4.0, 64.0])

    print("  Q_heads shape: ", shape(Q_heads))
    print("  K_heads shape: ", shape(K_heads))
    print("  V_heads shape: ", shape(V_heads))
    print("")

    // RoPE
    print("Step 5: Apply RoPE (position=0)")
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    print("  Q_rope shape: ", shape(Q_rope))
    print("  K_rope shape: ", shape(K_rope))
    print("")

    // GQA expansion
    print("Step 6: GQA expansion (4 -> 32 heads)")
    let K_reshaped = reshape(K_rope, [1.0, 4.0, 1.0, 64.0])
    let K_expanded = broadcast_to(K_reshaped, [1.0, 4.0, 8.0, 64.0])
    let K_final = reshape(K_expanded, [1.0, 32.0, 64.0])

    let V_reshaped = reshape(V_heads, [1.0, 4.0, 1.0, 64.0])
    let V_expanded = broadcast_to(V_reshaped, [1.0, 4.0, 8.0, 64.0])
    let V_final = reshape(V_expanded, [1.0, 32.0, 64.0])

    print("  K_final shape: ", shape(K_final))
    print("  V_final shape: ", shape(V_final))
    print("")

    // Attention
    print("Step 7: Attention")
    // Q_rope: [1, 32, 64]
    // K_final: [1, 32, 64]
    // V_final: [1, 32, 64]
    // Compute attention scores: Q @ K.T
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_final)  // [1, 32, 1]
    print("  scores shape: ", shape(scores))

    // Scale by 1/sqrt(64) = 0.125
    let scaled = scores * 0.125
    print("  scaled shape: ", shape(scaled))

    // Softmax
    let attn_weights = softmax(scaled, 2)
    print("  attn_weights shape: ", shape(attn_weights))

    // Weighted sum: attn @ V
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_final)  // [1, 32, 64]

    print("  attn_output shape: ", shape(attn_output))
    print("")

    // Output projection
    print("Step 8: Output projection")
    let attn_reshaped = reshape(attn_output, [1.0, 2048.0])
    let attn_proj = linear(attn_reshaped, W_o_0)

    print("  attn_proj shape: ", shape(attn_proj))
    print("")

    // Residual
    print("Step 9: Residual connection")
    let h1 = h + attn_proj

    print("  h1 shape: ", shape(h1))
    print("")

    // FFN
    print("Step 10: FFN normalization")
    let h1_norm = rms_norm(h1, ffn_norm_0)

    print("  h1_norm shape: ", shape(h1_norm))
    print("")

    print("Step 11: SwiGLU FFN")
    let gate = linear(h1_norm, W_gate_0)
    let up = linear(h1_norm, W_up_0)
    let silu_gate = silu(gate)
    let intermediate = silu_gate * up
    let ffn_output = linear(intermediate, W_down_0)

    print("  ffn_output shape: ", shape(ffn_output))
    print("")

    // Final residual
    print("Step 12: Final residual")
    let h2 = h1 + ffn_output

    print("  h2 (layer 0 output) shape: ", shape(h2))
    print("")

    print("=== Layer 0 Forward Pass Complete ===")
}
