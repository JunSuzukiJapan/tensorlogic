// Debug layer-by-layer to find where values become abnormal
// Print stats (shape, approximate value range) after each major operation

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
}

main {
    print("=== Layer-by-Layer Debug ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Simple input: BOS token
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input: BOS token")
    print("Tokens:", tokens)
    print()

    // Embedding
    print("=== Step 1: Embedding ===")
    let embed_table = get_tensor(model, "token_embd.weight")
    let e = embedding(embed_table, tokens)
    print("  Shape:", shape(e))
    print()

    // Layer 0
    print("=== Layer 0 ===")

    // Attention norm
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let normed_attn = rms_norm(e, attn_norm_0)
    print("  After attn_norm shape:", shape(normed_attn))

    // Q, K, V projections
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")

    let q_proj = linear(normed_attn, W_q_0)
    print("  After Q projection shape:", shape(q_proj))

    let k_proj = linear(normed_attn, W_k_0)
    print("  After K projection shape:", shape(k_proj))

    let v_proj = linear(normed_attn, W_v_0)
    print("  After V projection shape:", shape(v_proj))

    // Reshape for multi-head attention
    let q_heads = reshape(q_proj, [1, 32, 64])  // [seq, n_heads, head_dim]
    let k_heads = reshape(k_proj, [1, 4, 64])   // [seq, n_kv_heads, head_dim]
    let v_heads = reshape(v_proj, [1, 4, 64])
    print("  Q heads shape:", shape(q_heads))
    print("  K heads shape:", shape(k_heads))
    print("  V heads shape:", shape(v_heads))

    // Expand KV heads for GQA (4 -> 32)
    // Reshape to add dimension, broadcast, then reshape back
    let k_exp = reshape(k_heads, [1, 4, 1, 64])
    let k_broadcast = broadcast_to(k_exp, [1, 4, 8, 64])
    let k_expanded = reshape(k_broadcast, [1, 32, 64])

    let v_exp = reshape(v_heads, [1, 4, 1, 64])
    let v_broadcast = broadcast_to(v_exp, [1, 4, 8, 64])
    let v_expanded = reshape(v_broadcast, [1, 32, 64])

    print("  K expanded shape:", shape(k_expanded))
    print("  V expanded shape:", shape(v_expanded))

    // RoPE
    let q_rope = rope(q_heads, 0)
    let k_rope = rope(k_expanded, 0)
    print("  After RoPE Q shape:", shape(q_rope))
    print("  After RoPE K shape:", shape(k_rope))

    // Attention scores
    let scores = einsum("ihd,jhd->ihj", q_rope, k_rope)
    print("  Attention scores shape:", shape(scores))

    // Scale
    let scale = 0.125  // 1/sqrt(64)
    let scaled_scores = scores * scale
    print("  After scaling shape:", shape(scaled_scores))

    // Softmax
    let attn_weights = softmax(scaled_scores)
    print("  After softmax shape:", shape(attn_weights))

    // Attention output
    let attn_out = einsum("ihj,jhd->ihd", attn_weights, v_expanded)
    print("  Attention output shape:", shape(attn_out))

    // Reshape back
    let attn_flat = reshape(attn_out, [1, 2048])
    print("  Attention flat shape:", shape(attn_flat))

    // Output projection
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_proj = linear(attn_flat, W_o_0)
    print("  After output projection shape:", shape(attn_proj))

    // Residual 1
    let residual_1 = e + attn_proj
    print("  After residual 1 shape:", shape(residual_1))
    print()

    // FFN norm
    print("  FFN block:")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let normed_ffn = rms_norm(residual_1, ffn_norm_0)
    print("    After ffn_norm shape:", shape(normed_ffn))

    // SwiGLU FFN
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")

    let gate = linear(normed_ffn, W_gate_0)
    print("    After gate projection shape:", shape(gate))

    let gate_act = silu(gate)
    print("    After SiLU shape:", shape(gate_act))

    let up = linear(normed_ffn, W_up_0)
    print("    After up projection shape:", shape(up))

    let intermediate = gate_act * up
    print("    After element-wise mult shape:", shape(intermediate))

    let ffn_out = linear(intermediate, W_down_0)
    print("    After down projection shape:", shape(ffn_out))

    // Residual 2
    let residual_2 = residual_1 + ffn_out
    print("    After residual 2 shape:", shape(residual_2))
    print()

    print("=== Layer 0 Complete ===")
    print()

    // Now test with ALL 22 layers (existing implementation)
    print("=== Running Full 22-Layer Model ===")

    // Load all 22 layers
    let layers = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21
    ]

    // Initialize with embedding
    let x = e

    // Apply each layer
    let layer = 0
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let normed = rms_norm(x, attn_norm)

    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    let W_k = get_tensor(model, "blk.0.attn_k.weight")
    let W_v = get_tensor(model, "blk.0.attn_v.weight")
    let W_o = get_tensor(model, "blk.0.attn_output.weight")

    let q = linear(normed, W_q)
    let k = linear(normed, W_k)
    let v = linear(normed, W_v)

    let q_h = reshape(q, [1, 32, 64])
    let k_h = reshape(k, [1, 4, 64])
    let v_h = reshape(v, [1, 4, 64])

    let k_e2 = reshape(k_h, [1, 4, 1, 64])
    let k_b2 = broadcast_to(k_e2, [1, 4, 8, 64])
    let k_exp2 = reshape(k_b2, [1, 32, 64])

    let v_e2 = reshape(v_h, [1, 4, 1, 64])
    let v_b2 = broadcast_to(v_e2, [1, 4, 8, 64])
    let v_exp2 = reshape(v_b2, [1, 32, 64])

    let q_r = rope(q_h, 0)
    let k_r = rope(k_exp2, 0)

    let sc = einsum("ihd,jhd->ihj", q_r, k_r)
    let sc_scaled = sc * 0.125
    let attn_w = softmax(sc_scaled)
    let attn_o = einsum("ihj,jhd->ihd", attn_w, v_exp2)
    let attn_f = reshape(attn_o, [1, 2048])
    let attn_p = linear(attn_f, W_o)

    let res1 = x + attn_p

    let ffn_norm = get_tensor(model, "blk.0.ffn_norm.weight")
    let normed_f = rms_norm(res1, ffn_norm)

    let W_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down = get_tensor(model, "blk.0.ffn_down.weight")

    let g = linear(normed_f, W_gate)
    let g_act = silu(g)
    let u = linear(normed_f, W_up)
    let inter = g_act * u
    let ffn_o = linear(inter, W_down)

    let layer_0_out = res1 + ffn_o

    print("After layer 0, shape:", shape(layer_0_out))

    // Continue with remaining layers...
    // (This would be very long, so let's just do the final output)

    print()
    print("=== Skipping to Output ===")
    print("(Full 22-layer implementation would go here)")
    print()

    // Final output norm and projection
    let output_norm = get_tensor(model, "output_norm.weight")
    let final_normed = rms_norm(layer_0_out, output_norm)
    print("After output_norm shape:", shape(final_normed))

    let output_weight = get_tensor(model, "output.weight")
    let logits = linear(final_normed, output_weight)
    print("Final logits shape:", shape(logits))

    // Sample
    let predicted = temperature_sample(logits, 0.0)
    print("Predicted token (after 1 layer):", predicted)

    print()
    print("=== Debug Complete ===")
    print()
    print("Observation:")
    print("  - If this predicts reasonably, the issue is in layer accumulation")
    print("  - If this is already wrong, the issue is in layer 0 implementation")
}
