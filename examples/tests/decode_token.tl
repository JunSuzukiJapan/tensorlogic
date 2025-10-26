// Debug: Test single transformer layer
// Check if layer 0 produces reasonable outputs

main {
    print("=== Layer 0 Only Test ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // BOS token
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input:", bos_str)
    print("Tokens:", tokens, "\n")

    // Embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let e = embedding(embed_table, tokens)
    print("1. Initial embedding shape:", shape(e))

    // Layer 0 weights
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")

    // Attention
    print("\n2. Attention...")
    let x_norm = rms_norm(e, attn_norm_0)
    print("   After attn_norm:", shape(x_norm))
    
    let q = linear(x_norm, W_q_0)
    let k = linear(x_norm, W_k_0)
    let v = linear(x_norm, W_v_0)
    print("   Q shape:", shape(q))
    print("   K shape:", shape(k))
    print("   V shape:", shape(v))

    // Reshape to heads first
    let q_heads = reshape(q, [1.0, 32.0, 64.0])
    let k_heads = reshape(k, [1.0, 4.0, 64.0])
    let v_heads = reshape(v, [1.0, 4.0, 64.0])
    print("   Reshaped heads - Q:", shape(q_heads), "K:", shape(k_heads), "V:", shape(v_heads))

    // Apply RoPE
    let q_rope = rope(q_heads)
    let k_rope = rope(k_heads)
    print("   After RoPE - Q:", shape(q_rope), "K:", shape(k_rope))

    // Expand KV for GQA (4 -> 32 heads)
    let k_group = reshape(k_rope, [1.0, 4.0, 1.0, 64.0])
    let v_group = reshape(v_heads, [1.0, 4.0, 1.0, 64.0])
    let k_broadcast = broadcast_to(k_group, [1.0, 4.0, 8.0, 64.0])
    let v_broadcast = broadcast_to(v_group, [1.0, 4.0, 8.0, 64.0])
    let k_expanded = reshape(k_broadcast, [1.0, 32.0, 64.0])
    let v_expanded = reshape(v_broadcast, [1.0, 32.0, 64.0])
    print("   GQA expanded - K:", shape(k_expanded), "V:", shape(v_expanded))

    // Attention computation
    print("\n3. Computing attention scores...")
    let scores = einsum("ihd,jhd->ihj", q_rope, k_expanded)
    print("   Attention scores shape:", shape(scores))
    let scaled = scores * 0.125
    print("   After scaling:", shape(scaled))
    
    let attn_weights = softmax(scaled, 2)
    print("   After softmax:", shape(attn_weights))

    // Apply to values
    let attn_out = einsum("ihj,jhd->ihd", attn_weights, v_expanded)
    print("   Attention output:", shape(attn_out))

    // Output projection
    let attn_flat = reshape(attn_out, [1.0, 2048.0])
    let attn_proj = linear(attn_flat, W_o_0)
    print("   After output projection:", shape(attn_proj))

    // Residual
    let h0 = e + attn_proj
    print("   After residual (h0):", shape(h0))

    // FFN
    print("\n4. Feed-Forward Network...")
    let h0_norm = rms_norm(h0, ffn_norm_0)
    print("   After ffn_norm:", shape(h0_norm))
    
    let gate = linear(h0_norm, W_gate_0)
    let up = linear(h0_norm, W_up_0)
    let gate_sig = sigmoid(gate)
    let gate_act = gate * gate_sig
    let ffn_input = gate_act * up
    let ffn_output = linear(ffn_input, W_down_0)
    print("   FFN output:", shape(ffn_output))

    // Final residual
    let h1 = h0 + ffn_output
    print("   After final residual (h1):", shape(h1))

    // Test final projection
    print("\n5. Final projection to logits...")
    let output_norm = get_tensor(model, "output_norm.weight")
    let final_norm = rms_norm(h1, output_norm)
    print("   After output_norm:", shape(final_norm))

    let output_weight = get_tensor(model, "output.weight")
    let logits = linear(final_norm, output_weight)
    print("   Logits shape:", shape(logits))

    let predicted = temperature_sample(logits, 0.0)
    print("\n   Predicted token:", predicted)
    print("\n=== Test Complete ===")
}
