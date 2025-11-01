// Debug 2 layers with simple value tracking
// Simplified version without array slicing

// SiLU activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
}

// SwiGLU FFN
fn swiglu_ffn(x: float16[?, ?], W_gate: float16[?, ?], W_up: float16[?, ?], W_down: float16[?, ?]) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    result = linear(intermediate, W_down)
}

main {
    print("=== 2-Layer Simple Debug ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // BOS token only
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input:", bos_str)
    print("Tokens:", tokens, "\n")

    // Embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let e = embedding(embed_table, tokens)
    print("Embedding shape:", shape(e))

    // Layer 0
    print("\n=== LAYER 0 ===")

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
    let normed_attn0 = rms_norm(e, attn_norm_0)
    print("After attn_norm:", shape(normed_attn0))

    let q0 = linear(normed_attn0, W_q_0)
    let k0 = linear(normed_attn0, W_k_0)
    let v0 = linear(normed_attn0, W_v_0)

    print("Q shape:", shape(q0))
    print("K shape:", shape(k0))
    print("V shape:", shape(v0))

    // Reshape to heads
    let q_heads = reshape(q0, [1.0, 32.0, 64.0])
    let k_heads = reshape(k0, [1.0, 4.0, 64.0])
    let v_heads = reshape(v0, [1.0, 4.0, 64.0])

    // RoPE
    let q_rope = rope(q_heads)
    let k_rope = rope(k_heads)

    // GQA: Expand K,V from 4 to 32 heads
    let k_with_group = reshape(k_rope, [1.0, 4.0, 1.0, 64.0])
    let v_with_group = reshape(v_heads, [1.0, 4.0, 1.0, 64.0])

    let k_expanded = reshape(broadcast_to(k_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])
    let v_expanded = reshape(broadcast_to(v_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])

    // Attention scores
    let scores = einsum("ihd,jhd->ihj", q_rope, k_expanded)
    let scaled = scores * 0.125
    let attn_weights = softmax(scaled, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, v_expanded)
    let attn_output_flat = reshape(attn_output, [1.0, 2048.0])

    let attn_proj = linear(attn_output_flat, W_o_0)
    print("Attention output shape:", shape(attn_proj))

    let h0_after_attn = e + attn_proj
    print("After residual shape:", shape(h0_after_attn))

    // FFN
    let normed_ffn0 = rms_norm(h0_after_attn, ffn_norm_0)
    let ffn_output0 = swiglu_ffn(normed_ffn0, W_gate_0, W_up_0, W_down_0)
    print("FFN output shape:", shape(ffn_output0))

    let h0 = h0_after_attn + ffn_output0
    print("Layer 0 final output shape:", shape(h0))

    // Layer 1
    print("\n=== LAYER 1 ===")

    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")

    // Attention
    let normed_attn1 = rms_norm(h0, attn_norm_1)
    let q1 = linear(normed_attn1, W_q_1)
    let k1 = linear(normed_attn1, W_k_1)
    let v1 = linear(normed_attn1, W_v_1)

    let q1_heads = reshape(q1, [1.0, 32.0, 64.0])
    let k1_heads = reshape(k1, [1.0, 4.0, 64.0])
    let v1_heads = reshape(v1, [1.0, 4.0, 64.0])

    let q1_rope = rope(q1_heads)
    let k1_rope = rope(k1_heads)

    let k1_with_group = reshape(k1_rope, [1.0, 4.0, 1.0, 64.0])
    let v1_with_group = reshape(v1_heads, [1.0, 4.0, 1.0, 64.0])

    let k1_expanded = reshape(broadcast_to(k1_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])
    let v1_expanded = reshape(broadcast_to(v1_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])

    let scores1 = einsum("ihd,jhd->ihj", q1_rope, k1_expanded)
    let scaled1 = scores1 * 0.125
    let attn_weights1 = softmax(scaled1, 2)
    let attn_output1 = einsum("ihj,jhd->ihd", attn_weights1, v1_expanded)
    let attn_output1_flat = reshape(attn_output1, [1.0, 2048.0])

    let attn_proj1 = linear(attn_output1_flat, W_o_1)
    let h1_after_attn = h0 + attn_proj1

    let normed_ffn1 = rms_norm(h1_after_attn, ffn_norm_1)
    let ffn_output1 = swiglu_ffn(normed_ffn1, W_gate_1, W_up_1, W_down_1)
    let h1 = h1_after_attn + ffn_output1

    print("Layer 1 final output shape:", shape(h1))

    // Output projection
    print("\n=== OUTPUT ===")

    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let normed_output = rms_norm(h1, output_norm)
    let logits = linear(normed_output, output_weight)
    print("Logits shape:", shape(logits))

    let predicted = temperature_sample(logits, 0.0)
    print("\nPredicted token:", predicted)

    print("\n=== COMPLETE ===")
}
