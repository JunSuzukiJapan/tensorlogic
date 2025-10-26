// Check magnitude drift with REAL TinyLlama weights
// This will tell us if the issue is our test weights or a real problem

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

fn print_stats(t: float16[?, ?], name: string) {
    let s = shape(t)
    print("  ", name, " shape:", s)
    // Can't directly inspect values in TensorLogic, but shape is informative
    result := 0
}

main {
    print("=== Real Model Weights Magnitude Check ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // BOS token
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input: BOS token\n")

    // Embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let x0 = embedding(embed_table, tokens)
    print("Layer -1 (embedding):")
    print_stats(x0, "  embedding")
    print()

    // Layer 0
    print("=== Layer 0 ===")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let normed0 = rms_norm(x0, attn_norm_0)
    print_stats(normed0, "  after_attn_norm")

    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")

    let q0 = linear(normed0, W_q_0)
    let k0 = linear(normed0, W_k_0)
    let v0 = linear(normed0, W_v_0)

    let q_h = reshape(q0, [1, 32, 64])
    let k_h = reshape(k0, [1, 4, 64])
    let v_h = reshape(v0, [1, 4, 64])

    let k_e = reshape(k_h, [1, 4, 1, 64])
    let k_b = broadcast_to(k_e, [1, 4, 8, 64])
    let k_exp = reshape(k_b, [1, 32, 64])

    let v_e = reshape(v_h, [1, 4, 1, 64])
    let v_b = broadcast_to(v_e, [1, 4, 8, 64])
    let v_exp = reshape(v_b, [1, 32, 64])

    let q_r = rope(q_h, 0)
    let k_r = rope(k_exp, 0)

    let scores = einsum("ihd,jhd->ihj", q_r, k_r)
    let scaled = scores * 0.125
    let attn_w = softmax(scaled)
    let attn_o = einsum("ihj,jhd->ihd", attn_w, v_exp)
    let attn_flat = reshape(attn_o, [1, 2048])
    let attn_proj = linear(attn_flat, W_o_0)

    print_stats(attn_proj, "  attn_output")

    let res1 = x0 + attn_proj
    print_stats(res1, "  after_residual_1")

    // FFN
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let normed_ffn = rms_norm(res1, ffn_norm_0)
    print_stats(normed_ffn, "  after_ffn_norm")

    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")

    let gate = linear(normed_ffn, W_gate_0)
    let gate_act = silu(gate)
    let up = linear(normed_ffn, W_up_0)
    let inter = gate_act * up
    let ffn_out = linear(inter, W_down_0)

    print_stats(ffn_out, "  ffn_output")

    let x1 = res1 + ffn_out
    print_stats(x1, "  layer_0_final")
    print()

    // Layer 1
    print("=== Layer 1 ===")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let normed1 = rms_norm(x1, attn_norm_1)

    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")

    let q1 = linear(normed1, W_q_1)
    let k1 = linear(normed1, W_k_1)
    let v1 = linear(normed1, W_v_1)

    let q1_h = reshape(q1, [1, 32, 64])
    let k1_h = reshape(k1, [1, 4, 64])
    let v1_h = reshape(v1, [1, 4, 64])

    let k1_e = reshape(k1_h, [1, 4, 1, 64])
    let k1_b = broadcast_to(k1_e, [1, 4, 8, 64])
    let k1_exp = reshape(k1_b, [1, 32, 64])

    let v1_e = reshape(v1_h, [1, 4, 1, 64])
    let v1_b = broadcast_to(v1_e, [1, 4, 8, 64])
    let v1_exp = reshape(v1_b, [1, 32, 64])

    let q1_r = rope(q1_h, 0)
    let k1_r = rope(k1_exp, 0)

    let scores1 = einsum("ihd,jhd->ihj", q1_r, k1_r)
    let scaled1 = scores1 * 0.125
    let attn1_w = softmax(scaled1)
    let attn1_o = einsum("ihj,jhd->ihd", attn1_w, v1_exp)
    let attn1_flat = reshape(attn1_o, [1, 2048])
    let attn1_proj = linear(attn1_flat, W_o_1)

    let res1_1 = x1 + attn1_proj

    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    let normed1_ffn = rms_norm(res1_1, ffn_norm_1)

    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")

    let gate1 = linear(normed1_ffn, W_gate_1)
    let gate1_act = silu(gate1)
    let up1 = linear(normed1_ffn, W_up_1)
    let inter1 = gate1_act * up1
    let ffn1_out = linear(inter1, W_down_1)

    let x2 = res1_1 + ffn1_out
    print_stats(x2, "  layer_1_final")
    print()

    // Output
    print("=== Final Output ===")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let final_normed = rms_norm(x2, output_norm)
    let logits = linear(final_normed, output_weight)
    print_stats(logits, "  logits")

    let pred = temperature_sample(logits, 0.0)
    print("\nPredicted token (2 layers):", pred)
    print()

    print("=== Next: Check RMSNorm Implementation ===")
    print("We need to verify RMSNorm matches llama.cpp exactly:")
    print("  1. Compute RMS: sqrt(mean(x^2) + eps)")
    print("  2. Normalize: x / RMS")
    print("  3. Scale: normalized * weight")
    print()
    print("Expected RMSNorm eps: 1e-5 (standard for LLaMA)")
}
