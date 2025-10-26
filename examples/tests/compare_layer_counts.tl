// Compare predictions with different layer counts
// Find at which point the model starts producing abnormal logits

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

fn transformer_layer(
    x: float16[?, ?],
    layer_idx: int,
    model: Model
) -> float16[?, ?] {
    // Get layer weights
    let attn_norm = get_tensor(model, "blk." + str(layer_idx) + ".attn_norm.weight")
    let W_q = get_tensor(model, "blk." + str(layer_idx) + ".attn_q.weight")
    let W_k = get_tensor(model, "blk." + str(layer_idx) + ".attn_k.weight")
    let W_v = get_tensor(model, "blk." + str(layer_idx) + ".attn_v.weight")
    let W_o = get_tensor(model, "blk." + str(layer_idx) + ".attn_output.weight")
    let ffn_norm = get_tensor(model, "blk." + str(layer_idx) + ".ffn_norm.weight")
    let W_gate = get_tensor(model, "blk." + str(layer_idx) + ".ffn_gate.weight")
    let W_up = get_tensor(model, "blk." + str(layer_idx) + ".ffn_up.weight")
    let W_down = get_tensor(model, "blk." + str(layer_idx) + ".ffn_down.weight")

    // Attention block
    let normed = rms_norm(x, attn_norm)

    let q = linear(normed, W_q)
    let k = linear(normed, W_k)
    let v = linear(normed, W_v)

    let q_h = reshape(q, [1, 32, 64])
    let k_h = reshape(k, [1, 4, 64])
    let v_h = reshape(v, [1, 4, 64])

    // Expand KV heads (4 -> 32)
    let k_e = reshape(k_h, [1, 4, 1, 64])
    let k_b = broadcast_to(k_e, [1, 4, 8, 64])
    let k_exp = reshape(k_b, [1, 32, 64])

    let v_e = reshape(v_h, [1, 4, 1, 64])
    let v_b = broadcast_to(v_e, [1, 4, 8, 64])
    let v_exp = reshape(v_b, [1, 32, 64])

    // RoPE
    let q_r = rope(q_h, 0)
    let k_r = rope(k_exp, 0)

    // Attention
    let scores = einsum("ihd,jhd->ihj", q_r, k_r)
    let scaled = scores * 0.125  // 1/sqrt(64)
    let attn_w = softmax(scaled)
    let attn_o = einsum("ihj,jhd->ihd", attn_w, v_exp)

    let attn_flat = reshape(attn_o, [1, 2048])
    let attn_proj = linear(attn_flat, W_o)

    // Residual 1
    let res1 = x + attn_proj

    // FFN block
    let normed_ffn = rms_norm(res1, ffn_norm)

    let gate = linear(normed_ffn, W_gate)
    let gate_act = silu(gate)
    let up = linear(normed_ffn, W_up)
    let intermediate = gate_act * up
    let ffn_out = linear(intermediate, W_down)

    // Residual 2
    result := res1 + ffn_out
}

main {
    print("=== Compare Layer Counts ===\n")

    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Input: BOS token
    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input: BOS token\n")

    // Embedding
    let embed_table = get_tensor(model, "token_embd.weight")
    let e = embedding(embed_table, tokens)

    // Output weights (shared for all tests)
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Test 1: 1 layer
    print("=== Test 1: 1 Layer ===")
    let x1 = transformer_layer(e, 0, model)
    let normed1 = rms_norm(x1, output_norm)
    let logits1 = linear(normed1, output_weight)
    let pred1 = temperature_sample(logits1, 0.0)
    print("  Predicted token:", pred1)
    print()

    // Test 2: 2 layers
    print("=== Test 2: 2 Layers ===")
    let x2_0 = transformer_layer(e, 0, model)
    let x2_1 = transformer_layer(x2_0, 1, model)
    let normed2 = rms_norm(x2_1, output_norm)
    let logits2 = linear(normed2, output_weight)
    let pred2 = temperature_sample(logits2, 0.0)
    print("  Predicted token:", pred2)
    print()

    // Test 3: 5 layers
    print("=== Test 3: 5 Layers ===")
    let x5_0 = transformer_layer(e, 0, model)
    let x5_1 = transformer_layer(x5_0, 1, model)
    let x5_2 = transformer_layer(x5_1, 2, model)
    let x5_3 = transformer_layer(x5_2, 3, model)
    let x5_4 = transformer_layer(x5_3, 4, model)
    let normed5 = rms_norm(x5_4, output_norm)
    let logits5 = linear(normed5, output_weight)
    let pred5 = temperature_sample(logits5, 0.0)
    print("  Predicted token:", pred5)
    print()

    // Test 4: 10 layers
    print("=== Test 4: 10 Layers ===")
    let x10_0 = transformer_layer(e, 0, model)
    let x10_1 = transformer_layer(x10_0, 1, model)
    let x10_2 = transformer_layer(x10_1, 2, model)
    let x10_3 = transformer_layer(x10_2, 3, model)
    let x10_4 = transformer_layer(x10_3, 4, model)
    let x10_5 = transformer_layer(x10_4, 5, model)
    let x10_6 = transformer_layer(x10_5, 6, model)
    let x10_7 = transformer_layer(x10_6, 7, model)
    let x10_8 = transformer_layer(x10_7, 8, model)
    let x10_9 = transformer_layer(x10_8, 9, model)
    let normed10 = rms_norm(x10_9, output_norm)
    let logits10 = linear(normed10, output_weight)
    let pred10 = temperature_sample(logits10, 0.0)
    print("  Predicted token:", pred10)
    print()

    print("=== Summary ===")
    print("1 layer:  token", pred1)
    print("2 layers: token", pred2)
    print("5 layers: token", pred5)
    print("10 layers: token", pred10)
    print()
    print("Observation:")
    print("  - If predictions are similar, layers are working correctly")
    print("  - If predictions diverge, cumulative error is the problem")
    print("  - Check which layer count starts producing different tokens")
}
