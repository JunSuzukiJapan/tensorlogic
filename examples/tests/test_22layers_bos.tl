// Test 22 layers with BOS token only - compare with 2-layer result

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
}

fn swiglu_ffn(x: float16[?, ?], W_gate: float16[?, ?], W_up: float16[?, ?], W_down: float16[?, ?]) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    result = linear(intermediate, W_down)
}

fn transformer_layer(h: float16[?, ?], layer_idx: int) -> float16[?, ?] {
    let model_var = "blk." + string(layer_idx) + "."

    let attn_norm_weight = get_tensor(global_model, model_var + "attn_norm.weight")
    let W_q = get_tensor(global_model, model_var + "attn_q.weight")
    let W_k = get_tensor(global_model, model_var + "attn_k.weight")
    let W_v = get_tensor(global_model, model_var + "attn_v.weight")
    let W_o = get_tensor(global_model, model_var + "attn_output.weight")
    let ffn_norm_weight = get_tensor(global_model, model_var + "ffn_norm.weight")
    let W_gate = get_tensor(global_model, model_var + "ffn_gate.weight")
    let W_up = get_tensor(global_model, model_var + "ffn_up.weight")
    let W_down = get_tensor(global_model, model_var + "ffn_down.weight")

    // Attention
    let normed_attn = rms_norm(h, attn_norm_weight)
    let q = linear(normed_attn, W_q)
    let k = linear(normed_attn, W_k)
    let v = linear(normed_attn, W_v)

    let q_heads = reshape(q, [1.0, 32.0, 64.0])
    let k_heads = reshape(k, [1.0, 4.0, 64.0])
    let v_heads = reshape(v, [1.0, 4.0, 64.0])

    let q_rope = rope(q_heads)
    let k_rope = rope(k_heads)

    let k_with_group = reshape(k_rope, [1.0, 4.0, 1.0, 64.0])
    let v_with_group = reshape(v_heads, [1.0, 4.0, 1.0, 64.0])

    let k_expanded = reshape(broadcast_to(k_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])
    let v_expanded = reshape(broadcast_to(v_with_group, [1.0, 4.0, 8.0, 64.0]), [1.0, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", q_rope, k_expanded)
    let scaled = scores * 0.125
    let attn_weights = softmax(scaled, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, v_expanded)
    let attn_output_flat = reshape(attn_output, [1.0, 2048.0])

    let attn_proj = linear(attn_output_flat, W_o)
    let h_after_attn = h + attn_proj

    // FFN
    let normed_ffn = rms_norm(h_after_attn, ffn_norm_weight)
    let ffn_output = swiglu_ffn(normed_ffn, W_gate, W_up, W_down)

    result = h_after_attn + ffn_output
}

let global_model: Model

main {
    print("=== 22-Layer Test with BOS Token ===\n")

    global_model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let bos_str = "<s>"
    let tokens = tokenize(tokenizer, bos_str, false)
    print("Input:", bos_str)
    print("Tokens:", tokens, "\n")

    // Embedding
    let embed_table = get_tensor(global_model, "token_embd.weight")
    let h = embedding(embed_table, tokens)
    print("Embedding shape:", shape(h))

    // All 22 layers
    print("\nProcessing 22 layers...")

    h = transformer_layer(h, 0)
    print("Layer 0 done")

    h = transformer_layer(h, 1)
    print("Layer 1 done")

    h = transformer_layer(h, 2)
    print("Layer 2 done")

    h = transformer_layer(h, 3)
    h = transformer_layer(h, 4)
    h = transformer_layer(h, 5)
    h = transformer_layer(h, 6)
    h = transformer_layer(h, 7)
    h = transformer_layer(h, 8)
    h = transformer_layer(h, 9)
    print("Layer 0-9 done")

    h = transformer_layer(h, 10)
    h = transformer_layer(h, 11)
    h = transformer_layer(h, 12)
    h = transformer_layer(h, 13)
    h = transformer_layer(h, 14)
    h = transformer_layer(h, 15)
    h = transformer_layer(h, 16)
    h = transformer_layer(h, 17)
    h = transformer_layer(h, 18)
    h = transformer_layer(h, 19)
    print("Layer 10-19 done")

    h = transformer_layer(h, 20)
    h = transformer_layer(h, 21)
    print("All 22 layers done")

    // Output
    let output_norm = get_tensor(global_model, "output_norm.weight")
    let output_weight = get_tensor(global_model, "output.weight")

    let normed_output = rms_norm(h, output_norm)
    let logits = linear(normed_output, output_weight)

    print("\nLogits shape:", shape(logits))

    let predicted = temperature_sample(logits, 0.0)
    print("\n=== RESULT ===")
    print("Predicted token (22 layers):", predicted)
    print("\n(Compare with 2-layer result: 1647)")
}
