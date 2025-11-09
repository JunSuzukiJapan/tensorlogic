// Debug: Record outputs at each layer to identify where divergence occurs
//
// This script processes "Hello" through layers 0, 5, 10, 15, 20, 21
// and prints logits statistics at each checkpoint.

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let intermediate = silu(gate) * up
    linear(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q = linear(x, W_q)
    let K = linear(x, W_k)
    let V = linear(x, W_v)

    let Q_heads = reshape(Q, [shape(Q)[0], 32.0, 64.0])
    let K_heads = reshape(K, [shape(K)[0], 4.0, 64.0])
    let V_heads = reshape(V, [shape(V)[0], 4.0, 64.0])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_expanded = broadcast_to(
        reshape(K_rope, [shape(K_rope)[0], 4.0, 1.0, 64.0]),
        [shape(K_rope)[0], 4.0, 8.0, 64.0]
    )
    let K_final = reshape(K_expanded, [shape(K_rope)[0], 32.0, 64.0])

    let V_expanded = broadcast_to(
        reshape(V_heads, [shape(V_heads)[0], 4.0, 1.0, 64.0]),
        [shape(V_heads)[0], 4.0, 8.0, 64.0]
    )
    let V_final = reshape(V_expanded, [shape(V_heads)[0], 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_final)
    let scaled = scores * 0.125
    let attn_weights = softmax(scaled, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_final)

    let attn_reshaped = reshape(attn_output, [shape(attn_output)[0], 2048.0])
    linear(attn_reshaped, W_o)
}

fn transformer_layer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?]
) -> float16[?, ?] {
    let x_norm1 = rms_norm(x, attn_norm)
    let attn_out = tinyllama_gqa_attention(x_norm1, W_q, W_k, W_v, W_o)
    let h1 = x + attn_out
    let h1_norm = rms_norm(h1, ffn_norm)
    let ffn_out = swiglu_ffn(h1_norm, W_gate, W_up, W_down)
    h1 + ffn_out
}

main {
    print("=== Layer-by-Layer Output Debug ===\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    // Tokenize "Hello" (single token)
    let prompt = "Hello"
    let tokens = tokenize(tokenizer, prompt, false)
    print("Prompt: \"", prompt, "\"")
    print("Tokens: ", tokens, "\n")

    // Load all weights
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Embedding
    let h = embedding(embed_table, tokens)
    print("[After Embedding]")
    print("  Shape: ", shape(h))
    print("")

    // Process through all 22 layers with checkpoints at 0, 5, 10, 15, 20, 21

    // Layer 0
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let h0 = transformer_layer(h, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0,
                               W_gate_0, W_up_0, W_down_0, ffn_norm_0)

    print("[Checkpoint: Layer 0]")
    let h0_norm = rms_norm(h0, output_norm)
    let logits_0 = linear(h0_norm, output_weight)
    let token_0 = temperature_sample(logits_0, 0.0)
    print("  Token: ", token_0)
    print("")

    // Layers 1-4
    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    let h1 = transformer_layer(h0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1,
                               W_gate_1, W_up_1, W_down_1, ffn_norm_1)

    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")
    let h2 = transformer_layer(h1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2,
                               W_gate_2, W_up_2, W_down_2, ffn_norm_2)

    let W_q_3 = get_tensor(model, "blk.3.attn_q.weight")
    let W_k_3 = get_tensor(model, "blk.3.attn_k.weight")
    let W_v_3 = get_tensor(model, "blk.3.attn_v.weight")
    let W_o_3 = get_tensor(model, "blk.3.attn_output.weight")
    let attn_norm_3 = get_tensor(model, "blk.3.attn_norm.weight")
    let W_gate_3 = get_tensor(model, "blk.3.ffn_gate.weight")
    let W_up_3 = get_tensor(model, "blk.3.ffn_up.weight")
    let W_down_3 = get_tensor(model, "blk.3.ffn_down.weight")
    let ffn_norm_3 = get_tensor(model, "blk.3.ffn_norm.weight")
    let h3 = transformer_layer(h2, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3,
                               W_gate_3, W_up_3, W_down_3, ffn_norm_3)

    let W_q_4 = get_tensor(model, "blk.4.attn_q.weight")
    let W_k_4 = get_tensor(model, "blk.4.attn_k.weight")
    let W_v_4 = get_tensor(model, "blk.4.attn_v.weight")
    let W_o_4 = get_tensor(model, "blk.4.attn_output.weight")
    let attn_norm_4 = get_tensor(model, "blk.4.attn_norm.weight")
    let W_gate_4 = get_tensor(model, "blk.4.ffn_gate.weight")
    let W_up_4 = get_tensor(model, "blk.4.ffn_up.weight")
    let W_down_4 = get_tensor(model, "blk.4.ffn_down.weight")
    let ffn_norm_4 = get_tensor(model, "blk.4.ffn_norm.weight")
    let h4 = transformer_layer(h3, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4,
                               W_gate_4, W_up_4, W_down_4, ffn_norm_4)

    // Layer 5 checkpoint
    let W_q_5 = get_tensor(model, "blk.5.attn_q.weight")
    let W_k_5 = get_tensor(model, "blk.5.attn_k.weight")
    let W_v_5 = get_tensor(model, "blk.5.attn_v.weight")
    let W_o_5 = get_tensor(model, "blk.5.attn_output.weight")
    let attn_norm_5 = get_tensor(model, "blk.5.attn_norm.weight")
    let W_gate_5 = get_tensor(model, "blk.5.ffn_gate.weight")
    let W_up_5 = get_tensor(model, "blk.5.ffn_up.weight")
    let W_down_5 = get_tensor(model, "blk.5.ffn_down.weight")
    let ffn_norm_5 = get_tensor(model, "blk.5.ffn_norm.weight")
    let h5 = transformer_layer(h4, W_q_5, W_k_5, W_v_5, W_o_5, attn_norm_5,
                               W_gate_5, W_up_5, W_down_5, ffn_norm_5)

    print("[Checkpoint: Layer 5]")
    let h5_norm = rms_norm(h5, output_norm)
    let logits_5 = linear(h5_norm, output_weight)
    let token_5 = temperature_sample(logits_5, 0.0)
    print("  Token: ", token_5)
    print("")

    // Layers 6-9
    let W_q_6 = get_tensor(model, "blk.6.attn_q.weight")
    let W_k_6 = get_tensor(model, "blk.6.attn_k.weight")
    let W_v_6 = get_tensor(model, "blk.6.attn_v.weight")
    let W_o_6 = get_tensor(model, "blk.6.attn_output.weight")
    let attn_norm_6 = get_tensor(model, "blk.6.attn_norm.weight")
    let W_gate_6 = get_tensor(model, "blk.6.ffn_gate.weight")
    let W_up_6 = get_tensor(model, "blk.6.ffn_up.weight")
    let W_down_6 = get_tensor(model, "blk.6.ffn_down.weight")
    let ffn_norm_6 = get_tensor(model, "blk.6.ffn_norm.weight")
    let h6 = transformer_layer(h5, W_q_6, W_k_6, W_v_6, W_o_6, attn_norm_6,
                               W_gate_6, W_up_6, W_down_6, ffn_norm_6)

    let W_q_7 = get_tensor(model, "blk.7.attn_q.weight")
    let W_k_7 = get_tensor(model, "blk.7.attn_k.weight")
    let W_v_7 = get_tensor(model, "blk.7.attn_v.weight")
    let W_o_7 = get_tensor(model, "blk.7.attn_output.weight")
    let attn_norm_7 = get_tensor(model, "blk.7.attn_norm.weight")
    let W_gate_7 = get_tensor(model, "blk.7.ffn_gate.weight")
    let W_up_7 = get_tensor(model, "blk.7.ffn_up.weight")
    let W_down_7 = get_tensor(model, "blk.7.ffn_down.weight")
    let ffn_norm_7 = get_tensor(model, "blk.7.ffn_norm.weight")
    let h7 = transformer_layer(h6, W_q_7, W_k_7, W_v_7, W_o_7, attn_norm_7,
                               W_gate_7, W_up_7, W_down_7, ffn_norm_7)

    let W_q_8 = get_tensor(model, "blk.8.attn_q.weight")
    let W_k_8 = get_tensor(model, "blk.8.attn_k.weight")
    let W_v_8 = get_tensor(model, "blk.8.attn_v.weight")
    let W_o_8 = get_tensor(model, "blk.8.attn_output.weight")
    let attn_norm_8 = get_tensor(model, "blk.8.attn_norm.weight")
    let W_gate_8 = get_tensor(model, "blk.8.ffn_gate.weight")
    let W_up_8 = get_tensor(model, "blk.8.ffn_up.weight")
    let W_down_8 = get_tensor(model, "blk.8.ffn_down.weight")
    let ffn_norm_8 = get_tensor(model, "blk.8.ffn_norm.weight")
    let h8 = transformer_layer(h7, W_q_8, W_k_8, W_v_8, W_o_8, attn_norm_8,
                               W_gate_8, W_up_8, W_down_8, ffn_norm_8)

    let W_q_9 = get_tensor(model, "blk.9.attn_q.weight")
    let W_k_9 = get_tensor(model, "blk.9.attn_k.weight")
    let W_v_9 = get_tensor(model, "blk.9.attn_v.weight")
    let W_o_9 = get_tensor(model, "blk.9.attn_output.weight")
    let attn_norm_9 = get_tensor(model, "blk.9.attn_norm.weight")
    let W_gate_9 = get_tensor(model, "blk.9.ffn_gate.weight")
    let W_up_9 = get_tensor(model, "blk.9.ffn_up.weight")
    let W_down_9 = get_tensor(model, "blk.9.ffn_down.weight")
    let ffn_norm_9 = get_tensor(model, "blk.9.ffn_norm.weight")
    let h9 = transformer_layer(h8, W_q_9, W_k_9, W_v_9, W_o_9, attn_norm_9,
                               W_gate_9, W_up_9, W_down_9, ffn_norm_9)

    // Layer 10 checkpoint
    let W_q_10 = get_tensor(model, "blk.10.attn_q.weight")
    let W_k_10 = get_tensor(model, "blk.10.attn_k.weight")
    let W_v_10 = get_tensor(model, "blk.10.attn_v.weight")
    let W_o_10 = get_tensor(model, "blk.10.attn_output.weight")
    let attn_norm_10 = get_tensor(model, "blk.10.attn_norm.weight")
    let W_gate_10 = get_tensor(model, "blk.10.ffn_gate.weight")
    let W_up_10 = get_tensor(model, "blk.10.ffn_up.weight")
    let W_down_10 = get_tensor(model, "blk.10.ffn_down.weight")
    let ffn_norm_10 = get_tensor(model, "blk.10.ffn_norm.weight")
    let h10 = transformer_layer(h9, W_q_10, W_k_10, W_v_10, W_o_10, attn_norm_10,
                                W_gate_10, W_up_10, W_down_10, ffn_norm_10)

    print("[Checkpoint: Layer 10]")
    let h10_norm = rms_norm(h10, output_norm)
    let logits_10 = linear(h10_norm, output_weight)
    let token_10 = temperature_sample(logits_10, 0.0)
    print("  Token: ", token_10)
    print("")

    // Layers 11-14
    let W_q_11 = get_tensor(model, "blk.11.attn_q.weight")
    let W_k_11 = get_tensor(model, "blk.11.attn_k.weight")
    let W_v_11 = get_tensor(model, "blk.11.attn_v.weight")
    let W_o_11 = get_tensor(model, "blk.11.attn_output.weight")
    let attn_norm_11 = get_tensor(model, "blk.11.attn_norm.weight")
    let W_gate_11 = get_tensor(model, "blk.11.ffn_gate.weight")
    let W_up_11 = get_tensor(model, "blk.11.ffn_up.weight")
    let W_down_11 = get_tensor(model, "blk.11.ffn_down.weight")
    let ffn_norm_11 = get_tensor(model, "blk.11.ffn_norm.weight")
    let h11 = transformer_layer(h10, W_q_11, W_k_11, W_v_11, W_o_11, attn_norm_11,
                                W_gate_11, W_up_11, W_down_11, ffn_norm_11)

    let W_q_12 = get_tensor(model, "blk.12.attn_q.weight")
    let W_k_12 = get_tensor(model, "blk.12.attn_k.weight")
    let W_v_12 = get_tensor(model, "blk.12.attn_v.weight")
    let W_o_12 = get_tensor(model, "blk.12.attn_output.weight")
    let attn_norm_12 = get_tensor(model, "blk.12.attn_norm.weight")
    let W_gate_12 = get_tensor(model, "blk.12.ffn_gate.weight")
    let W_up_12 = get_tensor(model, "blk.12.ffn_up.weight")
    let W_down_12 = get_tensor(model, "blk.12.ffn_down.weight")
    let ffn_norm_12 = get_tensor(model, "blk.12.ffn_norm.weight")
    let h12 = transformer_layer(h11, W_q_12, W_k_12, W_v_12, W_o_12, attn_norm_12,
                                W_gate_12, W_up_12, W_down_12, ffn_norm_12)

    let W_q_13 = get_tensor(model, "blk.13.attn_q.weight")
    let W_k_13 = get_tensor(model, "blk.13.attn_k.weight")
    let W_v_13 = get_tensor(model, "blk.13.attn_v.weight")
    let W_o_13 = get_tensor(model, "blk.13.attn_output.weight")
    let attn_norm_13 = get_tensor(model, "blk.13.attn_norm.weight")
    let W_gate_13 = get_tensor(model, "blk.13.ffn_gate.weight")
    let W_up_13 = get_tensor(model, "blk.13.ffn_up.weight")
    let W_down_13 = get_tensor(model, "blk.13.ffn_down.weight")
    let ffn_norm_13 = get_tensor(model, "blk.13.ffn_norm.weight")
    let h13 = transformer_layer(h12, W_q_13, W_k_13, W_v_13, W_o_13, attn_norm_13,
                                W_gate_13, W_up_13, W_down_13, ffn_norm_13)

    let W_q_14 = get_tensor(model, "blk.14.attn_q.weight")
    let W_k_14 = get_tensor(model, "blk.14.attn_k.weight")
    let W_v_14 = get_tensor(model, "blk.14.attn_v.weight")
    let W_o_14 = get_tensor(model, "blk.14.attn_output.weight")
    let attn_norm_14 = get_tensor(model, "blk.14.attn_norm.weight")
    let W_gate_14 = get_tensor(model, "blk.14.ffn_gate.weight")
    let W_up_14 = get_tensor(model, "blk.14.ffn_up.weight")
    let W_down_14 = get_tensor(model, "blk.14.ffn_down.weight")
    let ffn_norm_14 = get_tensor(model, "blk.14.ffn_norm.weight")
    let h14 = transformer_layer(h13, W_q_14, W_k_14, W_v_14, W_o_14, attn_norm_14,
                                W_gate_14, W_up_14, W_down_14, ffn_norm_14)

    // Layer 15 checkpoint
    let W_q_15 = get_tensor(model, "blk.15.attn_q.weight")
    let W_k_15 = get_tensor(model, "blk.15.attn_k.weight")
    let W_v_15 = get_tensor(model, "blk.15.attn_v.weight")
    let W_o_15 = get_tensor(model, "blk.15.attn_output.weight")
    let attn_norm_15 = get_tensor(model, "blk.15.attn_norm.weight")
    let W_gate_15 = get_tensor(model, "blk.15.ffn_gate.weight")
    let W_up_15 = get_tensor(model, "blk.15.ffn_up.weight")
    let W_down_15 = get_tensor(model, "blk.15.ffn_down.weight")
    let ffn_norm_15 = get_tensor(model, "blk.15.ffn_norm.weight")
    let h15 = transformer_layer(h14, W_q_15, W_k_15, W_v_15, W_o_15, attn_norm_15,
                                W_gate_15, W_up_15, W_down_15, ffn_norm_15)

    print("[Checkpoint: Layer 15]")
    let h15_norm = rms_norm(h15, output_norm)
    let logits_15 = linear(h15_norm, output_weight)
    let token_15 = temperature_sample(logits_15, 0.0)
    print("  Token: ", token_15)
    print("")

    // Layers 16-19
    let W_q_16 = get_tensor(model, "blk.16.attn_q.weight")
    let W_k_16 = get_tensor(model, "blk.16.attn_k.weight")
    let W_v_16 = get_tensor(model, "blk.16.attn_v.weight")
    let W_o_16 = get_tensor(model, "blk.16.attn_output.weight")
    let attn_norm_16 = get_tensor(model, "blk.16.attn_norm.weight")
    let W_gate_16 = get_tensor(model, "blk.16.ffn_gate.weight")
    let W_up_16 = get_tensor(model, "blk.16.ffn_up.weight")
    let W_down_16 = get_tensor(model, "blk.16.ffn_down.weight")
    let ffn_norm_16 = get_tensor(model, "blk.16.ffn_norm.weight")
    let h16 = transformer_layer(h15, W_q_16, W_k_16, W_v_16, W_o_16, attn_norm_16,
                                W_gate_16, W_up_16, W_down_16, ffn_norm_16)

    let W_q_17 = get_tensor(model, "blk.17.attn_q.weight")
    let W_k_17 = get_tensor(model, "blk.17.attn_k.weight")
    let W_v_17 = get_tensor(model, "blk.17.attn_v.weight")
    let W_o_17 = get_tensor(model, "blk.17.attn_output.weight")
    let attn_norm_17 = get_tensor(model, "blk.17.attn_norm.weight")
    let W_gate_17 = get_tensor(model, "blk.17.ffn_gate.weight")
    let W_up_17 = get_tensor(model, "blk.17.ffn_up.weight")
    let W_down_17 = get_tensor(model, "blk.17.ffn_down.weight")
    let ffn_norm_17 = get_tensor(model, "blk.17.ffn_norm.weight")
    let h17 = transformer_layer(h16, W_q_17, W_k_17, W_v_17, W_o_17, attn_norm_17,
                                W_gate_17, W_up_17, W_down_17, ffn_norm_17)

    let W_q_18 = get_tensor(model, "blk.18.attn_q.weight")
    let W_k_18 = get_tensor(model, "blk.18.attn_k.weight")
    let W_v_18 = get_tensor(model, "blk.18.attn_v.weight")
    let W_o_18 = get_tensor(model, "blk.18.attn_output.weight")
    let attn_norm_18 = get_tensor(model, "blk.18.attn_norm.weight")
    let W_gate_18 = get_tensor(model, "blk.18.ffn_gate.weight")
    let W_up_18 = get_tensor(model, "blk.18.ffn_up.weight")
    let W_down_18 = get_tensor(model, "blk.18.ffn_down.weight")
    let ffn_norm_18 = get_tensor(model, "blk.18.ffn_norm.weight")
    let h18 = transformer_layer(h17, W_q_18, W_k_18, W_v_18, W_o_18, attn_norm_18,
                                W_gate_18, W_up_18, W_down_18, ffn_norm_18)

    let W_q_19 = get_tensor(model, "blk.19.attn_q.weight")
    let W_k_19 = get_tensor(model, "blk.19.attn_k.weight")
    let W_v_19 = get_tensor(model, "blk.19.attn_v.weight")
    let W_o_19 = get_tensor(model, "blk.19.attn_output.weight")
    let attn_norm_19 = get_tensor(model, "blk.19.attn_norm.weight")
    let W_gate_19 = get_tensor(model, "blk.19.ffn_gate.weight")
    let W_up_19 = get_tensor(model, "blk.19.ffn_up.weight")
    let W_down_19 = get_tensor(model, "blk.19.ffn_down.weight")
    let ffn_norm_19 = get_tensor(model, "blk.19.ffn_norm.weight")
    let h19 = transformer_layer(h18, W_q_19, W_k_19, W_v_19, W_o_19, attn_norm_19,
                                W_gate_19, W_up_19, W_down_19, ffn_norm_19)

    // Layer 20 checkpoint
    let W_q_20 = get_tensor(model, "blk.20.attn_q.weight")
    let W_k_20 = get_tensor(model, "blk.20.attn_k.weight")
    let W_v_20 = get_tensor(model, "blk.20.attn_v.weight")
    let W_o_20 = get_tensor(model, "blk.20.attn_output.weight")
    let attn_norm_20 = get_tensor(model, "blk.20.attn_norm.weight")
    let W_gate_20 = get_tensor(model, "blk.20.ffn_gate.weight")
    let W_up_20 = get_tensor(model, "blk.20.ffn_up.weight")
    let W_down_20 = get_tensor(model, "blk.20.ffn_down.weight")
    let ffn_norm_20 = get_tensor(model, "blk.20.ffn_norm.weight")
    let h20 = transformer_layer(h19, W_q_20, W_k_20, W_v_20, W_o_20, attn_norm_20,
                                W_gate_20, W_up_20, W_down_20, ffn_norm_20)

    print("[Checkpoint: Layer 20]")
    let h20_norm = rms_norm(h20, output_norm)
    let logits_20 = linear(h20_norm, output_weight)
    let token_20 = temperature_sample(logits_20, 0.0)
    print("  Token: ", token_20)
    print("")

    // Layer 21 (final layer)
    let W_q_21 = get_tensor(model, "blk.21.attn_q.weight")
    let W_k_21 = get_tensor(model, "blk.21.attn_k.weight")
    let W_v_21 = get_tensor(model, "blk.21.attn_v.weight")
    let W_o_21 = get_tensor(model, "blk.21.attn_output.weight")
    let attn_norm_21 = get_tensor(model, "blk.21.attn_norm.weight")
    let W_gate_21 = get_tensor(model, "blk.21.ffn_gate.weight")
    let W_up_21 = get_tensor(model, "blk.21.ffn_up.weight")
    let W_down_21 = get_tensor(model, "blk.21.ffn_down.weight")
    let ffn_norm_21 = get_tensor(model, "blk.21.ffn_norm.weight")
    let h21 = transformer_layer(h20, W_q_21, W_k_21, W_v_21, W_o_21, attn_norm_21,
                                W_gate_21, W_up_21, W_down_21, ffn_norm_21)

    print("[Checkpoint: Layer 21 (Final)]")
    let h21_norm = rms_norm(h21, output_norm)
    let logits_21 = linear(h21_norm, output_weight)
    let token_21 = temperature_sample(logits_21, 0.0)
    print("  Token: ", token_21)
    print("")

    print("✅ Full 22-layer checkpoint analysis complete")
    print("\n📝 Summary:")
    print("  Layer 0:  Token ", token_0)
    print("  Layer 5:  Token ", token_5)
    print("  Layer 10: Token ", token_10)
    print("  Layer 15: Token ", token_15)
    print("  Layer 20: Token ", token_20)
    print("  Layer 21: Token ", token_21)
    print("\n🔍 Next: Compare with llama.cpp layer-by-layer outputs")
}
