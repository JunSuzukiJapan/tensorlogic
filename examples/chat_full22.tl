// Full 22-Layer Chat Demo with f32
// Complete implementation using ALL 22 transformer layers

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    result := x * sigmoid(x)
}

fn swiglu_ffn(
    x: float32[?, ?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?]
) -> float32[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    result := linear(mul_result, W_down)
}

fn attention_with_cache(
    Q: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?],
    W_o: float32[?, ?]
) -> float32[?, ?] {
    let Q_shape = shape(Q)
    let seq_len_f = Q_shape[0]
    let K_shape = shape(K_cache)
    let cache_len_f = K_shape[0]

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let Q_rope = rope(Q_heads)

    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let K_rope = rope(K_heads)
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_exp = reshape(K_rope, [cache_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    let reshaped = reshape(out, [seq_len_f, 2048.0])
    result := linear(reshaped, W_o)
}

fn transformer_layer(
    x: float32[?, ?],
    W_attn_norm: float32[?],
    W_q: float32[?, ?],
    W_k: float32[?, ?],
    W_v: float32[?, ?],
    W_o: float32[?, ?],
    W_ffn_norm: float32[?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?]
) -> float32[?, ?] {
    let normed = rms_norm(x, W_attn_norm)
    let Q = linear(normed, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out
    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result := after_attn + ffn_out
}

main {
    print("=== TensorLogic Chat: FULL 22-Layer System ===")
    print("")

    let EOS_TOKEN = 2

    print("[1/3] Loading model (ALL 22 layers)...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("      Loading all 22 transformer layers...")


    // Layer 0
    let layer_0_attn_norm = model.blk[0].attn_norm.weight
    let layer_0_q = model.blk[0].attn_q.weight
    let layer_0_k = model.blk[0].attn_k.weight
    let layer_0_v = model.blk[0].attn_v.weight
    let layer_0_o = model.blk[0].attn_output.weight
    let layer_0_ffn_norm = model.blk[0].ffn_norm.weight
    let layer_0_gate = model.blk[0].ffn_gate.weight
    let layer_0_up = model.blk[0].ffn_up.weight
    let layer_0_down = model.blk[0].ffn_down.weight

    // Layer 1
    let layer_1_attn_norm = model.blk[1].attn_norm.weight
    let layer_1_q = model.blk[1].attn_q.weight
    let layer_1_k = model.blk[1].attn_k.weight
    let layer_1_v = model.blk[1].attn_v.weight
    let layer_1_o = model.blk[1].attn_output.weight
    let layer_1_ffn_norm = model.blk[1].ffn_norm.weight
    let layer_1_gate = model.blk[1].ffn_gate.weight
    let layer_1_up = model.blk[1].ffn_up.weight
    let layer_1_down = model.blk[1].ffn_down.weight

    // Layer 2
    let layer_2_attn_norm = model.blk[2].attn_norm.weight
    let layer_2_q = model.blk[2].attn_q.weight
    let layer_2_k = model.blk[2].attn_k.weight
    let layer_2_v = model.blk[2].attn_v.weight
    let layer_2_o = model.blk[2].attn_output.weight
    let layer_2_ffn_norm = model.blk[2].ffn_norm.weight
    let layer_2_gate = model.blk[2].ffn_gate.weight
    let layer_2_up = model.blk[2].ffn_up.weight
    let layer_2_down = model.blk[2].ffn_down.weight

    // Layer 3
    let layer_3_attn_norm = model.blk[3].attn_norm.weight
    let layer_3_q = model.blk[3].attn_q.weight
    let layer_3_k = model.blk[3].attn_k.weight
    let layer_3_v = model.blk[3].attn_v.weight
    let layer_3_o = model.blk[3].attn_output.weight
    let layer_3_ffn_norm = model.blk[3].ffn_norm.weight
    let layer_3_gate = model.blk[3].ffn_gate.weight
    let layer_3_up = model.blk[3].ffn_up.weight
    let layer_3_down = model.blk[3].ffn_down.weight

    // Layer 4
    let layer_4_attn_norm = model.blk[4].attn_norm.weight
    let layer_4_q = model.blk[4].attn_q.weight
    let layer_4_k = model.blk[4].attn_k.weight
    let layer_4_v = model.blk[4].attn_v.weight
    let layer_4_o = model.blk[4].attn_output.weight
    let layer_4_ffn_norm = model.blk[4].ffn_norm.weight
    let layer_4_gate = model.blk[4].ffn_gate.weight
    let layer_4_up = model.blk[4].ffn_up.weight
    let layer_4_down = model.blk[4].ffn_down.weight

    // Layer 5
    let layer_5_attn_norm = model.blk[5].attn_norm.weight
    let layer_5_q = model.blk[5].attn_q.weight
    let layer_5_k = model.blk[5].attn_k.weight
    let layer_5_v = model.blk[5].attn_v.weight
    let layer_5_o = model.blk[5].attn_output.weight
    let layer_5_ffn_norm = model.blk[5].ffn_norm.weight
    let layer_5_gate = model.blk[5].ffn_gate.weight
    let layer_5_up = model.blk[5].ffn_up.weight
    let layer_5_down = model.blk[5].ffn_down.weight

    // Layer 6
    let layer_6_attn_norm = model.blk[6].attn_norm.weight
    let layer_6_q = model.blk[6].attn_q.weight
    let layer_6_k = model.blk[6].attn_k.weight
    let layer_6_v = model.blk[6].attn_v.weight
    let layer_6_o = model.blk[6].attn_output.weight
    let layer_6_ffn_norm = model.blk[6].ffn_norm.weight
    let layer_6_gate = model.blk[6].ffn_gate.weight
    let layer_6_up = model.blk[6].ffn_up.weight
    let layer_6_down = model.blk[6].ffn_down.weight

    // Layer 7
    let layer_7_attn_norm = model.blk[7].attn_norm.weight
    let layer_7_q = model.blk[7].attn_q.weight
    let layer_7_k = model.blk[7].attn_k.weight
    let layer_7_v = model.blk[7].attn_v.weight
    let layer_7_o = model.blk[7].attn_output.weight
    let layer_7_ffn_norm = model.blk[7].ffn_norm.weight
    let layer_7_gate = model.blk[7].ffn_gate.weight
    let layer_7_up = model.blk[7].ffn_up.weight
    let layer_7_down = model.blk[7].ffn_down.weight

    // Layer 8
    let layer_8_attn_norm = model.blk[8].attn_norm.weight
    let layer_8_q = model.blk[8].attn_q.weight
    let layer_8_k = model.blk[8].attn_k.weight
    let layer_8_v = model.blk[8].attn_v.weight
    let layer_8_o = model.blk[8].attn_output.weight
    let layer_8_ffn_norm = model.blk[8].ffn_norm.weight
    let layer_8_gate = model.blk[8].ffn_gate.weight
    let layer_8_up = model.blk[8].ffn_up.weight
    let layer_8_down = model.blk[8].ffn_down.weight

    // Layer 9
    let layer_9_attn_norm = model.blk[9].attn_norm.weight
    let layer_9_q = model.blk[9].attn_q.weight
    let layer_9_k = model.blk[9].attn_k.weight
    let layer_9_v = model.blk[9].attn_v.weight
    let layer_9_o = model.blk[9].attn_output.weight
    let layer_9_ffn_norm = model.blk[9].ffn_norm.weight
    let layer_9_gate = model.blk[9].ffn_gate.weight
    let layer_9_up = model.blk[9].ffn_up.weight
    let layer_9_down = model.blk[9].ffn_down.weight

    // Layer 10
    let layer_10_attn_norm = model.blk[10].attn_norm.weight
    let layer_10_q = model.blk[10].attn_q.weight
    let layer_10_k = model.blk[10].attn_k.weight
    let layer_10_v = model.blk[10].attn_v.weight
    let layer_10_o = model.blk[10].attn_output.weight
    let layer_10_ffn_norm = model.blk[10].ffn_norm.weight
    let layer_10_gate = model.blk[10].ffn_gate.weight
    let layer_10_up = model.blk[10].ffn_up.weight
    let layer_10_down = model.blk[10].ffn_down.weight

    // Layer 11
    let layer_11_attn_norm = model.blk[11].attn_norm.weight
    let layer_11_q = model.blk[11].attn_q.weight
    let layer_11_k = model.blk[11].attn_k.weight
    let layer_11_v = model.blk[11].attn_v.weight
    let layer_11_o = model.blk[11].attn_output.weight
    let layer_11_ffn_norm = model.blk[11].ffn_norm.weight
    let layer_11_gate = model.blk[11].ffn_gate.weight
    let layer_11_up = model.blk[11].ffn_up.weight
    let layer_11_down = model.blk[11].ffn_down.weight

    // Layer 12
    let layer_12_attn_norm = model.blk[12].attn_norm.weight
    let layer_12_q = model.blk[12].attn_q.weight
    let layer_12_k = model.blk[12].attn_k.weight
    let layer_12_v = model.blk[12].attn_v.weight
    let layer_12_o = model.blk[12].attn_output.weight
    let layer_12_ffn_norm = model.blk[12].ffn_norm.weight
    let layer_12_gate = model.blk[12].ffn_gate.weight
    let layer_12_up = model.blk[12].ffn_up.weight
    let layer_12_down = model.blk[12].ffn_down.weight

    // Layer 13
    let layer_13_attn_norm = model.blk[13].attn_norm.weight
    let layer_13_q = model.blk[13].attn_q.weight
    let layer_13_k = model.blk[13].attn_k.weight
    let layer_13_v = model.blk[13].attn_v.weight
    let layer_13_o = model.blk[13].attn_output.weight
    let layer_13_ffn_norm = model.blk[13].ffn_norm.weight
    let layer_13_gate = model.blk[13].ffn_gate.weight
    let layer_13_up = model.blk[13].ffn_up.weight
    let layer_13_down = model.blk[13].ffn_down.weight

    // Layer 14
    let layer_14_attn_norm = model.blk[14].attn_norm.weight
    let layer_14_q = model.blk[14].attn_q.weight
    let layer_14_k = model.blk[14].attn_k.weight
    let layer_14_v = model.blk[14].attn_v.weight
    let layer_14_o = model.blk[14].attn_output.weight
    let layer_14_ffn_norm = model.blk[14].ffn_norm.weight
    let layer_14_gate = model.blk[14].ffn_gate.weight
    let layer_14_up = model.blk[14].ffn_up.weight
    let layer_14_down = model.blk[14].ffn_down.weight

    // Layer 15
    let layer_15_attn_norm = model.blk[15].attn_norm.weight
    let layer_15_q = model.blk[15].attn_q.weight
    let layer_15_k = model.blk[15].attn_k.weight
    let layer_15_v = model.blk[15].attn_v.weight
    let layer_15_o = model.blk[15].attn_output.weight
    let layer_15_ffn_norm = model.blk[15].ffn_norm.weight
    let layer_15_gate = model.blk[15].ffn_gate.weight
    let layer_15_up = model.blk[15].ffn_up.weight
    let layer_15_down = model.blk[15].ffn_down.weight

    // Layer 16
    let layer_16_attn_norm = model.blk[16].attn_norm.weight
    let layer_16_q = model.blk[16].attn_q.weight
    let layer_16_k = model.blk[16].attn_k.weight
    let layer_16_v = model.blk[16].attn_v.weight
    let layer_16_o = model.blk[16].attn_output.weight
    let layer_16_ffn_norm = model.blk[16].ffn_norm.weight
    let layer_16_gate = model.blk[16].ffn_gate.weight
    let layer_16_up = model.blk[16].ffn_up.weight
    let layer_16_down = model.blk[16].ffn_down.weight

    // Layer 17
    let layer_17_attn_norm = model.blk[17].attn_norm.weight
    let layer_17_q = model.blk[17].attn_q.weight
    let layer_17_k = model.blk[17].attn_k.weight
    let layer_17_v = model.blk[17].attn_v.weight
    let layer_17_o = model.blk[17].attn_output.weight
    let layer_17_ffn_norm = model.blk[17].ffn_norm.weight
    let layer_17_gate = model.blk[17].ffn_gate.weight
    let layer_17_up = model.blk[17].ffn_up.weight
    let layer_17_down = model.blk[17].ffn_down.weight

    // Layer 18
    let layer_18_attn_norm = model.blk[18].attn_norm.weight
    let layer_18_q = model.blk[18].attn_q.weight
    let layer_18_k = model.blk[18].attn_k.weight
    let layer_18_v = model.blk[18].attn_v.weight
    let layer_18_o = model.blk[18].attn_output.weight
    let layer_18_ffn_norm = model.blk[18].ffn_norm.weight
    let layer_18_gate = model.blk[18].ffn_gate.weight
    let layer_18_up = model.blk[18].ffn_up.weight
    let layer_18_down = model.blk[18].ffn_down.weight

    // Layer 19
    let layer_19_attn_norm = model.blk[19].attn_norm.weight
    let layer_19_q = model.blk[19].attn_q.weight
    let layer_19_k = model.blk[19].attn_k.weight
    let layer_19_v = model.blk[19].attn_v.weight
    let layer_19_o = model.blk[19].attn_output.weight
    let layer_19_ffn_norm = model.blk[19].ffn_norm.weight
    let layer_19_gate = model.blk[19].ffn_gate.weight
    let layer_19_up = model.blk[19].ffn_up.weight
    let layer_19_down = model.blk[19].ffn_down.weight

    // Layer 20
    let layer_20_attn_norm = model.blk[20].attn_norm.weight
    let layer_20_q = model.blk[20].attn_q.weight
    let layer_20_k = model.blk[20].attn_k.weight
    let layer_20_v = model.blk[20].attn_v.weight
    let layer_20_o = model.blk[20].attn_output.weight
    let layer_20_ffn_norm = model.blk[20].ffn_norm.weight
    let layer_20_gate = model.blk[20].ffn_gate.weight
    let layer_20_up = model.blk[20].ffn_up.weight
    let layer_20_down = model.blk[20].ffn_down.weight

    // Layer 21
    let layer_21_attn_norm = model.blk[21].attn_norm.weight
    let layer_21_q = model.blk[21].attn_q.weight
    let layer_21_k = model.blk[21].attn_k.weight
    let layer_21_v = model.blk[21].attn_v.weight
    let layer_21_o = model.blk[21].attn_output.weight
    let layer_21_ffn_norm = model.blk[21].ffn_norm.weight
    let layer_21_gate = model.blk[21].ffn_gate.weight
    let layer_21_up = model.blk[21].ffn_up.weight
    let layer_21_down = model.blk[21].ffn_down.weight

    print("      ✓ All 22 layers loaded successfully")
    print("")

    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("      User:", user_msg)
    print("")

    print("[3/3] Generating response (max 50 tokens)...")
    print("")
    print("      Assistant: ", "")

    let x = embedding(tok_embd, tokens)

    // Build initial KV caches for all 22 layers
    let K0 = linear(x, layer_0_k)
    let V0 = linear(x, layer_0_v)
    let K1 = linear(x, layer_1_k)
    let V1 = linear(x, layer_1_v)
    let K2 = linear(x, layer_2_k)
    let V2 = linear(x, layer_2_v)
    let K3 = linear(x, layer_3_k)
    let V3 = linear(x, layer_3_v)
    let K4 = linear(x, layer_4_k)
    let V4 = linear(x, layer_4_v)
    let K5 = linear(x, layer_5_k)
    let V5 = linear(x, layer_5_v)
    let K6 = linear(x, layer_6_k)
    let V6 = linear(x, layer_6_v)
    let K7 = linear(x, layer_7_k)
    let V7 = linear(x, layer_7_v)
    let K8 = linear(x, layer_8_k)
    let V8 = linear(x, layer_8_v)
    let K9 = linear(x, layer_9_k)
    let V9 = linear(x, layer_9_v)
    let K10 = linear(x, layer_10_k)
    let V10 = linear(x, layer_10_v)
    let K11 = linear(x, layer_11_k)
    let V11 = linear(x, layer_11_v)
    let K12 = linear(x, layer_12_k)
    let V12 = linear(x, layer_12_v)
    let K13 = linear(x, layer_13_k)
    let V13 = linear(x, layer_13_v)
    let K14 = linear(x, layer_14_k)
    let V14 = linear(x, layer_14_v)
    let K15 = linear(x, layer_15_k)
    let V15 = linear(x, layer_15_v)
    let K16 = linear(x, layer_16_k)
    let V16 = linear(x, layer_16_v)
    let K17 = linear(x, layer_17_k)
    let V17 = linear(x, layer_17_v)
    let K18 = linear(x, layer_18_k)
    let V18 = linear(x, layer_18_v)
    let K19 = linear(x, layer_19_k)
    let V19 = linear(x, layer_19_v)
    let K20 = linear(x, layer_20_k)
    let V20 = linear(x, layer_20_v)
    let K21 = linear(x, layer_21_k)
    let V21 = linear(x, layer_21_v)

    // Run through all 22 layers
    let h0 = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)
    let h1 = transformer_layer(h0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)
    let h2 = transformer_layer(h1, layer_2_attn_norm, layer_2_q, layer_2_k, layer_2_v, layer_2_o,
                                layer_2_ffn_norm, layer_2_gate, layer_2_up, layer_2_down, K2, V2)
    let h3 = transformer_layer(h2, layer_3_attn_norm, layer_3_q, layer_3_k, layer_3_v, layer_3_o,
                                layer_3_ffn_norm, layer_3_gate, layer_3_up, layer_3_down, K3, V3)
    let h4 = transformer_layer(h3, layer_4_attn_norm, layer_4_q, layer_4_k, layer_4_v, layer_4_o,
                                layer_4_ffn_norm, layer_4_gate, layer_4_up, layer_4_down, K4, V4)
    let h5 = transformer_layer(h4, layer_5_attn_norm, layer_5_q, layer_5_k, layer_5_v, layer_5_o,
                                layer_5_ffn_norm, layer_5_gate, layer_5_up, layer_5_down, K5, V5)
    let h6 = transformer_layer(h5, layer_6_attn_norm, layer_6_q, layer_6_k, layer_6_v, layer_6_o,
                                layer_6_ffn_norm, layer_6_gate, layer_6_up, layer_6_down, K6, V6)
    let h7 = transformer_layer(h6, layer_7_attn_norm, layer_7_q, layer_7_k, layer_7_v, layer_7_o,
                                layer_7_ffn_norm, layer_7_gate, layer_7_up, layer_7_down, K7, V7)
    let h8 = transformer_layer(h7, layer_8_attn_norm, layer_8_q, layer_8_k, layer_8_v, layer_8_o,
                                layer_8_ffn_norm, layer_8_gate, layer_8_up, layer_8_down, K8, V8)
    let h9 = transformer_layer(h8, layer_9_attn_norm, layer_9_q, layer_9_k, layer_9_v, layer_9_o,
                                layer_9_ffn_norm, layer_9_gate, layer_9_up, layer_9_down, K9, V9)
    let h10 = transformer_layer(h9, layer_10_attn_norm, layer_10_q, layer_10_k, layer_10_v, layer_10_o,
                                layer_10_ffn_norm, layer_10_gate, layer_10_up, layer_10_down, K10, V10)
    let h11 = transformer_layer(h10, layer_11_attn_norm, layer_11_q, layer_11_k, layer_11_v, layer_11_o,
                                layer_11_ffn_norm, layer_11_gate, layer_11_up, layer_11_down, K11, V11)
    let h12 = transformer_layer(h11, layer_12_attn_norm, layer_12_q, layer_12_k, layer_12_v, layer_12_o,
                                layer_12_ffn_norm, layer_12_gate, layer_12_up, layer_12_down, K12, V12)
    let h13 = transformer_layer(h12, layer_13_attn_norm, layer_13_q, layer_13_k, layer_13_v, layer_13_o,
                                layer_13_ffn_norm, layer_13_gate, layer_13_up, layer_13_down, K13, V13)
    let h14 = transformer_layer(h13, layer_14_attn_norm, layer_14_q, layer_14_k, layer_14_v, layer_14_o,
                                layer_14_ffn_norm, layer_14_gate, layer_14_up, layer_14_down, K14, V14)
    let h15 = transformer_layer(h14, layer_15_attn_norm, layer_15_q, layer_15_k, layer_15_v, layer_15_o,
                                layer_15_ffn_norm, layer_15_gate, layer_15_up, layer_15_down, K15, V15)
    let h16 = transformer_layer(h15, layer_16_attn_norm, layer_16_q, layer_16_k, layer_16_v, layer_16_o,
                                layer_16_ffn_norm, layer_16_gate, layer_16_up, layer_16_down, K16, V16)
    let h17 = transformer_layer(h16, layer_17_attn_norm, layer_17_q, layer_17_k, layer_17_v, layer_17_o,
                                layer_17_ffn_norm, layer_17_gate, layer_17_up, layer_17_down, K17, V17)
    let h18 = transformer_layer(h17, layer_18_attn_norm, layer_18_q, layer_18_k, layer_18_v, layer_18_o,
                                layer_18_ffn_norm, layer_18_gate, layer_18_up, layer_18_down, K18, V18)
    let h19 = transformer_layer(h18, layer_19_attn_norm, layer_19_q, layer_19_k, layer_19_v, layer_19_o,
                                layer_19_ffn_norm, layer_19_gate, layer_19_up, layer_19_down, K19, V19)
    let h20 = transformer_layer(h19, layer_20_attn_norm, layer_20_q, layer_20_k, layer_20_v, layer_20_o,
                                layer_20_ffn_norm, layer_20_gate, layer_20_up, layer_20_down, K20, V20)
    let h21 = transformer_layer(h20, layer_21_attn_norm, layer_21_q, layer_21_k, layer_21_v, layer_21_o,
                                layer_21_ffn_norm, layer_21_gate, layer_21_up, layer_21_down, K21, V21)

    let final_norm = rms_norm(h21, output_norm)
    let logits = linear(final_norm, output)

    // Initialize for autoregressive loop
    let temperature = 0.8
    let current_logits = logits
    let continue_generation = true
    let token_count = 0

    let KV0 = K0
    let KV0_V = V0
    let KV1 = K1
    let KV1_V = V1
    let KV2 = K2
    let KV2_V = V2
    let KV3 = K3
    let KV3_V = V3
    let KV4 = K4
    let KV4_V = V4
    let KV5 = K5
    let KV5_V = V5
    let KV6 = K6
    let KV6_V = V6
    let KV7 = K7
    let KV7_V = V7
    let KV8 = K8
    let KV8_V = V8
    let KV9 = K9
    let KV9_V = V9
    let KV10 = K10
    let KV10_V = V10
    let KV11 = K11
    let KV11_V = V11
    let KV12 = K12
    let KV12_V = V12
    let KV13 = K13
    let KV13_V = V13
    let KV14 = K14
    let KV14_V = V14
    let KV15 = K15
    let KV15_V = V15
    let KV16 = K16
    let KV16_V = V16
    let KV17 = K17
    let KV17_V = V17
    let KV18 = K18
    let KV18_V = V18
    let KV19 = K19
    let KV19_V = V19
    let KV20 = K20
    let KV20_V = V20
    let KV21 = K21
    let KV21_V = V21

    // Autoregressive generation (max 10 tokens for demo)
    for step in range(10) {
        if continue_generation {
            let token_id = temperature_sample(current_logits, temperature)
            let text = detokenize_single(tokenizer, token_id, false)
            print(text, "")

            token_count = token_count + 1

            if token_id == EOS_TOKEN {
                continue_generation = false
                print(" <EOS>")
            }

            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)


            // Layer 0
            let nK0 = linear(new_token_emb, layer_0_k)
            let nV0 = linear(new_token_emb, layer_0_v)
            KV0 = concat(KV0, nK0, 0.0)
            KV0_V = concat(KV0_V, nV0, 0.0)
            let nh0 = transformer_layer(new_token_emb, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                        layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, KV0, KV0_V)

            // Layer 1
            let nK1 = linear(nh0, layer_1_k)
            let nV1 = linear(nh0, layer_1_v)
            KV1 = concat(KV1, nK1, 0.0)
            KV1_V = concat(KV1_V, nV1, 0.0)
            let nh1 = transformer_layer(nh0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                        layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, KV1, KV1_V)

            // Layer 2
            let nK2 = linear(nh1, layer_2_k)
            let nV2 = linear(nh1, layer_2_v)
            KV2 = concat(KV2, nK2, 0.0)
            KV2_V = concat(KV2_V, nV2, 0.0)
            let nh2 = transformer_layer(nh1, layer_2_attn_norm, layer_2_q, layer_2_k, layer_2_v, layer_2_o,
                                        layer_2_ffn_norm, layer_2_gate, layer_2_up, layer_2_down, KV2, KV2_V)

            // Layer 3
            let nK3 = linear(nh2, layer_3_k)
            let nV3 = linear(nh2, layer_3_v)
            KV3 = concat(KV3, nK3, 0.0)
            KV3_V = concat(KV3_V, nV3, 0.0)
            let nh3 = transformer_layer(nh2, layer_3_attn_norm, layer_3_q, layer_3_k, layer_3_v, layer_3_o,
                                        layer_3_ffn_norm, layer_3_gate, layer_3_up, layer_3_down, KV3, KV3_V)

            // Layer 4
            let nK4 = linear(nh3, layer_4_k)
            let nV4 = linear(nh3, layer_4_v)
            KV4 = concat(KV4, nK4, 0.0)
            KV4_V = concat(KV4_V, nV4, 0.0)
            let nh4 = transformer_layer(nh3, layer_4_attn_norm, layer_4_q, layer_4_k, layer_4_v, layer_4_o,
                                        layer_4_ffn_norm, layer_4_gate, layer_4_up, layer_4_down, KV4, KV4_V)

            // Layer 5
            let nK5 = linear(nh4, layer_5_k)
            let nV5 = linear(nh4, layer_5_v)
            KV5 = concat(KV5, nK5, 0.0)
            KV5_V = concat(KV5_V, nV5, 0.0)
            let nh5 = transformer_layer(nh4, layer_5_attn_norm, layer_5_q, layer_5_k, layer_5_v, layer_5_o,
                                        layer_5_ffn_norm, layer_5_gate, layer_5_up, layer_5_down, KV5, KV5_V)

            // Layer 6
            let nK6 = linear(nh5, layer_6_k)
            let nV6 = linear(nh5, layer_6_v)
            KV6 = concat(KV6, nK6, 0.0)
            KV6_V = concat(KV6_V, nV6, 0.0)
            let nh6 = transformer_layer(nh5, layer_6_attn_norm, layer_6_q, layer_6_k, layer_6_v, layer_6_o,
                                        layer_6_ffn_norm, layer_6_gate, layer_6_up, layer_6_down, KV6, KV6_V)

            // Layer 7
            let nK7 = linear(nh6, layer_7_k)
            let nV7 = linear(nh6, layer_7_v)
            KV7 = concat(KV7, nK7, 0.0)
            KV7_V = concat(KV7_V, nV7, 0.0)
            let nh7 = transformer_layer(nh6, layer_7_attn_norm, layer_7_q, layer_7_k, layer_7_v, layer_7_o,
                                        layer_7_ffn_norm, layer_7_gate, layer_7_up, layer_7_down, KV7, KV7_V)

            // Layer 8
            let nK8 = linear(nh7, layer_8_k)
            let nV8 = linear(nh7, layer_8_v)
            KV8 = concat(KV8, nK8, 0.0)
            KV8_V = concat(KV8_V, nV8, 0.0)
            let nh8 = transformer_layer(nh7, layer_8_attn_norm, layer_8_q, layer_8_k, layer_8_v, layer_8_o,
                                        layer_8_ffn_norm, layer_8_gate, layer_8_up, layer_8_down, KV8, KV8_V)

            // Layer 9
            let nK9 = linear(nh8, layer_9_k)
            let nV9 = linear(nh8, layer_9_v)
            KV9 = concat(KV9, nK9, 0.0)
            KV9_V = concat(KV9_V, nV9, 0.0)
            let nh9 = transformer_layer(nh8, layer_9_attn_norm, layer_9_q, layer_9_k, layer_9_v, layer_9_o,
                                        layer_9_ffn_norm, layer_9_gate, layer_9_up, layer_9_down, KV9, KV9_V)

            // Layer 10
            let nK10 = linear(nh9, layer_10_k)
            let nV10 = linear(nh9, layer_10_v)
            KV10 = concat(KV10, nK10, 0.0)
            KV10_V = concat(KV10_V, nV10, 0.0)
            let nh10 = transformer_layer(nh9, layer_10_attn_norm, layer_10_q, layer_10_k, layer_10_v, layer_10_o,
                                        layer_10_ffn_norm, layer_10_gate, layer_10_up, layer_10_down, KV10, KV10_V)

            // Layer 11
            let nK11 = linear(nh10, layer_11_k)
            let nV11 = linear(nh10, layer_11_v)
            KV11 = concat(KV11, nK11, 0.0)
            KV11_V = concat(KV11_V, nV11, 0.0)
            let nh11 = transformer_layer(nh10, layer_11_attn_norm, layer_11_q, layer_11_k, layer_11_v, layer_11_o,
                                        layer_11_ffn_norm, layer_11_gate, layer_11_up, layer_11_down, KV11, KV11_V)

            // Layer 12
            let nK12 = linear(nh11, layer_12_k)
            let nV12 = linear(nh11, layer_12_v)
            KV12 = concat(KV12, nK12, 0.0)
            KV12_V = concat(KV12_V, nV12, 0.0)
            let nh12 = transformer_layer(nh11, layer_12_attn_norm, layer_12_q, layer_12_k, layer_12_v, layer_12_o,
                                        layer_12_ffn_norm, layer_12_gate, layer_12_up, layer_12_down, KV12, KV12_V)

            // Layer 13
            let nK13 = linear(nh12, layer_13_k)
            let nV13 = linear(nh12, layer_13_v)
            KV13 = concat(KV13, nK13, 0.0)
            KV13_V = concat(KV13_V, nV13, 0.0)
            let nh13 = transformer_layer(nh12, layer_13_attn_norm, layer_13_q, layer_13_k, layer_13_v, layer_13_o,
                                        layer_13_ffn_norm, layer_13_gate, layer_13_up, layer_13_down, KV13, KV13_V)

            // Layer 14
            let nK14 = linear(nh13, layer_14_k)
            let nV14 = linear(nh13, layer_14_v)
            KV14 = concat(KV14, nK14, 0.0)
            KV14_V = concat(KV14_V, nV14, 0.0)
            let nh14 = transformer_layer(nh13, layer_14_attn_norm, layer_14_q, layer_14_k, layer_14_v, layer_14_o,
                                        layer_14_ffn_norm, layer_14_gate, layer_14_up, layer_14_down, KV14, KV14_V)

            // Layer 15
            let nK15 = linear(nh14, layer_15_k)
            let nV15 = linear(nh14, layer_15_v)
            KV15 = concat(KV15, nK15, 0.0)
            KV15_V = concat(KV15_V, nV15, 0.0)
            let nh15 = transformer_layer(nh14, layer_15_attn_norm, layer_15_q, layer_15_k, layer_15_v, layer_15_o,
                                        layer_15_ffn_norm, layer_15_gate, layer_15_up, layer_15_down, KV15, KV15_V)

            // Layer 16
            let nK16 = linear(nh15, layer_16_k)
            let nV16 = linear(nh15, layer_16_v)
            KV16 = concat(KV16, nK16, 0.0)
            KV16_V = concat(KV16_V, nV16, 0.0)
            let nh16 = transformer_layer(nh15, layer_16_attn_norm, layer_16_q, layer_16_k, layer_16_v, layer_16_o,
                                        layer_16_ffn_norm, layer_16_gate, layer_16_up, layer_16_down, KV16, KV16_V)

            // Layer 17
            let nK17 = linear(nh16, layer_17_k)
            let nV17 = linear(nh16, layer_17_v)
            KV17 = concat(KV17, nK17, 0.0)
            KV17_V = concat(KV17_V, nV17, 0.0)
            let nh17 = transformer_layer(nh16, layer_17_attn_norm, layer_17_q, layer_17_k, layer_17_v, layer_17_o,
                                        layer_17_ffn_norm, layer_17_gate, layer_17_up, layer_17_down, KV17, KV17_V)

            // Layer 18
            let nK18 = linear(nh17, layer_18_k)
            let nV18 = linear(nh17, layer_18_v)
            KV18 = concat(KV18, nK18, 0.0)
            KV18_V = concat(KV18_V, nV18, 0.0)
            let nh18 = transformer_layer(nh17, layer_18_attn_norm, layer_18_q, layer_18_k, layer_18_v, layer_18_o,
                                        layer_18_ffn_norm, layer_18_gate, layer_18_up, layer_18_down, KV18, KV18_V)

            // Layer 19
            let nK19 = linear(nh18, layer_19_k)
            let nV19 = linear(nh18, layer_19_v)
            KV19 = concat(KV19, nK19, 0.0)
            KV19_V = concat(KV19_V, nV19, 0.0)
            let nh19 = transformer_layer(nh18, layer_19_attn_norm, layer_19_q, layer_19_k, layer_19_v, layer_19_o,
                                        layer_19_ffn_norm, layer_19_gate, layer_19_up, layer_19_down, KV19, KV19_V)

            // Layer 20
            let nK20 = linear(nh19, layer_20_k)
            let nV20 = linear(nh19, layer_20_v)
            KV20 = concat(KV20, nK20, 0.0)
            KV20_V = concat(KV20_V, nV20, 0.0)
            let nh20 = transformer_layer(nh19, layer_20_attn_norm, layer_20_q, layer_20_k, layer_20_v, layer_20_o,
                                        layer_20_ffn_norm, layer_20_gate, layer_20_up, layer_20_down, KV20, KV20_V)

            // Layer 21
            let nK21 = linear(nh20, layer_21_k)
            let nV21 = linear(nh20, layer_21_v)
            KV21 = concat(KV21, nK21, 0.0)
            KV21_V = concat(KV21_V, nV21, 0.0)
            let nh21 = transformer_layer(nh20, layer_21_attn_norm, layer_21_q, layer_21_k, layer_21_v, layer_21_o,
                                        layer_21_ffn_norm, layer_21_gate, layer_21_up, layer_21_down, KV21, KV21_V)

            let norm_new = rms_norm(nh21, output_norm)
            current_logits = linear(norm_new, output)
        }
    }

    print("")
    print("")
    print("=== Generation Complete ===")
    print("Tokens generated:", token_count)
}

