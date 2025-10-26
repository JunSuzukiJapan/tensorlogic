// 10-Layer Chat with float32 (No Scaling)
// Standard transformer architecture without residual scaling

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
    result := linear(silu(gate) * up, W_down)
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
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_exp = reshape(K_heads, [cache_len_f, 4.0, 1.0, 64.0])
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

    // Residual connection (NO SCALING)
    let after_attn = x + attn_out

    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)

    // Residual connection (NO SCALING)
    result := after_attn + ffn_out
}

main {
    print("=== TinyLlama 10-Layer Chat with float32 (No Scaling) ===\n")

    print("[1/4] Loading model and tokenizer...")
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = "~/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model loaded: TinyLlama 1.1B Chat\n")

    print("[2/4] Loading 10 Transformer layers...")

    // Load embeddings
    let tok_embd = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output = get_tensor(model, "output.weight")

    // Load 10 layers (0-9)
    let layer_0_attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let layer_0_q = get_tensor(model, "blk.0.attn_q.weight")
    let layer_0_k = get_tensor(model, "blk.0.attn_k.weight")
    let layer_0_v = get_tensor(model, "blk.0.attn_v.weight")
    let layer_0_o = get_tensor(model, "blk.0.attn_output.weight")
    let layer_0_ffn_norm = get_tensor(model, "blk.0.ffn_norm.weight")
    let layer_0_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    let layer_0_up = get_tensor(model, "blk.0.ffn_up.weight")
    let layer_0_down = get_tensor(model, "blk.0.ffn_down.weight")

    let layer_1_attn_norm = get_tensor(model, "blk.1.attn_norm.weight")
    let layer_1_q = get_tensor(model, "blk.1.attn_q.weight")
    let layer_1_k = get_tensor(model, "blk.1.attn_k.weight")
    let layer_1_v = get_tensor(model, "blk.1.attn_v.weight")
    let layer_1_o = get_tensor(model, "blk.1.attn_output.weight")
    let layer_1_ffn_norm = get_tensor(model, "blk.1.ffn_norm.weight")
    let layer_1_gate = get_tensor(model, "blk.1.ffn_gate.weight")
    let layer_1_up = get_tensor(model, "blk.1.ffn_up.weight")
    let layer_1_down = get_tensor(model, "blk.1.ffn_down.weight")

    let layer_2_attn_norm = get_tensor(model, "blk.2.attn_norm.weight")
    let layer_2_q = get_tensor(model, "blk.2.attn_q.weight")
    let layer_2_k = get_tensor(model, "blk.2.attn_k.weight")
    let layer_2_v = get_tensor(model, "blk.2.attn_v.weight")
    let layer_2_o = get_tensor(model, "blk.2.attn_output.weight")
    let layer_2_ffn_norm = get_tensor(model, "blk.2.ffn_norm.weight")
    let layer_2_gate = get_tensor(model, "blk.2.ffn_gate.weight")
    let layer_2_up = get_tensor(model, "blk.2.ffn_up.weight")
    let layer_2_down = get_tensor(model, "blk.2.ffn_down.weight")

    let layer_3_attn_norm = get_tensor(model, "blk.3.attn_norm.weight")
    let layer_3_q = get_tensor(model, "blk.3.attn_q.weight")
    let layer_3_k = get_tensor(model, "blk.3.attn_k.weight")
    let layer_3_v = get_tensor(model, "blk.3.attn_v.weight")
    let layer_3_o = get_tensor(model, "blk.3.attn_output.weight")
    let layer_3_ffn_norm = get_tensor(model, "blk.3.ffn_norm.weight")
    let layer_3_gate = get_tensor(model, "blk.3.ffn_gate.weight")
    let layer_3_up = get_tensor(model, "blk.3.ffn_up.weight")
    let layer_3_down = get_tensor(model, "blk.3.ffn_down.weight")

    let layer_4_attn_norm = get_tensor(model, "blk.4.attn_norm.weight")
    let layer_4_q = get_tensor(model, "blk.4.attn_q.weight")
    let layer_4_k = get_tensor(model, "blk.4.attn_k.weight")
    let layer_4_v = get_tensor(model, "blk.4.attn_v.weight")
    let layer_4_o = get_tensor(model, "blk.4.attn_output.weight")
    let layer_4_ffn_norm = get_tensor(model, "blk.4.ffn_norm.weight")
    let layer_4_gate = get_tensor(model, "blk.4.ffn_gate.weight")
    let layer_4_up = get_tensor(model, "blk.4.ffn_up.weight")
    let layer_4_down = get_tensor(model, "blk.4.ffn_down.weight")

    let layer_5_attn_norm = get_tensor(model, "blk.5.attn_norm.weight")
    let layer_5_q = get_tensor(model, "blk.5.attn_q.weight")
    let layer_5_k = get_tensor(model, "blk.5.attn_k.weight")
    let layer_5_v = get_tensor(model, "blk.5.attn_v.weight")
    let layer_5_o = get_tensor(model, "blk.5.attn_output.weight")
    let layer_5_ffn_norm = get_tensor(model, "blk.5.ffn_norm.weight")
    let layer_5_gate = get_tensor(model, "blk.5.ffn_gate.weight")
    let layer_5_up = get_tensor(model, "blk.5.ffn_up.weight")
    let layer_5_down = get_tensor(model, "blk.5.ffn_down.weight")

    let layer_6_attn_norm = get_tensor(model, "blk.6.attn_norm.weight")
    let layer_6_q = get_tensor(model, "blk.6.attn_q.weight")
    let layer_6_k = get_tensor(model, "blk.6.attn_k.weight")
    let layer_6_v = get_tensor(model, "blk.6.attn_v.weight")
    let layer_6_o = get_tensor(model, "blk.6.attn_output.weight")
    let layer_6_ffn_norm = get_tensor(model, "blk.6.ffn_norm.weight")
    let layer_6_gate = get_tensor(model, "blk.6.ffn_gate.weight")
    let layer_6_up = get_tensor(model, "blk.6.ffn_up.weight")
    let layer_6_down = get_tensor(model, "blk.6.ffn_down.weight")

    let layer_7_attn_norm = get_tensor(model, "blk.7.attn_norm.weight")
    let layer_7_q = get_tensor(model, "blk.7.attn_q.weight")
    let layer_7_k = get_tensor(model, "blk.7.attn_k.weight")
    let layer_7_v = get_tensor(model, "blk.7.attn_v.weight")
    let layer_7_o = get_tensor(model, "blk.7.attn_output.weight")
    let layer_7_ffn_norm = get_tensor(model, "blk.7.ffn_norm.weight")
    let layer_7_gate = get_tensor(model, "blk.7.ffn_gate.weight")
    let layer_7_up = get_tensor(model, "blk.7.ffn_up.weight")
    let layer_7_down = get_tensor(model, "blk.7.ffn_down.weight")

    let layer_8_attn_norm = get_tensor(model, "blk.8.attn_norm.weight")
    let layer_8_q = get_tensor(model, "blk.8.attn_q.weight")
    let layer_8_k = get_tensor(model, "blk.8.attn_k.weight")
    let layer_8_v = get_tensor(model, "blk.8.attn_v.weight")
    let layer_8_o = get_tensor(model, "blk.8.attn_output.weight")
    let layer_8_ffn_norm = get_tensor(model, "blk.8.ffn_norm.weight")
    let layer_8_gate = get_tensor(model, "blk.8.ffn_gate.weight")
    let layer_8_up = get_tensor(model, "blk.8.ffn_up.weight")
    let layer_8_down = get_tensor(model, "blk.8.ffn_down.weight")

    let layer_9_attn_norm = get_tensor(model, "blk.9.attn_norm.weight")
    let layer_9_q = get_tensor(model, "blk.9.attn_q.weight")
    let layer_9_k = get_tensor(model, "blk.9.attn_k.weight")
    let layer_9_v = get_tensor(model, "blk.9.attn_v.weight")
    let layer_9_o = get_tensor(model, "blk.9.attn_output.weight")
    let layer_9_ffn_norm = get_tensor(model, "blk.9.ffn_norm.weight")
    let layer_9_gate = get_tensor(model, "blk.9.ffn_gate.weight")
    let layer_9_up = get_tensor(model, "blk.9.ffn_up.weight")
    let layer_9_down = get_tensor(model, "blk.9.ffn_down.weight")

    print("      ✓ All 10 layers loaded\n")

    print("[3/4] Preparing chat...")
    let user_msg = " Hello "
    let chat_fmt = format_chat(user_msg)
    let tokens = encode(tokenizer, chat_fmt)
    print("      User: \"", user_msg, "\"")
    print("      Tokens: ", tokens, "\n")

    print("[4/4] Generating response...")
    print("      (Using 10 Transformer layers + Temperature=0.8)\n")

    let seq_len = token_count(tokens)
    let x = token_embedding(tok_embd, tokens)

    // Build KV caches for all 10 layers
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

    // Run through all 10 layers
    let h0 = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o, layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)
    let h1 = transformer_layer(h0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o, layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)
    let h2 = transformer_layer(h1, layer_2_attn_norm, layer_2_q, layer_2_k, layer_2_v, layer_2_o, layer_2_ffn_norm, layer_2_gate, layer_2_up, layer_2_down, K2, V2)
    let h3 = transformer_layer(h2, layer_3_attn_norm, layer_3_q, layer_3_k, layer_3_v, layer_3_o, layer_3_ffn_norm, layer_3_gate, layer_3_up, layer_3_down, K3, V3)
    let h4 = transformer_layer(h3, layer_4_attn_norm, layer_4_q, layer_4_k, layer_4_v, layer_4_o, layer_4_ffn_norm, layer_4_gate, layer_4_up, layer_4_down, K4, V4)
    let h5 = transformer_layer(h4, layer_5_attn_norm, layer_5_q, layer_5_k, layer_5_v, layer_5_o, layer_5_ffn_norm, layer_5_gate, layer_5_up, layer_5_down, K5, V5)
    let h6 = transformer_layer(h5, layer_6_attn_norm, layer_6_q, layer_6_k, layer_6_v, layer_6_o, layer_6_ffn_norm, layer_6_gate, layer_6_up, layer_6_down, K6, V6)
    let h7 = transformer_layer(h6, layer_7_attn_norm, layer_7_q, layer_7_k, layer_7_v, layer_7_o, layer_7_ffn_norm, layer_7_gate, layer_7_up, layer_7_down, K7, V7)
    let h8 = transformer_layer(h7, layer_8_attn_norm, layer_8_q, layer_8_k, layer_8_v, layer_8_o, layer_8_ffn_norm, layer_8_gate, layer_8_up, layer_8_down, K8, V8)
    let h9 = transformer_layer(h8, layer_9_attn_norm, layer_9_q, layer_9_k, layer_9_v, layer_9_o, layer_9_ffn_norm, layer_9_gate, layer_9_up, layer_9_down, K9, V9)

    let final_norm = rms_norm(h9, output_norm)
    let logits = linear(final_norm, output)

    // Generate 5 tokens
    let temperature = 0.8

    for i in range(5) {
        print("      [", i + 1, "/5] Generating...")
        let next_token = temperature_sample(logits, temperature)
        print("            Token: ", next_token)
    }

    print("\n=== Generation Complete ===\n")
    print("float32 precision allows for higher numerical stability.\n")
    print("Expected: Consistent logit values without scaling.\n")
}
