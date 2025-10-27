// Chat Demo with Token Decoding (f32, 2 layers)
// Goal: Display generated text in readable form

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
    print("=== Chat Demo with Decoding (f32, 2 layers) ===")
    print("")

    // Load model and tokenizer
    print("[1/4] Loading...")
    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = home + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model and tokenizer loaded")
    print("")

    // Load weights (2 layers only)
    print("[2/4] Loading weights...")
    let tok_embd = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output = get_tensor(model, "output.weight")

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
    print("      ✓ Loaded")
    print("")

    // Prepare prompt
    print("[3/4] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let prompt_tokens = tokenize(tokenizer, chat_prompt, true)
    print("      User:", user_msg)
    print("")

    // Initial forward pass
    print("[4/4] Generating response...")
    let x = embedding(tok_embd, prompt_tokens)

    let K0 = linear(x, layer_0_k)
    let V0 = linear(x, layer_0_v)
    let K1 = linear(x, layer_1_k)
    let V1 = linear(x, layer_1_v)

    let h0 = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)
    let h1 = transformer_layer(h0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)

    let final_norm = rms_norm(h1, output_norm)
    let logits = linear(final_norm, output)

    // Generate and decode token
    let temperature = 0.8
    let token_id = temperature_sample(logits, temperature)

    print("      Token ID:", token_id)

    let decoded_text = detokenize_single(tokenizer, token_id, false)
    print("      Decoded text:", decoded_text)
    print("")

    print("=== Generation Complete ===")
    print("")
    print("Success! Generated text:", decoded_text)
    print("")
    print("Note:")
    print("  - 2-layer model (not full 22 layers)")
    print("  - Single token generation")
    print("  - Token decoding now working!")
    print("")
    print("Next: Generate multiple tokens in autoregressive loop")
}
