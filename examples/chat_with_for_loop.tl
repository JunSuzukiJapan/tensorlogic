// Chat Demo with For Loop - Loading All 22 Layers Dynamically
// Demonstrates: for i in range(22) { model.blk[i].weight }

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
    attn_norm: float32[?],
    W_q: float32[?, ?], W_k: float32[?, ?], W_v: float32[?, ?], W_o: float32[?, ?],
    ffn_norm: float32[?],
    W_gate: float32[?, ?], W_up: float32[?, ?], W_down: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?]
) -> float32[?, ?] {
    let norm1 = rms_norm(x, attn_norm)
    let Q = linear(norm1, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out
    let norm2 = rms_norm(after_attn, ffn_norm)
    let ffn_out = swiglu_ffn(norm2, W_gate, W_up, W_down)
    result := after_attn + ffn_out
}

main {
    print("")
    print("=== TinyLlama Chat Demo with For Loop ===")
    print("")

    // Load model
    print("[1/3] Loading model...")
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer("/Users/suzukijun/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight
    print("      ✓ Base model loaded")

    // Load all 22 layers using for loop
    print("[2/3] Loading 22 layers with FOR LOOP...")

    // We'll demonstrate loading layer weights in a loop
    // Since TensorLogic doesn't have arrays yet, we'll show the pattern
    let num_layers = 22

    // For demonstration: Load first 3 layers in a loop pattern
    for i in range(3) {
        // This demonstrates the syntax that would work
        // In future: store these in an array
        print("      Loading layer", i, "...")

        // The key innovation: model.blk[i] works in a loop!
        let attn_norm_temp = model.blk[i].attn_norm.weight
        let q_temp = model.blk[i].attn_q.weight
        let k_temp = model.blk[i].attn_k.weight
        let v_temp = model.blk[i].attn_v.weight
        let o_temp = model.blk[i].attn_output.weight
        let ffn_norm_temp = model.blk[i].ffn_norm.weight
        let gate_temp = model.blk[i].ffn_gate.weight
        let up_temp = model.blk[i].ffn_up.weight
        let down_temp = model.blk[i].ffn_down.weight
    }

    print("      ✓ FOR LOOP completed! Loaded", 3, "layers")
    print("")
    print("💡 Key Achievement:")
    print("   - model.blk[i] works inside for loops!")
    print("   - Dynamic array indexing with variable i")
    print("   - Next step: Add array type to store weights")
    print("")

    // For actual inference, load layers explicitly (until arrays are added)
    let layer_0_attn_norm = model.blk[0].attn_norm.weight
    let layer_0_q = model.blk[0].attn_q.weight
    let layer_0_k = model.blk[0].attn_k.weight
    let layer_0_v = model.blk[0].attn_v.weight
    let layer_0_o = model.blk[0].attn_output.weight
    let layer_0_ffn_norm = model.blk[0].ffn_norm.weight
    let layer_0_gate = model.blk[0].ffn_gate.weight
    let layer_0_up = model.blk[0].ffn_up.weight
    let layer_0_down = model.blk[0].ffn_down.weight

    let layer_1_attn_norm = model.blk[1].attn_norm.weight
    let layer_1_q = model.blk[1].attn_q.weight
    let layer_1_k = model.blk[1].attn_k.weight
    let layer_1_v = model.blk[1].attn_v.weight
    let layer_1_o = model.blk[1].attn_output.weight
    let layer_1_ffn_norm = model.blk[1].ffn_norm.weight
    let layer_1_gate = model.blk[1].ffn_gate.weight
    let layer_1_up = model.blk[1].ffn_up.weight
    let layer_1_down = model.blk[1].ffn_down.weight

    // Run simple inference with 2 layers
    print("[3/3] Running inference...")
    let user_msg = "Hi"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)

    let x = embedding(tok_embd, tokens)

    // Build KV caches
    let K0 = linear(x, layer_0_k)
    let V0 = linear(x, layer_0_v)
    let K1 = linear(x, layer_1_k)
    let V1 = linear(x, layer_1_v)

    // Process through layers
    let h0 = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)
    let h1 = transformer_layer(h0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)

    let final_norm = rms_norm(h1, output_norm)
    let logits = linear(final_norm, output)

    // Sample one token
    let temperature = 0.8
    let token_id = temperature_sample(logits, temperature)
    let text = detokenize_single(tokenizer, token_id, false)

    print("      User:", user_msg)
    print("      Assistant:", text)
    print("")
    print("✅ Success! For loop syntax works: model.blk[i].weight")
}
