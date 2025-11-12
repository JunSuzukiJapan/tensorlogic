// Minimal test to reproduce Layer 1 hang
// Tests 2 generation steps with detailed logging

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
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    linear(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125

    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    linear(attn_reshaped, W_o)
}

fn transformer_layer(
    x: float16[?, ?],
    attn_norm: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    ffn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    print("      [layer] rms_norm (pre-attention)...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("      [layer] linear Q...")
    let Q = linear(x_norm1, W_q)
    print("      [layer] linear K...")
    let K = linear(x_norm1, W_k)
    print("      [layer] linear V...")
    let V = linear(x_norm1, W_v)
    print("      [layer] attention...")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    print("      [layer] residual 1...")
    let x1 = x + attn_out
    print("      [layer] rms_norm (pre-FFN)...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("      [layer] FFN...")
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    print("      [layer] residual 2...")
    x1 + ffn_out
}

main {
    print("=== 10 Layer Test ===")
    print("")

    print("[1] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("    ✓ Loaded")

    print("[2] Getting weights...")
    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("    ✓ Weights loaded")

    print("[3] Tokenizing...")
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("    ✓ Tokens:", len(tokens))

    print("[4] Generation loop (2 steps)...")
    let gen_tokens = tokens

    // Step 0
    print("")
    print("  === STEP 0 (len=", len(gen_tokens), ") ===")
    let emb0 = embedding(emb_weight, gen_tokens)

    let x = emb0
    for layer_idx in range_i(10) {
        print("    Layer", layer_idx, "starting...")
        let attn_norm = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight
        let ffn_norm = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        print("    Layer", layer_idx, "done")
    }

    let last0 = slice_last(x, 0)
    let reshaped0 = reshape(last0, [1, 2048])
    let normed0 = rms_norm(reshaped0, output_norm_weight)
    let logits0 = linear(normed0, output_weight)
    let token0 = temperature_sample(logits0, 0.8)
    print("  ✓ Step 0 sampled:", token0)

    gen_tokens = append(gen_tokens, token0)

    // Step 1
    print("")
    print("  === STEP 1 (len=", len(gen_tokens), ") ===")
    let emb1 = embedding(emb_weight, gen_tokens)

    x = emb1
    for layer_idx in range_i(10) {
        print("    Layer", layer_idx, "starting...")
        let attn_norm = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight
        let ffn_norm = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        print("    Layer", layer_idx, "done")
    }

    let last1 = slice_last(x, 0)
    let reshaped1 = reshape(last1, [1, 2048])
    let normed1 = rms_norm(reshaped1, output_norm_weight)
    let logits1 = linear(normed1, output_weight)
    let token1 = temperature_sample(logits1, 0.8)
    print("  ✓ Step 1 sampled:", token1)

    print("")
    print("=== Test passed! ===")
}
