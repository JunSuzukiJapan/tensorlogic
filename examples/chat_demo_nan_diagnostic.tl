// Diagnostic test to trace where NaN appears
// Adds sum checks after each operation

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

    // Generate mask for this layer
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
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    x1 + ffn_out
}

main {
    print("=== NaN Diagnostic Test ===")
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
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("    ✓ Tokens:", len(tokens))

    print("[4] Computing embeddings...")
    let embeddings = embedding(emb_weight, tokens)
    print("    ✓ Embeddings computed")

    // CHECK 1: Embeddings
    print("    [CHECK] Embeddings sum:", sum(embeddings))
    print("    [CHECK] First embedding value:", embeddings[0, 0])

    print("[5] Processing Layer 0 with detailed checks...")
    let x = embeddings

    // Layer 0
    print("    [5.1] Getting Layer 0 weights...")
    let attn_norm = model.blk[0].attn_norm.weight
    let W_q = model.blk[0].attn_q.weight
    let W_k = model.blk[0].attn_k.weight
    let W_v = model.blk[0].attn_v.weight
    let W_o = model.blk[0].attn_output.weight
    let ffn_norm = model.blk[0].ffn_norm.weight
    let W_gate = model.blk[0].ffn_gate.weight
    let W_up = model.blk[0].ffn_up.weight
    let W_down = model.blk[0].ffn_down.weight

    // CHECK 2: After rms_norm (pre-attention)
    print("    [5.2] rms_norm (pre-attention)...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("    [CHECK] After rms_norm sum:", sum(x_norm1))
    print("    [CHECK] After rms_norm first value:", x_norm1[0, 0])

    // CHECK 3: After Q, K, V projections
    print("    [5.3] Q, K, V projections...")
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    print("    [CHECK] Q sum:", sum(Q))
    print("    [CHECK] K sum:", sum(K))
    print("    [CHECK] V sum:", sum(V))
    print("    [CHECK] Q first value:", Q[0, 0])

    // CHECK 4: After attention
    print("    [5.4] Attention...")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    print("    [CHECK] Attention output sum:", sum(attn_out))
    print("    [CHECK] Attention output first value:", attn_out[0, 0])

    // CHECK 5: After residual connection
    print("    [5.5] Residual connection...")
    let x1 = x + attn_out
    print("    [CHECK] After residual sum:", sum(x1))
    print("    [CHECK] After residual first value:", x1[0, 0])

    // CHECK 6: After rms_norm (pre-FFN)
    print("    [5.6] rms_norm (pre-FFN)...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("    [CHECK] After FFN rms_norm sum:", sum(x_norm2))
    print("    [CHECK] After FFN rms_norm first value:", x_norm2[0, 0])

    // CHECK 7: After FFN
    print("    [5.7] FFN...")
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    print("    [CHECK] FFN output sum:", sum(ffn_out))
    print("    [CHECK] FFN output first value:", ffn_out[0, 0])

    // CHECK 8: After final residual
    print("    [5.8] Final residual...")
    let x_out = x1 + ffn_out
    print("    [CHECK] Layer 0 output sum:", sum(x_out))
    print("    [CHECK] Layer 0 output first value:", x_out[0, 0])

    print("    ✓ Layer 0 done")

    // Continue with 2 more layers
    x = x_out
    for layer_idx in range_i(1, 3) {
        print("    Layer", layer_idx, "...")
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
        print("    [CHECK] Layer", layer_idx, "output sum:", sum(x))
        print("    [CHECK] Layer", layer_idx, "output first value:", x[0, 0])
        print("    Layer", layer_idx, "done")
    }
    print("    ✓ 3 layers completed")

    print("[6] Final processing...")

    print("    [6.1] slice_last()...")
    let last_hidden_1d = slice_last(x, 0)
    print("    [CHECK] After slice_last sum:", sum(last_hidden_1d))
    print("    [CHECK] After slice_last first value:", last_hidden_1d[0])
    print("    ✓ slice_last() done")

    print("    [6.2] reshape()...")
    let last_hidden = reshape(last_hidden_1d, [1, 2048])
    print("    [CHECK] After reshape sum:", sum(last_hidden))
    print("    ✓ reshape() done")

    print("    [6.3] rms_norm()...")
    let normed = rms_norm(last_hidden, output_norm_weight)
    print("    [CHECK] After final rms_norm sum:", sum(normed))
    print("    [CHECK] After final rms_norm first value:", normed[0, 0])
    print("    ✓ rms_norm() done")

    print("    [6.4] linear() for logits (BIG: [1,2048] @ [32000,2048])...")
    let logits = linear(normed, output_weight)
    print("    [CHECK] Logits sum:", sum(logits))
    print("    [CHECK] First 5 logit values:")
    print("      logits[0, 0] =", logits[0, 0])
    print("      logits[0, 1] =", logits[0, 1])
    print("      logits[0, 2] =", logits[0, 2])
    print("      logits[0, 3] =", logits[0, 3])
    print("      logits[0, 4] =", logits[0, 4])
    print("    ✓ linear() done")

    print("    [6.5] Checking logits shape...")
    let logits_shape = shape(logits)
    print("    Logits shape:", logits_shape)

    print("[7] Sampling...")
    print("    [7.1] temperature_sample()...")
    let next_token = temperature_sample(logits, 0.8)
    print("    ✓ Sampled token:", next_token)

    print("")
    print("=== Diagnostic test complete ===")
}
