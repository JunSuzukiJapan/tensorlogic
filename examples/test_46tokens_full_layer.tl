// Test full transformer layer with 46 tokens

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
    print("      [Attention] Starting...")
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])
    print("      [Attention] Reshaped")

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)
    print("      [Attention] RoPE done")

    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])
    print("      [Attention] GQA expansion done")

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    print("      [Attention] Scores computed")

    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)
    print("      [Attention] Mask applied")

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    print("      [Attention] Output projection...")

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
    print("    [Layer] Starting")

    print("    [Layer] RMS norm 1...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("    [Layer] RMS norm 1 done")

    print("    [Layer] QKV projections...")
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    print("    [Layer] QKV projections done")

    print("    [Layer] Calling attention...")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    print("    [Layer] Attention done")

    print("    [Layer] First residual...")
    let x1 = x + attn_out
    print("    [Layer] First residual done")

    print("    [Layer] RMS norm 2...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("    [Layer] RMS norm 2 done")

    print("    [Layer] FFN...")
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    print("    [Layer] FFN done")

    print("    [Layer] Second residual...")
    x1 + ffn_out
}

main {
    print("=== Testing Full Layer with 46 Tokens ===")
    print("")

    print("[1] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("    ✓ Loaded")

    print("[2] Creating 46 token input...")
    let emb_weight = get_tensor(model, "token_embd.weight")
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("    ✓ Tokens:", len(tokens))

    let embeddings = embedding(emb_weight, tokens)
    print("    ✓ Embeddings computed")

    print("[3] Processing layer 0...")
    let attn_norm = model.blk[0].attn_norm.weight
    let W_q = model.blk[0].attn_q.weight
    let W_k = model.blk[0].attn_k.weight
    let W_v = model.blk[0].attn_v.weight
    let W_o = model.blk[0].attn_output.weight
    let ffn_norm = model.blk[0].ffn_norm.weight
    let W_gate = model.blk[0].ffn_gate.weight
    let W_up = model.blk[0].ffn_up.weight
    let W_down = model.blk[0].ffn_down.weight

    let output = transformer_layer(embeddings, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)

    print("    ✓ Layer 0 completed")
    let out_shape = shape(output)
    print("    Output shape:", out_shape)

    print("")
    print("=== Test passed! ===")
}
