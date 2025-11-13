// Debug Layer 11 in detail to find exact NaN source

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn swiglu_ffn_debug(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    print("      [FFN] Input sum:", sum(x))

    let gate = linear(x, W_gate)
    print("      [FFN] gate sum:", sum(gate))

    let sig = sigmoid(gate)
    print("      [FFN] sigmoid(gate) sum:", sum(sig))

    let gate_act = silu(gate)
    print("      [FFN] gate_act (silu) sum:", sum(gate_act))

    let up = linear(x, W_up)
    print("      [FFN] up sum:", sum(up))

    let intermediate = gate_act * up
    print("      [FFN] intermediate sum:", sum(intermediate))

    let output = linear(intermediate, W_down)
    print("      [FFN] output sum:", sum(output))

    output
}

fn tinyllama_gqa_attention_debug(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    print("      [ATT] Q sum:", sum(Q))
    print("      [ATT] K sum:", sum(K))
    print("      [ATT] V sum:", sum(V))

    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)
    print("      [ATT] After RoPE - Q_rope sum:", sum(Q_rope))
    print("      [ATT] After RoPE - K_rope sum:", sum(K_rope))

    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("      [ATT] scores sum:", sum(scores))

    let scaled_scores = scores * 0.125
    print("      [ATT] scaled_scores sum:", sum(scaled_scores))

    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)
    print("      [ATT] masked_scores sum:", sum(masked_scores))

    let attn_weights = softmax(masked_scores, 2)
    print("      [ATT] attn_weights sum:", sum(attn_weights))

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("      [ATT] attn_output sum:", sum(attn_output))

    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    let output = linear(attn_reshaped, W_o)
    print("      [ATT] final output sum:", sum(output))

    output
}

fn transformer_layer_simple(
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
    let attn_out = linear(attn_reshaped, W_o)

    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)

    let gate = linear(x_norm2, W_gate)
    let gate_act = silu(gate)
    let up = linear(x_norm2, W_up)
    let intermediate = gate_act * up
    let ffn_out = linear(intermediate, W_down)

    x1 + ffn_out
}

main {
    print("================================================================================")
    print("🔍 Detailed Debug of Layer 11 (where NaN first appears)")
    print("================================================================================")
    print("")

    print("[1] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("")

    print("[2] Creating 46 token input...")
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("    ✓ Tokens:", len(tokens))
    let embeddings = embedding(emb_weight, tokens)
    print("    ✓ Embeddings sum:", sum(embeddings))
    print("")

    print("[3] Processing layers 0-10 (simple pass)...")
    let x = embeddings
    for layer_idx in range_i(11) {
        let attn_norm = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight
        let ffn_norm = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        x = transformer_layer_simple(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        print("    Layer", layer_idx, "output sum:", sum(x))
    }

    print("")
    print("================================================================================")
    print("LAYER 11 DETAILED DEBUG")
    print("================================================================================")
    print("")

    let layer_idx = 11
    print("  Input to Layer 11 sum:", sum(x))

    let attn_norm = model.blk[layer_idx].attn_norm.weight
    let W_q = model.blk[layer_idx].attn_q.weight
    let W_k = model.blk[layer_idx].attn_k.weight
    let W_v = model.blk[layer_idx].attn_v.weight
    let W_o = model.blk[layer_idx].attn_output.weight
    let ffn_norm = model.blk[layer_idx].ffn_norm.weight
    let W_gate = model.blk[layer_idx].ffn_gate.weight
    let W_up = model.blk[layer_idx].ffn_up.weight
    let W_down = model.blk[layer_idx].ffn_down.weight

    print("")
    print("  [1] Attention Norm...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("    x_norm1 sum:", sum(x_norm1))

    print("")
    print("  [2] Computing Q, K, V...")
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)

    print("")
    print("  [3] Attention mechanism...")
    let attn_out = tinyllama_gqa_attention_debug(Q, K, V, W_o)

    print("")
    print("  [4] Residual connection...")
    let x1 = x + attn_out
    print("    x1 (after attn) sum:", sum(x1))

    print("")
    print("  [5] FFN Norm...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("    x_norm2 sum:", sum(x_norm2))

    print("")
    print("  [6] FFN computation...")
    let ffn_out = swiglu_ffn_debug(x_norm2, W_gate, W_up, W_down)

    print("")
    print("  [7] Final residual connection...")
    let output = x1 + ffn_out
    print("    Layer 11 final output sum:", sum(output))

    print("")
    print("================================================================================")
    print("✅ Layer 11 Debug Completed")
    print("================================================================================")
}
