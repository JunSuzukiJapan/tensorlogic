// Test 3 layers with 46 tokens - Debug version with intermediate values

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

fn transformer_layer_debug(
    x: float16[?, ?],
    attn_norm: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    ffn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    layer_idx: int
) -> float16[?, ?] {
    print("  [Layer", layer_idx, "] Input sum:", sum(x))

    let x_norm1 = rms_norm(x, attn_norm)
    print("  [Layer", layer_idx, "] x_norm1 sum:", sum(x_norm1))

    let Q = linear(x_norm1, W_q)
    print("  [Layer", layer_idx, "] Q sum:", sum(Q))

    let K = linear(x_norm1, W_k)
    print("  [Layer", layer_idx, "] K sum:", sum(K))

    let V = linear(x_norm1, W_v)
    print("  [Layer", layer_idx, "] V sum:", sum(V))

    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    print("  [Layer", layer_idx, "] attn_out sum:", sum(attn_out))

    let x1 = x + attn_out
    print("  [Layer", layer_idx, "] x1 (after attn) sum:", sum(x1))

    let x_norm2 = rms_norm(x1, ffn_norm)
    print("  [Layer", layer_idx, "] x_norm2 sum:", sum(x_norm2))

    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    print("  [Layer", layer_idx, "] ffn_out sum:", sum(ffn_out))

    let output = x1 + ffn_out
    print("  [Layer", layer_idx, "] Output sum:", sum(output))
    print("")

    output
}

main {
    print("================================================================================")
    print("🔍 3-Layer Test with Detailed Intermediate Values")
    print("================================================================================")
    print("")

    print("[1] Loading model and tokenizer...")
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
    print("")

    print("[3] Computing embeddings...")
    let embeddings = embedding(emb_weight, tokens)
    print("    Embeddings shape:", shape(embeddings))
    print("    Embeddings sum:", sum(embeddings))
    print("")

    print("================================================================================")
    print("Processing 3 Transformer Layers")
    print("================================================================================")
    print("")

    let x = embeddings
    for layer_idx in range_i(3) {
        print("──────────────────────────────────────────────────────────────────────────")
        print("LAYER", layer_idx)
        print("──────────────────────────────────────────────────────────────────────────")

        let attn_norm = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight
        let ffn_norm = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        x = transformer_layer_debug(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down, layer_idx)
    }

    print("================================================================================")
    print("✅ 3-Layer Test Completed")
    print("================================================================================")
    print("")
    print("Final output shape:", shape(x))
    print("Final output sum:", sum(x))
}
