// Candle参照値との比較テスト
// Prompt: "Hello! How are you?" (39 tokens with BOS in Candle, 38 without BOS in TL)

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
    print("🔍 Candle参照値との比較テスト")
    print("================================================================================")
    print("")

    print("[1] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("")

    print("[2] Tokenizing (same prompt as Candle, WITH BOS token)...")
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello! How are you?</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, true)  // true = add BOS token
    print("    ✓ Tokens:", len(tokens))
    print("    Token IDs:", tokens)
    print("")

    print("[3] Embedding...")
    let embeddings = embedding(emb_weight, tokens)
    let emb_sum = sum(embeddings)
    print("    TensorLogic embedding sum:", emb_sum)
    print("    Candle reference:          3.6829965")
    print("")

    print("[4] Processing all 22 layers...")
    print("────────────────────────────────────────────────────────────────────────────")
    print("Layer │ TensorLogic Sum │ Candle Reference │ Diff      │ Status")
    print("────────────────────────────────────────────────────────────────────────────")

    let candle_refs = [
        3.830357,    // layer 0
        -5.474568,   // layer 1
        -227.37646,  // layer 2
        -232.49507,  // layer 3
        -228.67639,  // layer 4
        -230.96385,  // layer 5
        -248.82343,  // layer 6
        -539.43805,  // layer 7
        -537.8561,   // layer 8
        -562.3587,   // layer 9
        -575.6366,   // layer 10
        -632.66644,  // layer 11
        -646.21375,  // layer 12
        -692.21484,  // layer 13
        -722.07874,  // layer 14
        -759.60657,  // layer 15
        -764.71735,  // layer 16
        -774.4859,   // layer 17
        -691.85187,  // layer 18
        -464.73773,  // layer 19
        -555.889,    // layer 20
        30.48526     // layer 21
    ]

    let x = embeddings
    for layer_idx in range_i(22) {
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

        let tl_sum = sum(x)
        let candle_ref = candle_refs[layer_idx]
        let diff = tl_sum - candle_ref

        // Status判定: 差が10%以内なら○、それ以外は×
        let abs_candle = candle_ref
        if candle_ref < 0.0 {
            abs_candle = -candle_ref
        }
        let abs_diff = diff
        if diff < 0.0 {
            abs_diff = -diff
        }
        let rel_error = abs_diff / abs_candle

        if layer_idx < 10 {
            print(" ", layer_idx, "   │")
        } else {
            print("", layer_idx, "   │")
        }
        print("    ", tl_sum, " │", candle_ref, " │", diff)

        if rel_error < 0.1 {
            print("      │ ✓")
        } else {
            print("      │ ✗ MISMATCH")
        }
    }

    print("────────────────────────────────────────────────────────────────────────────")
    print("")

    print("[5] Final norm and logits...")
    let output_norm = model.output_norm.weight
    let x_final = rms_norm(x, output_norm)
    let final_norm_sum = sum(x_final)
    print("    Final norm sum (TL):      ", final_norm_sum)
    print("    Candle reference:          -286.91577")
    print("")

    let lm_head = model.output.weight
    let logits = linear(x_final, lm_head)
    print("    Logits shape:", shape(logits))
    let logits_sum = sum(logits)
    print("    Logits sum (TL):           ", logits_sum)
    print("    Candle reference:          -38186.4")
    print("")

    print("================================================================================")
    print("✅ Comparison Complete")
    print("================================================================================")
}
