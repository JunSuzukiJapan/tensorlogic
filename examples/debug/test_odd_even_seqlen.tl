// ============================================================================
// Test: Odd vs Even Sequence Length Bug
// ============================================================================
//
// Purpose: Isolate the odd/even sequence length bug causing zero logits
//
// Hypothesis: Odd sequence lengths (47, 49, 51) produce all-zero logits
//             Even sequence lengths (46, 48, 50) work correctly
//
// Test strategy: Run minimal forward pass with:
//   1. seq_len = 46 (even - should work)
//   2. seq_len = 47 (odd - should fail with zeros)
//
// Usage:
//   timeout 60 ./target/release/tl run examples/debug/test_odd_even_seqlen.tl
//
// ============================================================================

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
    print("================================================================================")
    print("🔍 Test: Odd vs Even Sequence Length Bug")
    print("================================================================================")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("[1] Loaded model weights")
    print("")

    // Create test prompt
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    print("[2] Tokenized prompt:", len(tokens), "tokens")
    print("")

    // ========================================================================
    // TEST 1: Forward pass with EVEN sequence length (46 tokens - should work)
    // ========================================================================
    print("────────────────────────────────────────────────────────────────────────────────")
    print("TEST 1: Even sequence length (46 tokens) - Expected: Valid logits")
    print("────────────────────────────────────────────────────────────────────────────────")
    print("")

    let emb_46 = embedding(emb_weight, tokens)
    print("  Embedding shape:", shape(emb_46))

    let x_46 = emb_46
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

        x_46 = transformer_layer(x_46, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
    }

    let last_hidden_46_1d = slice_last(x_46, 0)
    let last_hidden_46 = reshape(last_hidden_46_1d, [1, 2048])
    let normed_46 = rms_norm(last_hidden_46, output_norm_weight)
    let logits_46 = linear(normed_46, output_weight)

    print("  Logits shape:", shape(logits_46))
    print("  Logits min:", min(logits_46))
    print("  Logits max:", max(logits_46))
    print("")

    // ========================================================================
    // TEST 2: Forward pass with ODD sequence length (47 tokens - should fail)
    // ========================================================================
    print("────────────────────────────────────────────────────────────────────────────────")
    print("TEST 2: Odd sequence length (47 tokens) - Expected: ALL ZERO logits (BUG!)")
    print("────────────────────────────────────────────────────────────────────────────────")
    print("")

    // Add one dummy token to make length odd
    let tokens_47 = append(tokens, 100)  // Token ID 100 is arbitrary
    print("  Token sequence length:", len(tokens_47))

    let emb_47 = embedding(emb_weight, tokens_47)
    print("  Embedding shape:", shape(emb_47))

    let x_47 = emb_47
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

        x_47 = transformer_layer(x_47, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
    }

    let last_hidden_47_1d = slice_last(x_47, 0)
    let last_hidden_47 = reshape(last_hidden_47_1d, [1, 2048])
    let normed_47 = rms_norm(last_hidden_47, output_norm_weight)
    let logits_47 = linear(normed_47, output_weight)

    print("  Logits shape:", shape(logits_47))
    print("  Logits min:", min(logits_47))
    print("  Logits max:", max(logits_47))
    print("")

    // ========================================================================
    // RESULTS
    // ========================================================================
    print("================================================================================")
    print("📊 Results")
    print("================================================================================")
    print("")

    let min_46 = min(logits_46)
    let max_46 = max(logits_46)
    let min_47 = min(logits_47)
    let max_47 = max(logits_47)

    print("  Even length (46): min =", min_46, ", max =", max_46)
    print("  Odd length  (47): min =", min_47, ", max =", max_47)
    print("")

    if max_47 == 0.0 && min_47 == 0.0 {
        print("❌ BUG CONFIRMED: Odd sequence length produces ALL ZERO logits!")
    } else {
        print("✅ No bug detected - odd sequence length works correctly")
    }
    print("")
}
