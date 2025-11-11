// Progressive layer testing: 2, 3, 5, 10, 22 layers
// Compares with Candle reference values

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    // Reshape to heads
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    // Apply RoPE to Q and K
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA: Expand K, V from 4 to 32 heads
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125

    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Flatten and project
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
    // Attention
    let normed = rms_norm(x, attn_norm)
    let Q = linear(normed, W_q)
    let K = linear(normed, W_k)
    let V = linear(normed, W_v)
    let attn_output = gqa_attention(Q, K, V, W_o)
    let x_after_attn = x + attn_output

    // FFN
    let normed2 = rms_norm(x_after_attn, ffn_norm)
    let up = linear(normed2, W_up)
    let gate = linear(normed2, W_gate)
    let gate_act = silu(gate)
    let ffn_hidden = up * gate_act
    let ffn_output = linear(ffn_hidden, W_down)

    x_after_attn + ffn_output
}

main {
    print("=== Progressive Layer Testing ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Use the same prompt as Candle reference (38 tokens)
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("[0] Token count:", len(tokens))
    print("")

    // Get embeddings
    let emb = embedding(emb_weight, tokens)
    let emb_sum = sum(emb)
    print("Embedding sum:", emb_sum)
    print("  Reference (Candle): 3.66015625")
    print("")

    // Test with 2 layers
    print("=== Testing with 2 layers ===")
    let x = emb
    for layer_idx in range_i(2) {
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

        let layer_sum = sum(x)
        print("  Layer", layer_idx, "output sum:", layer_sum)
        if layer_idx == 0 {
            print("    Reference (Candle): 3.830357")
        }
        if layer_idx == 1 {
            print("    Reference (Candle): -5.474568")
        }
    }
    print("")

    // Test with 3 layers
    print("=== Testing with 3 layers ===")
    x = emb
    for layer_idx in range_i(3) {
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

        if layer_idx == 2 {
            let layer_sum = sum(x)
            print("  Layer 2 output sum:", layer_sum)
            print("    Reference (Candle): -227.37646")
        }
    }
    print("")

    // Test with 5 layers
    print("=== Testing with 5 layers ===")
    x = emb
    for layer_idx in range_i(5) {
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

        if layer_idx == 4 {
            let layer_sum = sum(x)
            print("  Layer 4 output sum:", layer_sum)
            print("    Reference (Candle): -228.67639")
        }
    }
    print("")

    // Test with 10 layers
    print("=== Testing with 10 layers ===")
    x = emb
    for layer_idx in range_i(10) {
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

        if layer_idx == 9 {
            let layer_sum = sum(x)
            print("  Layer 9 output sum:", layer_sum)
            print("    Reference (Candle): -562.3587")
        }
    }
    print("")

    // Test with all 22 layers
    print("=== Testing with all 22 layers ===")
    x = emb
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

        x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)

        if layer_idx == 21 {
            let layer_sum = sum(x)
            print("  Layer 21 output sum:", layer_sum)
            print("    Reference (Candle): 30.48526")
        }
    }

    // Final norm and logits
    let normed = rms_norm(x, output_norm_weight)
    let norm_sum = sum(normed)
    print("  Final norm sum:", norm_sum)
    print("    Reference (Candle): -286.91577")

    let logits = linear(normed, output_weight)
    let last_hidden_1d = slice_last(x, 0)
    let last_hidden = reshape(last_hidden_1d, [1, 2048])
    let last_normed = rms_norm(last_hidden, output_norm_weight)
    let last_logits = linear(last_normed, output_weight)
    let logits_sum = sum(last_logits)
    print("  Last token logits sum:", logits_sum)
    print("    Reference (Candle): -38186.4")
    print("")

    print("=== Test Complete ===")
}
