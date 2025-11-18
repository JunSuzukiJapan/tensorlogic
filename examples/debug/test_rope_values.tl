// Test RoPE output values at sequence length 46
// Check for overflow in float16 range

main {
    print("Testing RoPE at sequence length 46...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Generate tokens to reach sequence length 46
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let eos_token_id = 2

    let gen_tokens = prompt_tokens

    // Generate tokens quickly to reach 46
    for step in range_i(10) {
        let emb_weight = get_tensor(model, "token_embd.weight")
        let emb = embedding(emb_weight, gen_tokens)
        let x = emb

        // Process through layers quickly
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

            let x_norm1 = rms_norm(x, attn_norm)
            let Q = linear(x_norm1, W_q)
            let K = linear(x_norm1, W_k)
            let V = linear(x_norm1, W_v)

            let Q_shape = shape(Q)
            let seq_len = Q_shape[0]
            let Q_heads = reshape(Q, [seq_len, 32, 64])
            let K_heads = reshape(K, [seq_len, 4, 64])
            let V_heads = reshape(V, [seq_len, 4, 64])

            let Q_rope = rope(Q_heads)
            let K_rope = rope(K_heads)

            let K_group = reshape(K_rope, [seq_len, 4, 1, 64])
            let V_group = reshape(V_heads, [seq_len, 4, 1, 64])
            let K_broadcast = broadcast_to(K_group, [seq_len, 4, 8, 64])
            let V_broadcast = broadcast_to(V_group, [seq_len, 4, 8, 64])
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
            let gate_act = gate * sigmoid(gate)
            let up = linear(x_norm2, W_up)
            let intermediate = gate_act * up
            let ffn_out = linear(intermediate, W_down)

            x = x1 + ffn_out
        }

        let last_hidden_1d = slice_last(x, 0)
        let last_hidden = reshape(last_hidden_1d, [1, 2048])
        let normed = rms_norm(last_hidden, model.output_norm.weight)
        let logits = linear(normed, model.output.weight)
        let next_token_id = argmax(logits)

        if next_token_id == eos_token_id {
            break
        }
        gen_tokens = append(gen_tokens, next_token_id)
    }

    print("Current sequence length:", len(gen_tokens))
    print("")

    // Now test RoPE values
    let emb_weight = get_tensor(model, "token_embd.weight")
    let emb = embedding(emb_weight, gen_tokens)

    let attn_norm = model.blk[0].attn_norm.weight
    let W_q = model.blk[0].attn_q.weight

    let x_norm1 = rms_norm(emb, attn_norm)
    let Q = linear(x_norm1, W_q)

    print("Q before reshape:")
    print("  sum:", sum(Q))
    print("  max:", max(Q))
    print("  min:", min(Q))

    let Q_shape = shape(Q)
    let seq_len = Q_shape[0]
    let Q_heads = reshape(Q, [seq_len, 32, 64])

    print("")
    print("Q_heads after reshape:")
    print("  sum:", sum(Q_heads))
    print("  max:", max(Q_heads))
    print("  min:", min(Q_heads))

    print("")
    print("Applying RoPE...")
    let Q_rope = rope(Q_heads)

    print("Q_rope after RoPE:")
    print("  sum:", sum(Q_rope))
    print("  max:", max(Q_rope))
    print("  min:", min(Q_rope))
    print("")

    // Check if max/min are within float16 range
    let q_max = max(Q_rope)
    let q_min = min(Q_rope)

    if q_max > 65000.0 {
        print("⚠️  WARNING: Q_rope max", q_max, "exceeds float16 range!")
    }
    if q_min < -65000.0 {
        print("⚠️  WARNING: Q_rope min", q_min, "exceeds float16 range!")
    }

    print("Test complete")
}
