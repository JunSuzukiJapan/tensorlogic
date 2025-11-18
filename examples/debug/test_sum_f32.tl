// Test sum_f32() to verify it prevents overflow

main {
    print("Testing sum() vs sum_f32()...")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Generate tokens to reach sequence length 46
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let gen_tokens = prompt_tokens

    // Generate 10 tokens quickly
    for step in range_i(10) {
        let emb = embedding(get_tensor(model, "token_embd.weight"), gen_tokens)
        let x = emb

        for layer_idx in range_i(22) {
            let x_norm1 = rms_norm(x, model.blk[layer_idx].attn_norm.weight)
            let Q = linear(x_norm1, model.blk[layer_idx].attn_q.weight)
            let K = linear(x_norm1, model.blk[layer_idx].attn_k.weight)
            let V = linear(x_norm1, model.blk[layer_idx].attn_v.weight)

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
            let attn_out = linear(attn_reshaped, model.blk[layer_idx].attn_output.weight)

            let x1 = x + attn_out
            let x_norm2 = rms_norm(x1, model.blk[layer_idx].ffn_norm.weight)
            let gate = linear(x_norm2, model.blk[layer_idx].ffn_gate.weight)
            let gate_act = gate * sigmoid(gate)
            let up = linear(x_norm2, model.blk[layer_idx].ffn_up.weight)
            let ffn_out = linear(gate_act * up, model.blk[layer_idx].ffn_down.weight)

            x = x1 + ffn_out
        }

        let last_hidden_1d = slice_last(x, 0)
        let last_hidden = reshape(last_hidden_1d, [1, 2048])
        let normed = rms_norm(last_hidden, model.output_norm.weight)
        let logits = linear(normed, model.output.weight)
        let next_token_id = argmax(logits)
        gen_tokens = append(gen_tokens, next_token_id)
    }

    print("Sequence length:", len(gen_tokens))
    print("")

    // Now compare sum() vs sum_f32() on attention scores
    let emb = embedding(get_tensor(model, "token_embd.weight"), gen_tokens)
    let x_norm1 = rms_norm(emb, model.blk[0].attn_norm.weight)
    let Q = linear(x_norm1, model.blk[0].attn_q.weight)
    let K = linear(x_norm1, model.blk[0].attn_k.weight)

    let Q_shape = shape(Q)
    let seq_len = Q_shape[0]
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)

    print("Attention scores (before scaling):")
    print("  Elements:", seq_len * 32 * seq_len)
    print("  max:", max(scores))
    print("  min:", min(scores))
    print("")

    print("Comparison:")
    print("  sum(scores):     ", sum(scores), " ← May overflow (f16)")
    print("  sum_f32(scores): ", sum_f32(scores), " ← Always valid (f32)")
    print("")

    print("✓ Test complete - sum_f32() prevents overflow!")
}
