// Test long sequence generation to identify where it stops

main {
    print("Testing long sequence generation...")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    print("Initial prompt length:", len(prompt_tokens))

    let gen_tokens = prompt_tokens
    let eos_token_id = 2
    let max_new_tokens = 20

    print("Generating up to", max_new_tokens, "new tokens...")
    print("")

    for token_step in range_i(max_new_tokens) {
        let current_len = len(gen_tokens)
        print("Token", token_step + 1, "| Seq len:", current_len)

        let emb = embedding(get_tensor(model, "token_embd.weight"), gen_tokens)
        let x = emb

        // Process all 22 layers
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
        print("  → Generated token:", next_token_id)

        if next_token_id == eos_token_id {
            print("")
            print("EOS reached after", token_step + 1, "tokens")
            break
        }

        gen_tokens = append(gen_tokens, next_token_id)
    }

    print("")
    print("Final sequence length:", len(gen_tokens))
    print("✓ Test complete")
}
