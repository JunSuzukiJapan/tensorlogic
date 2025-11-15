// Debug Layer 0 NaN issue at sequence length 46
// Track intermediate values through each operation in Layer 0

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("🔍 Layer 0 NaN Debug - Detailed Operation Tracking")
    print("================================================================================")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("✓ Model loaded")

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("✓ Tokenizer loaded")
    print("")

    // Prepare prompt (same as before)
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\\n" + system_prompt + "</s>\\n<|user|>\\n" + user_message + "</s>\\n<|assistant|>\\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let eos_token_id = 2
    let temperature = 0.7

    let gen_tokens = prompt_tokens
    let generated_count = 0

    // Generate 11 tokens to hit the bug
    for token_step in range_i(11) {
        if token_step < 10 {
            // Fast path for first 10 tokens
            let current_emb = embedding(get_tensor(model, "token_embd.weight"), gen_tokens)
            let x = current_emb

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

                // Inline transformer layer
                let x_norm1 = rms_norm(x, attn_norm)
                let Q = linear(x_norm1, W_q)
                let K = linear(x_norm1, W_k)
                let V = linear(x_norm1, W_v)

                // GQA attention
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
                let gate_act = silu(gate)
                let up = linear(x_norm2, W_up)
                let intermediate = gate_act * up
                let ffn_out = linear(intermediate, W_down)

                x = x1 + ffn_out
            }

            let last_hidden_1d = slice_last(x, 0)
            let last_hidden = reshape(last_hidden_1d, [1, 2048])
            let normed = rms_norm(last_hidden, model.output_norm.weight)
            let logits = linear(normed, model.output.weight)
            let next_token_id = temperature_sample(logits, temperature)

            if next_token_id == eos_token_id {
                break
            }
            gen_tokens = append(gen_tokens, next_token_id)
            generated_count = generated_count + 1
        } else {
            // Token 11 - detailed debug
            print("================================================================================")
            print("Token 11 - DETAILED LAYER 0 DEBUG")
            print("================================================================================")
            print("")

            let emb_weight = get_tensor(model, "token_embd.weight")
            let current_emb = embedding(emb_weight, gen_tokens)
            print("1. Embedding sum:", sum(current_emb))

            let x = current_emb
            print("2. Input to Layer 0 sum:", sum(x))

            // Layer 0 weights
            let attn_norm = model.blk[0].attn_norm.weight
            let W_q = model.blk[0].attn_q.weight
            let W_k = model.blk[0].attn_k.weight
            let W_v = model.blk[0].attn_v.weight
            let W_o = model.blk[0].attn_output.weight

            // Step by step
            let x_norm1 = rms_norm(x, attn_norm)
            print("3. After RMS norm sum:", sum(x_norm1))

            let Q = linear(x_norm1, W_q)
            print("4. After Q projection sum:", sum(Q))

            let K = linear(x_norm1, W_k)
            print("5. After K projection sum:", sum(K))

            let V = linear(x_norm1, W_v)
            print("6. After V projection sum:", sum(V))

            // Reshape
            let Q_shape = shape(Q)
            let seq_len = Q_shape[0]
            print("7. Sequence length:", seq_len)

            let Q_heads = reshape(Q, [seq_len, 32, 64])
            print("8. Q_heads sum:", sum(Q_heads))

            let K_heads = reshape(K, [seq_len, 4, 64])
            print("9. K_heads sum:", sum(K_heads))

            let V_heads = reshape(V, [seq_len, 4, 64])
            print("10. V_heads sum:", sum(V_heads))

            // RoPE
            let Q_rope = rope(Q_heads)
            print("11. After RoPE on Q sum:", sum(Q_rope))

            let K_rope = rope(K_heads)
            print("12. After RoPE on K sum:", sum(K_rope))

            // GQA expansion
            let K_group = reshape(K_rope, [seq_len, 4, 1, 64])
            let V_group = reshape(V_heads, [seq_len, 4, 1, 64])
            let K_broadcast = broadcast_to(K_group, [seq_len, 4, 8, 64])
            let V_broadcast = broadcast_to(V_group, [seq_len, 4, 8, 64])
            let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
            let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])
            print("13. K_expanded sum:", sum(K_expanded))
            print("14. V_expanded sum:", sum(V_expanded))

            // Attention scores
            let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
            print("15. Attention scores sum:", sum(scores))

            let scaled_scores = scores * 0.125
            print("16. Scaled scores sum:", sum(scaled_scores))

            // Mask
            let seq_len_int = seq_len as int
            let mask_2d = causal_mask(seq_len_int)
            let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
            let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
            let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)
            print("17. Masked scores sum:", sum(masked_scores))

            // Softmax
            let attn_weights = softmax(masked_scores, 2)
            print("18. Attention weights sum:", sum(attn_weights))

            // Attention output
            let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
            print("19. Attention output sum:", sum(attn_output))

            let attn_reshaped = reshape(attn_output, [seq_len, 2048])
            print("20. Attention reshaped sum:", sum(attn_reshaped))

            let attn_out = linear(attn_reshaped, W_o)
            print("21. After output projection sum:", sum(attn_out))

            print("")
            print("❌ Bug should be identified above")
            break
        }
    }

    print("")
    print("Debug complete -", generated_count, "tokens generated before detailed debug")
}
