// Debug script to investigate zero logits issue at token 11
// Tracks intermediate values through the forward pass

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
    print("🔍 Zero Logits Debug - Tracking Values Through Forward Pass")
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

    // Load weights
    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Prepare prompt
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let prompt_len = len(prompt_tokens)
    print("Prompt tokens:", prompt_len)
    print("")

    let eos_token_id = 2
    let temperature = 0.7

    let gen_tokens = prompt_tokens
    let generated_count = 0

    print("================================================================================")
    print("Starting token generation with detailed tracking...")
    print("================================================================================")
    print("")

    // Generate 12 tokens to see the transition from working to zero logits
    for token_step in range_i(12) {
        print("--------------------------------------------------------------------------------")
        print("Token", token_step + 1, "- Sequence length:", len(gen_tokens))
        print("--------------------------------------------------------------------------------")

        // Get embeddings
        let current_emb = embedding(emb_weight, gen_tokens)
        let emb_sum = sum(current_emb)
        let emb_shape = shape(current_emb)
        print("  Embedding shape:", emb_shape[0], "x", emb_shape[1])
        print("  Embedding sum:", emb_sum)

        // Process through layers with detailed tracking
        let x = current_emb

        // Track first layer in detail
        print("  Layer 0 processing:")
        let attn_norm_0 = model.blk[0].attn_norm.weight
        let W_q_0 = model.blk[0].attn_q.weight
        let W_k_0 = model.blk[0].attn_k.weight
        let W_v_0 = model.blk[0].attn_v.weight
        let W_o_0 = model.blk[0].attn_output.weight
        let ffn_norm_0 = model.blk[0].ffn_norm.weight
        let W_gate_0 = model.blk[0].ffn_gate.weight
        let W_up_0 = model.blk[0].ffn_up.weight
        let W_down_0 = model.blk[0].ffn_down.weight

        let x_before_layer0 = sum(x)
        print("    Before layer 0 sum:", x_before_layer0)

        x = transformer_layer(x, attn_norm_0, W_q_0, W_k_0, W_v_0, W_o_0, ffn_norm_0, W_gate_0, W_up_0, W_down_0)

        let x_after_layer0 = sum(x)
        print("    After layer 0 sum:", x_after_layer0)

        // Process remaining layers
        for layer_idx in range_i(21) {
            let actual_layer = layer_idx + 1
            let attn_norm = model.blk[actual_layer].attn_norm.weight
            let W_q = model.blk[actual_layer].attn_q.weight
            let W_k = model.blk[actual_layer].attn_k.weight
            let W_v = model.blk[actual_layer].attn_v.weight
            let W_o = model.blk[actual_layer].attn_output.weight
            let ffn_norm = model.blk[actual_layer].ffn_norm.weight
            let W_gate = model.blk[actual_layer].ffn_gate.weight
            let W_up = model.blk[actual_layer].ffn_up.weight
            let W_down = model.blk[actual_layer].ffn_down.weight

            x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        }

        let x_after_all_layers = sum(x)
        print("  After all 22 layers sum:", x_after_all_layers)

        // Extract last token
        let last_hidden_1d = slice_last(x, 0)
        let last_hidden = reshape(last_hidden_1d, [1, 2048])
        let last_hidden_sum = sum(last_hidden)
        print("  Last hidden state sum:", last_hidden_sum)

        // Final norm
        let normed = rms_norm(last_hidden, output_norm_weight)
        let normed_sum = sum(normed)
        print("  After output norm sum:", normed_sum)

        // Logits
        let logits = linear(normed, output_weight)
        let logits_sum = sum(logits)
        let logits_max = max(logits)
        let logits_min = min(logits)
        print("  Logits sum:", logits_sum)
        print("  Logits max:", logits_max)
        print("  Logits min:", logits_min)

        // Sample
        let next_token_id = temperature_sample(logits, temperature)
        print("  Sampled token ID:", next_token_id)
        print("")

        if next_token_id == eos_token_id {
            print("EOS reached after", generated_count, "tokens")
            break
        }

        gen_tokens = append(gen_tokens, next_token_id)
        generated_count = generated_count + 1
    }

    print("================================================================================")
    print("Debug session complete -", generated_count, "tokens generated")
    print("================================================================================")
}
