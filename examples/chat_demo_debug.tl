// Chat demo with extensive debugging

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
    print("=== Chat Demo with Debug Output ===")
    print("")

    print("[1] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("    ✓ Loaded")

    print("[2] Loading weights...")
    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("    ✓ Core weights loaded")

    print("[3] Preparing prompt...")
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let prompt_len = len(prompt_tokens)
    print("    ✓ Prompt tokens:", prompt_len)

    print("[4] Starting generation...")
    print("")

    let eos_token_id = 2
    let temperature = 0.8
    let gen_tokens = prompt_tokens
    let generated_count = 0

    // Generate only 3 tokens for debugging
    for token_step in range_i(3) {
        print("  [Token", token_step, "] Starting iteration...")
        print("  [Token", token_step, "] Current sequence length:", len(gen_tokens))

        print("  [Token", token_step, "] Computing embeddings...")
        let current_emb = embedding(emb_weight, gen_tokens)
        print("  [Token", token_step, "] Embeddings done")

        print("  [Token", token_step, "] Processing 22 layers...")
        let x = current_emb
        for layer_idx in range_i(22) {
            if layer_idx % 5 == 0 {
                print("    Layer", layer_idx)
            }
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
        }
        print("  [Token", token_step, "] All layers completed")

        print("  [Token", token_step, "] Extracting last token...")
        let last_hidden_1d = slice_last(x, 0)
        let last_hidden = reshape(last_hidden_1d, [1, 2048])
        print("  [Token", token_step, "] Last token extracted")

        print("  [Token", token_step, "] Computing logits...")
        let normed = rms_norm(last_hidden, output_norm_weight)
        let logits = linear(normed, output_weight)
        print("  [Token", token_step, "] Logits computed")

        print("  [Token", token_step, "] Sampling...")
        let next_token_id = temperature_sample(logits, temperature)
        print("  [Token", token_step, "] Sampled token:", next_token_id)

        if next_token_id == eos_token_id {
            print("  [Token", token_step, "] EOS detected, stopping")
            break
        }

        print("  [Token", token_step, "] Appending token...")
        gen_tokens = append(gen_tokens, next_token_id)
        generated_count = generated_count + 1
        print("  [Token", token_step, "] Iteration complete")
        print("")
    }

    print("")
    print("=== Generation completed ===")
    print("Generated", generated_count, "tokens")
    print("Final sequence length:", len(gen_tokens))
    print("")

    let final_text = detokenize(tokenizer, gen_tokens, false)
    print("Result:")
    print(final_text)
}
