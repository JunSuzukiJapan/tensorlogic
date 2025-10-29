// Chat Demo with Incremental Decoding (fixes garbled characters)
// Uses detokenize_incremental to properly handle multi-byte UTF-8 tokens

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    result := x * sigmoid(x)
}

fn swiglu_ffn(
    x: float32[?, ?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?]
) -> float32[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    result := linear(mul_result, W_down)
}

fn attention_with_cache(
    Q: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?],
    W_o: float32[?, ?]
) -> float32[?, ?] {
    let Q_shape = shape(Q)
    let seq_len_f = Q_shape[0]
    let K_shape = shape(K_cache)
    let cache_len_f = K_shape[0]

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let Q_rope = rope(Q_heads)

    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let K_rope = rope(K_heads)
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_exp = reshape(K_rope, [cache_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    let reshaped = reshape(out, [seq_len_f, 2048.0])
    result := linear(reshaped, W_o)
}

fn transformer_layer(
    x: float32[?, ?],
    W_attn_norm: float32[?],
    W_q: float32[?, ?],
    W_k: float32[?, ?],
    W_v: float32[?, ?],
    W_o: float32[?, ?],
    W_ffn_norm: float32[?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?]
) -> float32[?, ?] {
    let normed = rms_norm(x, W_attn_norm)
    let Q = linear(normed, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out
    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result := after_attn + ffn_out
}

main {
    print("=== TensorLogic Chat: Incremental Decoding ===")
    print("  - Fixed character encoding issues")
    print("  - Uses detokenize_incremental()")
    print("")

    let EOS_TOKEN = 2

    // Load model
    print("[1/3] Loading model...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("      Loading 2 layers (demo)...")
    let L0 = model.blk[0]
    let layer_0_attn_norm = L0.attn_norm.weight
    let layer_0_q = L0.attn_q.weight
    let layer_0_k = L0.attn_k.weight
    let layer_0_v = L0.attn_v.weight
    let layer_0_o = L0.attn_output.weight
    let layer_0_ffn_norm = L0.ffn_norm.weight
    let layer_0_gate = L0.ffn_gate.weight
    let layer_0_up = L0.ffn_up.weight
    let layer_0_down = L0.ffn_down.weight

    let L1 = model.blk[1]
    let layer_1_attn_norm = L1.attn_norm.weight
    let layer_1_q = L1.attn_q.weight
    let layer_1_k = L1.attn_k.weight
    let layer_1_v = L1.attn_v.weight
    let layer_1_o = L1.attn_output.weight
    let layer_1_ffn_norm = L1.ffn_norm.weight
    let layer_1_gate = L1.ffn_gate.weight
    let layer_1_up = L1.ffn_up.weight
    let layer_1_down = L1.ffn_down.weight
    print("      ✓ Loaded 2 layers")
    print("")

    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("      User:", user_msg)
    print("")

    print("[3/3] Generating response...")
    print("")
    print("      Assistant: ", "")

    let x = embedding(tok_embd, tokens)

    let K0 = linear(x, layer_0_k)
    let V0 = linear(x, layer_0_v)
    let K1 = linear(x, layer_1_k)
    let V1 = linear(x, layer_1_v)

    let temperature = 0.8

    // Initialize generated token IDs array (using tokenize with empty string)
    let generated_tokens = tokenize(tokenizer, "", false)

    // Generate first token
    let layer_0_out = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o, layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)
    let layer_1_out = transformer_layer(layer_0_out, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o, layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)
    let normed = rms_norm(layer_1_out, output_norm)
    let current_logits = linear(normed, output)

    let token_id = temperature_sample(current_logits, temperature)
    generated_tokens = append_token(generated_tokens, token_id)

    // Decode and print
    let text = detokenize_incremental(tokenizer, generated_tokens)
    print(text, "")

    // Continue generation loop
    let token_count = 1
    let max_tokens = 50
    let continue_generation = true

    while continue_generation {
        if token_count >= max_tokens {
            continue_generation = false
        }

        if token_id == EOS_TOKEN {
            continue_generation = false
        }

        if continue_generation {
            // Sample token
            token_id = temperature_sample(current_logits, temperature)
            generated_tokens = append_token(generated_tokens, token_id)

            // Decode ALL tokens and print
            text = detokenize_incremental(tokenizer, generated_tokens)
            print(text, "")

            token_count = token_count + 1

            // Continue transformer forward pass...
            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)
            let new_K0 = linear(new_token_emb, layer_0_k)
            let new_V0 = linear(new_token_emb, layer_0_v)
            K0 = concat(K0, new_K0, 0)
            V0 = concat(V0, new_V0, 0)

            let layer_0_out_new = transformer_layer(new_token_emb, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o, layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down, K0, V0)

            let new_K1 = linear(layer_0_out_new, layer_1_k)
            let new_V1 = linear(layer_0_out_new, layer_1_v)
            K1 = concat(K1, new_K1, 0)
            V1 = concat(V1, new_V1, 0)

            let layer_1_out_new = transformer_layer(layer_0_out_new, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o, layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down, K1, V1)

            let normed_new = rms_norm(layer_1_out_new, output_norm)
            current_logits = linear(normed_new, output)
        }
    }

    print("")
    print("")
    print("✓ Generated", token_count, "tokens")
}
