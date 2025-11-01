// Chat Demo with Object-Oriented Method Call Style
// Demonstrates: tokenizer.tokenize(), tokenizer.detokenize(), tokens.append_token()

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    result = x * sigmoid(x)
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
    result = linear(mul_result, W_down)
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
    result = linear(reshaped, W_o)
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
    result = after_attn + ffn_out
}

main {
    print("=== TensorLogic Chat: OO Style ===")
    print("  - Object-oriented method calls")
    print("  - tokenizer.tokenize(), tokenizer.detokenize()")
    print("  - tokens.append_token()")
    print("")

    let EOS_TOKEN = 2

    print("[1/3] Loading model...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("      Loading 2 layers...")
    let L0 = model.blk[0]
    let L1 = model.blk[1]
    print("      ✓ Loaded 2 layers")
    print("")

    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"

    // OO-style tokenization
    let tokens = tokenizer.tokenize(chat_prompt, true)
    print("      User:", user_msg)
    print("")

    print("[3/3] Generating response (max 50 tokens)...")
    print("")
    print("      Assistant: ", "")

    let x = embedding(tok_embd, tokens)

    let K0 = linear(x, L0.attn_k.weight)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = linear(x, L1.attn_k.weight)
    let V1 = linear(x, L1.attn_v.weight)

    let temperature = 0.8

    // Initialize empty token array using OO style
    let all_generated = tokenizer.tokenize("", false)
    let prev_length = 0

    // First token
    let layer_0_out = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)
    let layer_1_out = transformer_layer(layer_0_out, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)
    let normed = rms_norm(layer_1_out, output_norm)
    let current_logits = linear(normed, output)

    let token_id = temperature_sample(current_logits, temperature)

    // OO-style: tokens.append_token()
    all_generated = all_generated.append_token(token_id)

    // OO-style: tokenizer.detokenize()
    let full_text = tokenizer.detokenize(all_generated, false)
    let current_length = string_length(full_text)
    let new_text = string_substring(full_text, prev_length, current_length - prev_length)
    print(new_text, "")
    prev_length = current_length

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
            token_id = temperature_sample(current_logits, temperature)

            // OO-style method chaining
            all_generated = all_generated.append_token(token_id)
            full_text = tokenizer.detokenize(all_generated, false)

            current_length = string_length(full_text)
            new_text = string_substring(full_text, prev_length, current_length - prev_length)
            print(new_text, "")
            prev_length = current_length

            token_count = token_count + 1

            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)
            let new_K0 = linear(new_token_emb, L0.attn_k.weight)
            let new_V0 = linear(new_token_emb, L0.attn_v.weight)
            K0 = concat(K0, new_K0, 0)
            V0 = concat(V0, new_V0, 0)

            let layer_0_out_new = transformer_layer(new_token_emb, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)

            let new_K1 = linear(layer_0_out_new, L1.attn_k.weight)
            let new_V1 = linear(layer_0_out_new, L1.attn_v.weight)
            K1 = concat(K1, new_K1, 0)
            V1 = concat(V1, new_V1, 0)

            let layer_1_out_new = transformer_layer(layer_0_out_new, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)

            let normed_new = rms_norm(layer_1_out_new, output_norm)
            current_logits = linear(normed_new, output)
        }
    }

    print("")
    print("")
    print("✓ Generated", token_count, "tokens with OO-style methods!")
}
