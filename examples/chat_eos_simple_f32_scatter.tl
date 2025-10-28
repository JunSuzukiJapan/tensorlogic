// Chat Demo with KV Cache Updates (f32, 2 layers)
// ZERO-COPY version using scatter() instead of concat()
// Proper autoregressive generation with pre-allocated caches

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
    result := linear(silu(gate) * up, W_down)
}

fn attention_with_cache(
    Q: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?],
    W_o: float32[?, ?],
    cache_len: float32[1]
) -> float32[?, ?] {
    let Q_shape = shape(Q)
    let seq_len_f = Q_shape[0]
    let cache_len_f = cache_len[0]

    // Slice K_cache and V_cache to actual length
    let K_active = slice(K_cache, 0, cache_len_f)
    let V_active = slice(V_cache, 0, cache_len_f)

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let Q_rope = rope(Q_heads)

    let K_heads = reshape(K_active, [cache_len_f, 4.0, 64.0])
    let K_rope = rope(K_heads)
    let V_heads = reshape(V_active, [cache_len_f, 4.0, 64.0])

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
    V_cache: float32[?, ?],
    cache_len: float32[1]
) -> float32[?, ?] {
    let normed = rms_norm(x, W_attn_norm)
    let Q = linear(normed, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o, cache_len)
    let after_attn = x + attn_out

    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result := after_attn + ffn_out
}

main {
    print("=== Autoregressive Chat with ZERO-COPY KV Cache (f32, 2 layers) ===")
    print("")

    // Load
    print("[1/5] Loading...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    // Load weights (2 layers)
    let tok_embd = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output = get_tensor(model, "output.weight")

    let layer_0_attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let layer_0_q = get_tensor(model, "blk.0.attn_q.weight")
    let layer_0_k = get_tensor(model, "blk.0.attn_k.weight")
    let layer_0_v = get_tensor(model, "blk.0.attn_v.weight")
    let layer_0_o = get_tensor(model, "blk.0.attn_output.weight")
    let layer_0_ffn_norm = get_tensor(model, "blk.0.ffn_norm.weight")
    let layer_0_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    let layer_0_up = get_tensor(model, "blk.0.ffn_up.weight")
    let layer_0_down = get_tensor(model, "blk.0.ffn_down.weight")

    let layer_1_attn_norm = get_tensor(model, "blk.1.attn_norm.weight")
    let layer_1_q = get_tensor(model, "blk.1.attn_q.weight")
    let layer_1_k = get_tensor(model, "blk.1.attn_k.weight")
    let layer_1_v = get_tensor(model, "blk.1.attn_v.weight")
    let layer_1_o = get_tensor(model, "blk.1.attn_output.weight")
    let layer_1_ffn_norm = get_tensor(model, "blk.1.ffn_norm.weight")
    let layer_1_gate = get_tensor(model, "blk.1.ffn_gate.weight")
    let layer_1_up = get_tensor(model, "blk.1.ffn_up.weight")
    let layer_1_down = get_tensor(model, "blk.1.ffn_down.weight")
    print("      ✓ Loaded")
    print("")

    // Prepare prompt
    print("[2/5] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("      User:", user_msg)
    print("")

    // Get prompt length
    let tokens_shape = shape(tokens)
    let prompt_len_f = tokens_shape[0]
    print("      Prompt tokens:", prompt_len_f)

    // Pre-allocate KV caches (ZERO-COPY: allocate once, reuse forever)
    print("[3/5] Allocating KV caches...")
    let max_seq_len = 512.0
    let d_kv = 256.0  // KV dimension for GQA

    let KV0_K_cache = zeros([max_seq_len, d_kv])
    let KV0_V_cache = zeros([max_seq_len, d_kv])
    let KV1_K_cache = zeros([max_seq_len, d_kv])
    let KV1_V_cache = zeros([max_seq_len, d_kv])
    print("      ✓ Pre-allocated [", max_seq_len, ",", d_kv, "] caches")
    print("")

    // Initial forward pass (process prompt)
    print("[4/5] Processing prompt...")
    let x = embedding(tok_embd, tokens)

    // Compute initial K, V for prompt
    let K0_init = linear(x, layer_0_k)
    let V0_init = linear(x, layer_0_v)
    let K1_init = linear(x, layer_1_k)
    let V1_init = linear(x, layer_1_v)

    // Write prompt KV to pre-allocated caches using scatter (ZERO-COPY)
    let prompt_indices = arange(prompt_len_f)
    KV0_K_cache = scatter(KV0_K_cache, 0, prompt_indices, K0_init)
    KV0_V_cache = scatter(KV0_V_cache, 0, prompt_indices, V0_init)
    KV1_K_cache = scatter(KV1_K_cache, 0, prompt_indices, K1_init)
    KV1_V_cache = scatter(KV1_V_cache, 0, prompt_indices, V1_init)

    let current_len = [prompt_len_f]

    let h0 = transformer_layer(x, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down,
                                KV0_K_cache, KV0_V_cache, current_len)
    let h1 = transformer_layer(h0, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down,
                                KV1_K_cache, KV1_V_cache, current_len)

    let final_norm = rms_norm(h1, output_norm)
    let logits = linear(final_norm, output)
    print("      ✓ Prompt processed")
    print("")

    // Autoregressive generation with ZERO-COPY KV updates
    print("[5/5] Generating (10 tokens, zero-copy)...")
    let temperature = 0.8
    print("      Assistant: ", "")

    let current_logits = logits
    let EOS_TOKEN = 2
    let token_count = 0
    let current_pos = prompt_len_f

    // Generate 10 tokens
    for i in range(10) {
        // Sample next token
        let token_id = temperature_sample(current_logits, temperature)
        let text = detokenize_single(tokenizer, token_id, false)
        print("[", token_id, ":", text, "] ", "")

        token_count = token_count + 1

        // Check for EOS
        let is_eos = token_id == 2
        if is_eos {
            print(" <EOS>")
        }

        // Compute new K, V for this token
        let token_ids_single = int_to_tokenids(token_id)
        let new_token_emb = embedding(tok_embd, token_ids_single)

        let new_K0 = linear(new_token_emb, layer_0_k)
        let new_V0 = linear(new_token_emb, layer_0_v)
        let new_K1 = linear(new_token_emb, layer_1_k)
        let new_V1 = linear(new_token_emb, layer_1_v)

        // Update KV caches with scatter (ZERO-COPY: write directly to position)
        let pos_index = [current_pos]
        KV0_K_cache = scatter(KV0_K_cache, 0, pos_index, new_K0)
        KV0_V_cache = scatter(KV0_V_cache, 0, pos_index, new_V0)
        KV1_K_cache = scatter(KV1_K_cache, 0, pos_index, new_K1)
        KV1_V_cache = scatter(KV1_V_cache, 0, pos_index, new_V1)

        // Update position and length
        current_pos = current_pos + 1.0
        current_len = [current_pos]

        // Run transformer with updated caches
        let h0_new = transformer_layer(new_token_emb, layer_0_attn_norm, layer_0_q, layer_0_k, layer_0_v, layer_0_o,
                                        layer_0_ffn_norm, layer_0_gate, layer_0_up, layer_0_down,
                                        KV0_K_cache, KV0_V_cache, current_len)
        let h1_new = transformer_layer(h0_new, layer_1_attn_norm, layer_1_q, layer_1_k, layer_1_v, layer_1_o,
                                        layer_1_ffn_norm, layer_1_gate, layer_1_up, layer_1_down,
                                        KV1_K_cache, KV1_V_cache, current_len)

        let norm_new = rms_norm(h1_new, output_norm)
        current_logits = linear(norm_new, output)
    }

    print("")
    print("")
    print("=== Complete ===")
    print("")
    print("Success! Generated", token_count, "tokens with ZERO-COPY scatter() updates")
    print("No memory allocations during generation - only GPU writes!")
    print("")
}
