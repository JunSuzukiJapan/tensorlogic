// Simple test: Run full model with 38 vs 46 tokens

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

    let attn_weights = softmax(scaled_scores, 2)
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
    let normed = rms_norm(x, attn_norm)
    let Q = linear(normed, W_q)
    let K = linear(normed, W_k)
    let V = linear(normed, W_v)
    let attn_output = gqa_attention(Q, K, V, W_o)
    let x_after_attn = x + attn_output

    let normed2 = rms_norm(x_after_attn, ffn_norm)
    let up = linear(normed2, W_up)
    let gate = linear(normed2, W_gate)
    let gate_act = silu(gate)
    let ffn_hidden = up * gate_act
    let ffn_output = linear(ffn_hidden, W_down)

    x_after_attn + ffn_output
}

main {
    print("=== Simple Test: 38 vs 46 Tokens ===")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")

    // Test 1: 38 tokens
    print("=== 38 Tokens ===")
    let system_prompt_38 = "You are a friendly chatbot."
    let user_message_38 = "Hello! How are you?"
    let formatted_prompt_38 = "<|system|>\n" + system_prompt_38 + "</s>\n<|user|>\n" + user_message_38 + "</s>\n<|assistant|>\n"

    let tokens_38 = tokenize(tokenizer, formatted_prompt_38, false)
    print("Token count:", len(tokens_38))

    let emb_38 = embedding(emb_weight, tokens_38)
    let x = emb_38

    // Run 2 layers only
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
        print("  Layer", layer_idx, "sum:", sum(x))
    }
    print("")

    // Test 2: 46 tokens
    print("=== 46 Tokens ===")
    let system_prompt_46 = "You are a friendly and helpful AI assistant."
    let user_message_46 = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt_46 = "<|system|>\n" + system_prompt_46 + "</s>\n<|user|>\n" + user_message_46 + "</s>\n<|assistant|>\n"

    let tokens_46 = tokenize(tokenizer, formatted_prompt_46, false)
    print("Token count:", len(tokens_46))

    let emb_46 = embedding(emb_weight, tokens_46)
    let y = emb_46

    // Run 2 layers only
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

        y = transformer_layer(y, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        print("  Layer", layer_idx, "sum:", sum(y))
    }
    print("")

    print("=== Expected: Both should complete without NaN/inf ===")
    print("=== Test Complete ===")
}
