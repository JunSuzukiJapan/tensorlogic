// Detailed analysis of Layer 2 with 46 tokens

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn gqa_attention_debug(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    print("    [GQA] seq_len:", seq_len)

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    print("    [GQA] Q_heads sum:", sum(Q_heads), " min:", min(Q_heads), " max:", max(Q_heads))

    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    let Q_rope = rope(Q_heads)
    print("    [GQA] Q_rope sum:", sum(Q_rope), " min:", min(Q_rope), " max:", max(Q_rope))

    let K_rope = rope(K_heads)
    print("    [GQA] K_rope sum:", sum(K_rope), " min:", min(K_rope), " max:", max(K_rope))

    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("    [GQA] Scores sum:", sum(scores), " min:", min(scores), " max:", max(scores))

    let scaled_scores = scores * 0.125
    print("    [GQA] Scaled scores sum:", sum(scaled_scores), " min:", min(scaled_scores), " max:", max(scaled_scores))

    let attn_weights = softmax(scaled_scores, 2)
    print("    [GQA] Attention weights sum:", sum(attn_weights), " min:", min(attn_weights), " max:", max(attn_weights))

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("    [GQA] Attention output sum:", sum(attn_output), " min:", min(attn_output), " max:", max(attn_output))

    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    linear(attn_reshaped, W_o)
}

fn transformer_layer_debug(
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
    print("  [Attn] Norm")
    let normed = rms_norm(x, attn_norm)
    print("    sum:", sum(normed), " min:", min(normed), " max:", max(normed))

    print("  [Attn] Q, K, V projections")
    let Q = linear(normed, W_q)
    print("    Q sum:", sum(Q), " min:", min(Q), " max:", max(Q))
    let K = linear(normed, W_k)
    print("    K sum:", sum(K), " min:", min(K), " max:", max(K))
    let V = linear(normed, W_v)
    print("    V sum:", sum(V), " min:", min(V), " max:", max(V))

    print("  [Attn] GQA")
    let attn_output = gqa_attention_debug(Q, K, V, W_o)
    print("    Final attn output sum:", sum(attn_output), " min:", min(attn_output), " max:", max(attn_output))

    print("  [Attn] Residual")
    let x_after_attn = x + attn_output
    print("    sum:", sum(x_after_attn), " min:", min(x_after_attn), " max:", max(x_after_attn))

    print("  [FFN] Norm")
    let normed2 = rms_norm(x_after_attn, ffn_norm)
    print("    sum:", sum(normed2), " min:", min(normed2), " max:", max(normed2))

    print("  [FFN] Up and Gate projections")
    let up = linear(normed2, W_up)
    print("    Up sum:", sum(up), " min:", min(up), " max:", max(up))
    let gate = linear(normed2, W_gate)
    print("    Gate sum:", sum(gate), " min:", min(gate), " max:", max(gate))

    print("  [FFN] SiLU")
    let gate_act = silu(gate)
    print("    sum:", sum(gate_act), " min:", min(gate_act), " max:", max(gate_act))

    print("  [FFN] Element-wise multiply")
    let ffn_hidden = up * gate_act
    print("    sum:", sum(ffn_hidden), " min:", min(ffn_hidden), " max:", max(ffn_hidden))

    print("  [FFN] Down projection")
    let ffn_output = linear(ffn_hidden, W_down)
    print("    sum:", sum(ffn_output), " min:", min(ffn_output), " max:", max(ffn_output))

    print("  [FFN] Final residual")
    let result = x_after_attn + ffn_output
    print("    sum:", sum(result), " min:", min(result), " max:", max(result))

    result
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
    let attn_output_projected = linear(attn_reshaped, W_o)
    let x_after_attn = x + attn_output_projected

    let normed2 = rms_norm(x_after_attn, ffn_norm)
    let up = linear(normed2, W_up)
    let gate = linear(normed2, W_gate)
    let gate_act = silu(gate)
    let ffn_hidden = up * gate_act
    let ffn_output = linear(ffn_hidden, W_down)

    x_after_attn + ffn_output
}

main {
    print("=== 46 Tokens - Layer 2 Detail ===")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")

    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("Token count:", len(tokens))

    let emb = embedding(emb_weight, tokens)
    print("Embedding sum:", sum(emb))
    print("")

    let x = emb

    print("=== Layer 0 ===")
    x = transformer_layer(x, model.blk[0].attn_norm.weight, model.blk[0].attn_q.weight, model.blk[0].attn_k.weight, model.blk[0].attn_v.weight, model.blk[0].attn_output.weight, model.blk[0].ffn_norm.weight, model.blk[0].ffn_gate.weight, model.blk[0].ffn_up.weight, model.blk[0].ffn_down.weight)
    print("Layer 0 sum:", sum(x))
    print("")

    print("=== Layer 1 ===")
    x = transformer_layer(x, model.blk[1].attn_norm.weight, model.blk[1].attn_q.weight, model.blk[1].attn_k.weight, model.blk[1].attn_v.weight, model.blk[1].attn_output.weight, model.blk[1].ffn_norm.weight, model.blk[1].ffn_gate.weight, model.blk[1].ffn_up.weight, model.blk[1].ffn_down.weight)
    print("Layer 1 sum:", sum(x))
    print("")

    print("=== Layer 2 (DETAILED) ===")
    x = transformer_layer_debug(x, model.blk[2].attn_norm.weight, model.blk[2].attn_q.weight, model.blk[2].attn_k.weight, model.blk[2].attn_v.weight, model.blk[2].attn_output.weight, model.blk[2].ffn_norm.weight, model.blk[2].ffn_gate.weight, model.blk[2].ffn_up.weight, model.blk[2].ffn_down.weight)
    print("Layer 2 sum:", sum(x))
    print("")

    print("=== Test Complete ===")
}
