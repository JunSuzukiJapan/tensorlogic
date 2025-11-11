// Detailed analysis of Layer 1 with 46 tokens
// To find where values explode

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

    print("    [GQA] seq_len:", seq_len)

    // Reshape to heads
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    print("    [GQA] Q_heads sum:", sum(Q_heads), " min:", min(Q_heads), " max:", max(Q_heads))

    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    // Apply RoPE to Q and K
    let Q_rope = rope(Q_heads)
    print("    [GQA] Q_rope sum:", sum(Q_rope), " min:", min(Q_rope), " max:", max(Q_rope))

    let K_rope = rope(K_heads)
    print("    [GQA] K_rope sum:", sum(K_rope), " min:", min(K_rope), " max:", max(K_rope))

    // GQA: Expand K, V from 4 to 32 heads
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])

    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("    [GQA] Attention scores sum:", sum(scores), " min:", min(scores), " max:", max(scores))

    let scaled_scores = scores * 0.125
    print("    [GQA] Scaled scores sum:", sum(scaled_scores), " min:", min(scaled_scores), " max:", max(scaled_scores))

    let attn_weights = softmax(scaled_scores, 2)
    print("    [GQA] Attention weights sum:", sum(attn_weights), " min:", min(attn_weights), " max:", max(attn_weights))

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("    [GQA] Attention output sum:", sum(attn_output), " min:", min(attn_output), " max:", max(attn_output))

    // Flatten and project
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
    // Attention
    let normed = rms_norm(x, attn_norm)
    let Q = linear(normed, W_q)
    let K = linear(normed, W_k)
    let V = linear(normed, W_v)
    let attn_output = gqa_attention(Q, K, V, W_o)
    let x_after_attn = x + attn_output

    // FFN
    let normed2 = rms_norm(x_after_attn, ffn_norm)
    let up = linear(normed2, W_up)
    let gate = linear(normed2, W_gate)
    let gate_act = silu(gate)
    let ffn_hidden = up * gate_act
    let ffn_output = linear(ffn_hidden, W_down)

    x_after_attn + ffn_output
}

main {
    print("=== 46-Token Layer 1 Detailed Analysis ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")

    // 46-token prompt
    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let tokens = tokenize(tokenizer, formatted_prompt, false)
    print("[0] Token count:", len(tokens))
    print("")

    // Get embeddings
    let emb = embedding(emb_weight, tokens)
    print("Embedding sum:", sum(emb))
    print("  min:", min(emb), " max:", max(emb))
    print("")

    // Process Layer 0
    print("=== Processing Layer 0 ===")
    let L0_attn_norm = model.blk[0].attn_norm.weight
    let L0_W_q = model.blk[0].attn_q.weight
    let L0_W_k = model.blk[0].attn_k.weight
    let L0_W_v = model.blk[0].attn_v.weight
    let L0_W_o = model.blk[0].attn_output.weight
    let L0_ffn_norm = model.blk[0].ffn_norm.weight
    let L0_W_gate = model.blk[0].ffn_gate.weight
    let L0_W_up = model.blk[0].ffn_up.weight
    let L0_W_down = model.blk[0].ffn_down.weight

    let x = transformer_layer(emb, L0_attn_norm, L0_W_q, L0_W_k, L0_W_v, L0_W_o, L0_ffn_norm, L0_W_gate, L0_W_up, L0_W_down)
    print("Layer 0 output sum:", sum(x))
    print("  min:", min(x), " max:", max(x))
    print("")

    // Process Layer 1 with detailed intermediate outputs
    print("=== Processing Layer 1 (Detailed) ===")
    let L1_attn_norm = model.blk[1].attn_norm.weight
    let L1_W_q = model.blk[1].attn_q.weight
    let L1_W_k = model.blk[1].attn_k.weight
    let L1_W_v = model.blk[1].attn_v.weight
    let L1_W_o = model.blk[1].attn_output.weight
    let L1_ffn_norm = model.blk[1].ffn_norm.weight
    let L1_W_gate = model.blk[1].ffn_gate.weight
    let L1_W_up = model.blk[1].ffn_up.weight
    let L1_W_down = model.blk[1].ffn_down.weight

    // Attention block
    print("[1] Attention norm")
    let normed = rms_norm(x, L1_attn_norm)
    print("  sum:", sum(normed), " min:", min(normed), " max:", max(normed))

    print("[2] Q, K, V projections")
    let Q = linear(normed, L1_W_q)
    print("  Q sum:", sum(Q), " min:", min(Q), " max:", max(Q))
    let K = linear(normed, L1_W_k)
    print("  K sum:", sum(K), " min:", min(K), " max:", max(K))
    let V = linear(normed, L1_W_v)
    print("  V sum:", sum(V), " min:", min(V), " max:", max(V))

    print("[3] GQA Attention (with internal details)")
    let attn_output = gqa_attention(Q, K, V, L1_W_o)
    print("  Final attention output sum:", sum(attn_output))
    print("  min:", min(attn_output), " max:", max(attn_output))

    print("[4] Residual connection (after attention)")
    let x_after_attn = x + attn_output
    print("  sum:", sum(x_after_attn), " min:", min(x_after_attn), " max:", max(x_after_attn))

    // FFN block
    print("[5] FFN norm")
    let normed2 = rms_norm(x_after_attn, L1_ffn_norm)
    print("  sum:", sum(normed2), " min:", min(normed2), " max:", max(normed2))

    print("[6] FFN projections")
    let up = linear(normed2, L1_W_up)
    print("  Up sum:", sum(up), " min:", min(up), " max:", max(up))
    let gate = linear(normed2, L1_W_gate)
    print("  Gate sum:", sum(gate), " min:", min(gate), " max:", max(gate))

    print("[7] SiLU activation")
    let gate_act = silu(gate)
    print("  SiLU sum:", sum(gate_act), " min:", min(gate_act), " max:", max(gate_act))

    print("[8] Element-wise multiply")
    let ffn_hidden = up * gate_act
    print("  Multiply sum:", sum(ffn_hidden), " min:", min(ffn_hidden), " max:", max(ffn_hidden))

    print("[9] Down projection")
    let ffn_output = linear(ffn_hidden, L1_W_down)
    print("  FFN output sum:", sum(ffn_output))
    print("  min:", min(ffn_output), " max:", max(ffn_output))

    print("[10] Final residual connection")
    let layer1_output = x_after_attn + ffn_output
    print("  Layer 1 final sum:", sum(layer1_output))
    print("  min:", min(layer1_output), " max:", max(layer1_output))
    print("")

    print("=== Expected (38 tokens): -6.65 ===")
    print("=== Actual (46 tokens): ? ===")
    print("=== Test Complete ===")
}
