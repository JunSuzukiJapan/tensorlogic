// Debug 30-token generation step by step
// Print every intermediate value to find where NaN appears

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    print("    FFN: gate projection...")
    let gate = linear(x, W_gate)
    print("      gate (first 5):", gate)

    print("    FFN: silu activation...")
    let gate_act = silu(gate)
    print("      gate_act (first 5):", gate_act)

    print("    FFN: up projection...")
    let up = linear(x, W_up)
    print("      up (first 5):", up)

    print("    FFN: intermediate...")
    let intermediate = gate_act * up
    print("      intermediate (first 5):", intermediate)

    print("    FFN: down projection...")
    let down = linear(intermediate, W_down)
    print("      down (first 5):", down)

    result := down
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    print("    Attention: reshaping...")
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]
    print("      seq_len:", seq_len_f)

    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)
    print("      Q_heads (first 5):", Q_heads)
    print("      K_heads (first 5):", K_heads)
    print("      V_heads (first 5):", V_heads)

    print("    Attention: RoPE...")
    let Q_rope = rope(Q_heads)
    print("      Q_rope (first 5):", Q_rope)

    let K_rope = rope(K_heads)
    print("      K_rope (first 5):", K_rope)

    print("    Attention: GQA expansion...")
    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)
    print("      K_with_group (first 5):", K_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)
    print("      K_broadcast (first 5):", K_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)
    print("      K_expanded (first 5):", K_expanded)

    print("    Attention: Einsum QK...")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("      scores (first 5):", scores)

    print("    Attention: Scaling...")
    let scaled_scores = scores * 0.125
    print("      scaled_scores (first 5):", scaled_scores)

    print("    Attention: Softmax...")
    let attn_weights = softmax(scaled_scores, 2)
    print("      attn_weights (first 5):", attn_weights)

    print("    Attention: Einsum AttnV...")
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("      attn_output (first 5):", attn_output)

    print("    Attention: Output projection...")
    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    print("      attn_reshaped (first 5):", attn_reshaped)

    let out = linear(attn_reshaped, W_o)
    print("      final output (first 5):", out)

    result := out
}

fn transformer_layer(
    layer_idx: int,
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?]
) -> float16[?, ?] {
    print("  [Layer", layer_idx, "]")
    print("    Input (first 5):", x)

    print("    Attention norm...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("      x_norm1 (first 5):", x_norm1)

    print("    QKV projections...")
    let Q = linear(x_norm1, W_q)
    print("      Q (first 5):", Q)
    let K = linear(x_norm1, W_k)
    print("      K (first 5):", K)
    let V = linear(x_norm1, W_v)
    print("      V (first 5):", V)

    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

    print("    Residual 1...")
    let x1 = x + attn_out
    print("      x1 (first 5):", x1)

    print("    FFN norm...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("      x_norm2 (first 5):", x_norm2)

    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

    print("    Residual 2...")
    let x2 = x1 + ffn_out
    print("      x2 (first 5):", x2)

    result := x2
}

main {
    print("=== Step-by-step 30-token Debug ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = home + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    print("[1] Loading model...")
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    print("[2] Tokenizing (29 tokens)...")
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    let gen_tokens = tokens
    print("  Tokens:", gen_tokens)
    print("")

    print("[3] First generation (29 tokens)...")
    print("")
    let e1 = embedding(embed_table, gen_tokens)
    print("  Embedding (first 5):", e1)

    let h0_1 = transformer_layer(0, e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)

    let final_norm1 = rms_norm(h0_1, output_norm)
    print("  Final norm (first 5):", final_norm1)

    let logits1 = linear(final_norm1, output_weight)
    print("  Logits (first 5):", logits1)

    let t1 = temperature_sample(logits1, 0.8)
    gen_tokens := append(gen_tokens, t1)
    print("  Generated token:", t1)
    print("")

    print("[4] Second generation (30 tokens) - CRITICAL...")
    print("")
    let e2 = embedding(embed_table, gen_tokens)
    print("  Embedding (first 5):", e2)

    let h0_2 = transformer_layer(0, e2, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)

    let final_norm2 = rms_norm(h0_2, output_norm)
    print("  Final norm (first 5):", final_norm2)

    let logits2 = linear(final_norm2, output_weight)
    print("  Logits (first 5):", logits2)

    print("")
    print("=== Check where NaN first appears ===")
}
