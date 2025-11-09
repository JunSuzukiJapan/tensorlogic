// Layer-by-layer debugging with 29 tokens (same as chat demo)
// Find where NaN first appears with longer sequence

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    x * sig
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
    let seq_len_f = Q_shape_tensor[0]

    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    linear(attn_reshaped, W_o)
}

main {
    print("=== 1-Layer Debug with 29 Tokens ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = home + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    print("[1] Tokenizing prompt")
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("  Tokens:", tokens)
    print("")

    print("[2] Embedding")
    let embed_table = get_tensor(model, "token_embd.weight")
    let x = embedding(embed_table, tokens)
    print("  x shape:", shape(x))
    print("  x (first 5):", x)
    print("")

    print("[3] Layer 0")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")

    print("  [3.1] Attention norm")
    let x_norm1 = rms_norm(x, attn_norm_0)
    print("    x_norm1 (first 5):", x_norm1)

    print("  [3.2] Q/K/V projections")
    let Q = linear(x_norm1, W_q_0)
    let K = linear(x_norm1, W_k_0)
    let V = linear(x_norm1, W_v_0)
    print("    Q (first 5):", Q)
    print("    K (first 5):", K)
    print("    V (first 5):", V)

    print("  [3.3] Attention")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o_0)
    print("    attn_out (first 5):", attn_out)

    print("  [3.4] Residual 1")
    let x1 = x + attn_out
    print("    x1 (first 5):", x1)

    print("  [3.5] FFN norm")
    let x_norm2 = rms_norm(x1, ffn_norm_0)
    print("    x_norm2 (first 5):", x_norm2)

    print("  [3.6] FFN")
    let ffn_out = swiglu_ffn(x_norm2, W_gate_0, W_up_0, W_down_0)
    print("    ffn_out (first 5):", ffn_out)

    print("  [3.7] Residual 2")
    let x2 = x1 + ffn_out
    print("    x2 (first 5):", x2)
    print("")

    print("[4] Final norm")
    let output_norm = get_tensor(model, "output_norm.weight")
    let x_final = rms_norm(x2, output_norm)
    print("  x_final (first 5):", x_final)
    print("")

    print("[5] LM head")
    let lm_head = get_tensor(model, "output.weight")
    let logits = linear(x_final, lm_head)
    print("  logits shape:", shape(logits))
    print("  logits (first 5):", logits)
    print("")

    print("=== Check for NaN/Inf above ===")
}
