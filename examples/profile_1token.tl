// Profile single token generation with 2 layers

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    print("        [Attention] Starting...")
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    print("        [Attention] Reshaping to heads...")
    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    print("        [Attention] Q_heads shape:", shape(Q_heads))
    let K_heads = reshape(K, [seq_len_f, 4.0, 64.0])
    print("        [Attention] K_heads shape:", shape(K_heads))
    let V_heads = reshape(V, [seq_len_f, 4.0, 64.0])

    print("        [Attention] >>>>> Applying RoPE to Q_heads...")
    let Q_rope = rope(Q_heads)
    print("        [Attention] <<<<< RoPE Q done")

    print("        [Attention] >>>>> Applying RoPE to K_heads...")
    let K_rope = rope(K_heads)
    print("        [Attention] <<<<< RoPE K done")

    print("        [Attention] GQA expansion...")
    let K_exp = reshape(K_rope, [seq_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [seq_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [seq_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [seq_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [seq_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len_f, 32.0, 64.0])

    print("        [Attention] Computing scores (einsum)...")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("        [Attention] Scaling and softmax...")
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    print("        [Attention] Output einsum...")
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    print("        [Attention] Output projection...")
    let reshaped = reshape(out, [seq_len_f, 2048.0])
    let attn_result = matmul(reshaped, W_o)
    print("        [Attention] Done")
    result := attn_result
}

fn transformer_layer(
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
    print("      [Layer] RMSNorm...")
    let x_norm1 = rms_norm(x, attn_norm)
    print("      [Layer] QKV projections...")
    let Q = matmul(x_norm1, W_q)
    let K = matmul(x_norm1, W_k)
    let V = matmul(x_norm1, W_v)
    print("      [Layer] Attention...")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out

    print("      [Layer] FFN RMSNorm...")
    let x_norm2 = rms_norm(x1, ffn_norm)
    print("      [Layer] FFN projections...")
    let gate = matmul(x_norm2, W_gate)
    let up = matmul(x_norm2, W_up)
    let intermediate = silu(gate) * up
    let ffn_out = matmul(intermediate, W_down)
    print("      [Layer] Done")

    result := x1 + ffn_out
}

main {
    print("=== Single Token Profile (2 layers, 27 input tokens) ===")
    print("")

    print("[1/3] Loading...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

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

    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    print("    ✓ Model and 2 layers loaded")
    print("")

    print("[2/3] Tokenize...")
    let user_input = "Hello"
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\n" + user_input + "\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("    ✓ Tokens:", tokens)
    print("")

    print("[3/3] Generate 1 token with detailed profiling...")
    print("    Embedding...")
    let emb = embedding(embed_table, tokens)
    print("    Embedding done, shape:", shape(emb))
    print("")

    print("    Layer 0...")
    let h0 = transformer_layer(emb, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    print("    Layer 0 done")
    print("")

    print("    Layer 1...")
    let h1 = transformer_layer(h0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    print("    Layer 1 done")
    print("")

    print("    Final norm and sampling...")
    let norm = rms_norm(h1, output_norm)
    let logits = matmul(norm, output_weight)
    let token = temperature_sample(logits, 0.0)

    print("")
    print("========================================")
    print("Generated token:", token)
    print("========================================")
    print("")
    print("✅ Profile complete!")
}
