// 2-layer chat demo with RoPE - Performance profiling

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
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

    // Apply RoPE
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA expansion
    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    // Attention
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Output projection
    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    result := matmul(attn_reshaped, W_o)
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
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = matmul(x_norm1, W_q)
    let K = matmul(x_norm1, W_k)
    let V = matmul(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out

    let x_norm2 = rms_norm(x1, ffn_norm)
    let gate = matmul(x_norm2, W_gate)
    let up = matmul(x_norm2, W_up)
    let intermediate = silu(gate) * up
    let ffn_out = matmul(intermediate, W_down)

    result := x1 + ffn_out
}

main {
    print("=== 2-Layer Chat with RoPE - Performance Test ===")
    print("")

    print("[1/4] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model loaded")
    print("")

    print("[2/4] Loading 2 layers...")
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
    print("      ✓ 2 layers loaded")
    print("")

    print("[3/4] Tokenizing...")
    let user_input = "Hello"
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\n" + user_input + "\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("      Tokens:", tokens)
    print("")

    print("[4/4] Generating 3 tokens...")
    let gen_tokens = tokens

    // Token 1
    print("      [1/3] Token 1...")
    let e1 = embedding(embed_table, gen_tokens)
    let h0_1 = transformer_layer(e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_1 = transformer_layer(h0_1, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let norm1 = rms_norm(h1_1, output_norm)
    let logits1 = matmul(norm1, output_weight)
    let t1 = temperature_sample(logits1, 0.8)
    gen_tokens := append(gen_tokens, t1)
    print("            Token:", t1)

    // Token 2
    print("      [2/3] Token 2...")
    let e2 = embedding(embed_table, gen_tokens)
    let h0_2 = transformer_layer(e2, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_2 = transformer_layer(h0_2, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let norm2 = rms_norm(h1_2, output_norm)
    let logits2 = matmul(norm2, output_weight)
    let t2 = temperature_sample(logits2, 0.8)
    gen_tokens := append(gen_tokens, t2)
    print("            Token:", t2)

    // Token 3
    print("      [3/3] Token 3...")
    let e3 = embedding(embed_table, gen_tokens)
    let h0_3 = transformer_layer(e3, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_3 = transformer_layer(h0_3, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let norm3 = rms_norm(h1_3, output_norm)
    let logits3 = matmul(norm3, output_weight)
    let t3 = temperature_sample(logits3, 0.8)
    gen_tokens := append(gen_tokens, t3)
    print("            Token:", t3)

    print("")
    print("========================================")
    let generated_text = detokenize(tokenizer, gen_tokens, true)
    print("Response:", generated_text)
    print("========================================")
    print("")
    print("✅ 2-layer test complete!")
}
