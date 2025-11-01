// 5-Layer greedy decoding (temp=0.0) for deterministic NaN reproduction
// Use argmax instead of temperature sampling

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
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
    result = linear(intermediate, W_down)
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
    result = linear(attn_reshaped, W_o)
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
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    result = x1 + ffn_out
}

main {
    print("=== 5-Layer Greedy Decoding (temp=0.0) ===")
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

    // Load 5 layers
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

    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")

    let W_q_3 = get_tensor(model, "blk.3.attn_q.weight")
    let W_k_3 = get_tensor(model, "blk.3.attn_k.weight")
    let W_v_3 = get_tensor(model, "blk.3.attn_v.weight")
    let W_o_3 = get_tensor(model, "blk.3.attn_output.weight")
    let attn_norm_3 = get_tensor(model, "blk.3.attn_norm.weight")
    let W_gate_3 = get_tensor(model, "blk.3.ffn_gate.weight")
    let W_up_3 = get_tensor(model, "blk.3.ffn_up.weight")
    let W_down_3 = get_tensor(model, "blk.3.ffn_down.weight")
    let ffn_norm_3 = get_tensor(model, "blk.3.ffn_norm.weight")

    let W_q_4 = get_tensor(model, "blk.4.attn_q.weight")
    let W_k_4 = get_tensor(model, "blk.4.attn_k.weight")
    let W_v_4 = get_tensor(model, "blk.4.attn_v.weight")
    let W_o_4 = get_tensor(model, "blk.4.attn_output.weight")
    let attn_norm_4 = get_tensor(model, "blk.4.attn_norm.weight")
    let W_gate_4 = get_tensor(model, "blk.4.ffn_gate.weight")
    let W_up_4 = get_tensor(model, "blk.4.ffn_up.weight")
    let W_down_4 = get_tensor(model, "blk.4.ffn_down.weight")
    let ffn_norm_4 = get_tensor(model, "blk.4.ffn_norm.weight")

    print("[2] Tokenizing...")
    let chat_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    let gen_tokens = tokens
    print("  Initial tokens:", gen_tokens)
    print("")

    print("[3] Generating 10 tokens (greedy)...")
    print("")

    // Token 1
    print("  [1/10] seq_len=29...")
    let e1 = embedding(embed_table, gen_tokens)
    let h0_1 = transformer_layer(e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_1 = transformer_layer(h0_1, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let h2_1 = transformer_layer(h1_1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)
    let h3_1 = transformer_layer(h2_1, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3)
    let h4_1 = transformer_layer(h3_1, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4)
    let final_norm1 = rms_norm(h4_1, output_norm)
    let logits1 = linear(final_norm1, output_weight)
    print("    logits (first 5):", logits1)
    let t1 = argmax(logits1)
    gen_tokens = append(gen_tokens, t1)
    print("    Token:", t1)

    // Token 2
    print("  [2/10] seq_len=30...")
    let e2 = embedding(embed_table, gen_tokens)
    let h0_2 = transformer_layer(e2, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_2 = transformer_layer(h0_2, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let h2_2 = transformer_layer(h1_2, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)
    let h3_2 = transformer_layer(h2_2, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3)
    let h4_2 = transformer_layer(h3_2, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4)
    let final_norm2 = rms_norm(h4_2, output_norm)
    let logits2 = linear(final_norm2, output_weight)
    print("    logits (first 5):", logits2)
    let t2 = argmax(logits2)
    gen_tokens = append(gen_tokens, t2)
    print("    Token:", t2)

    // Token 3
    print("  [3/10] seq_len=31...")
    let e3 = embedding(embed_table, gen_tokens)
    let h0_3 = transformer_layer(e3, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_3 = transformer_layer(h0_3, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let h2_3 = transformer_layer(h1_3, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)
    let h3_3 = transformer_layer(h2_3, W_q_3, W_k_3, W_v_3, W_o_3, attn_norm_3, W_gate_3, W_up_3, W_down_3, ffn_norm_3)
    let h4_3 = transformer_layer(h3_3, W_q_4, W_k_4, W_v_4, W_o_4, attn_norm_4, W_gate_4, W_up_4, W_down_4, ffn_norm_4)
    let final_norm3 = rms_norm(h4_3, output_norm)
    let logits3 = linear(final_norm3, output_weight)
    print("    logits (first 5):", logits3)
    let t3 = argmax(logits3)
    gen_tokens = append(gen_tokens, t3)
    print("    Token:", t3)

    // Continue for more tokens...
    print("  (Continue generating more tokens...)")

    print("")
    print("=== Check for NaN above ===")
    print("Final tokens:", gen_tokens)
}
