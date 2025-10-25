// Greedy Sampling Version for Debugging
// Temperature = 0.0 (always pick highest probability token)

fn transformer_layer(x, W_q, W_k, W_v, W_o, attn_norm, W_gate, W_up, W_down, ffn_norm) {
    // RMSNorm before attention
    let x_norm = rms_norm(x, attn_norm)

    // GQA: 32 Q heads, 4 KV heads
    let Q = matmul(x_norm, W_q)     // [seq_len, 2048]
    let K = matmul(x_norm, W_k)     // [seq_len, 256]
    let V = matmul(x_norm, W_v)     // [seq_len, 256]

    // Reshape for multi-head attention
    let Q_heads = reshape(Q, -1.0, 32.0, 64.0)   // [seq_len, 32, 64]
    let K_heads = reshape(K, -1.0, 4.0, 64.0)    // [seq_len, 4, 64]
    let V_heads = reshape(V, -1.0, 4.0, 64.0)    // [seq_len, 4, 64]

    // RoPE
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // Attention with GQA
    let attn_out = gqa_attention(Q_rope, K_rope, V_heads)

    // Output projection + residual
    let attn_proj = matmul(attn_out, W_o)
    let h_attn = add(x, attn_proj)

    // RMSNorm before FFN
    let h_norm = rms_norm(h_attn, ffn_norm)

    // SwiGLU FFN
    let gate = matmul(h_norm, W_gate)
    let up = matmul(h_norm, W_up)
    let swiglu = mul(silu(gate), up)
    let down = matmul(swiglu, W_down)

    // Residual
    return add(h_attn, down)
}

main {
    print("=== TinyLlama Greedy Sampling Debug ===")
    print("")

    // Load model
    print("[1/3] Loading model...")
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = "~/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model loaded")
    print("")

    // Get weights
    print("[2/3] Loading weights...")
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load only first 2 layers for faster debugging
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

    // Tokenize
    print("[3/3] Generating with greedy sampling...")
    let user_input = "Hello"
    let chat_prompt = "<|system|>\\nYou are a helpful assistant.\\n<|user|>\\n" + user_input + "\\n<|assistant|>\\n"
    let tokens = tokenize(tokenizer, chat_prompt, true)
    print("      Input tokens:", tokens)
    print("")

    let gen_tokens = tokens

    // ========== Token 1 ==========
    print("      [1/3] Token 1 (Greedy)...")
    let e1 = embedding(embed_table, gen_tokens)
    print("            Embedding shape:", shape(e1))

    let h0 = transformer_layer(e1, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    print("            Layer 0 output shape:", shape(h0))

    let h1 = transformer_layer(h0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    print("            Layer 1 output shape:", shape(h1))

    let final_norm1 = rms_norm(h1, output_norm)
    let logits1 = matmul(final_norm1, output_weight)
    print("            Logits shape:", shape(logits1))

    // Greedy sampling (temperature = 0.0)
    let t1 = temperature_sample(logits1, 0.0)
    gen_tokens := append(gen_tokens, t1)
    print("            Sampled token:", t1)
    print("")

    // ========== Token 2 ==========
    print("      [2/3] Token 2 (Greedy)...")
    let e2 = embedding(embed_table, gen_tokens)
    let h0_2 = transformer_layer(e2, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_2 = transformer_layer(h0_2, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let final_norm2 = rms_norm(h1_2, output_norm)
    let logits2 = matmul(final_norm2, output_weight)
    let t2 = temperature_sample(logits2, 0.0)
    gen_tokens := append(gen_tokens, t2)
    print("            Sampled token:", t2)
    print("")

    // ========== Token 3 ==========
    print("      [3/3] Token 3 (Greedy)...")
    let e3 = embedding(embed_table, gen_tokens)
    let h0_3 = transformer_layer(e3, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    let h1_3 = transformer_layer(h0_3, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
    let final_norm3 = rms_norm(h1_3, output_norm)
    let logits3 = matmul(final_norm3, output_weight)
    let t3 = temperature_sample(logits3, 0.0)
    gen_tokens := append(gen_tokens, t3)
    print("            Sampled token:", t3)
    print("")

    // Decode
    print("========================================")
    print("")
    let response = detokenize(tokenizer, gen_tokens)
    print("Full response:")
    print(response)
    print("")

    let gen_only = [t1, t2, t3]
    let generated = detokenize(tokenizer, gen_only)
    print("Generated tokens only:")
    print("  Tokens:", gen_only)
    print("  Text:", generated)
    print("")

    print("✅ Greedy sampling complete!")
}
