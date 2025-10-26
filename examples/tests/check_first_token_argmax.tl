// Check argmax of first token prediction
// Compare greedy (argmax) output with llama.cpp

main {
    print("=== First Token Argmax Check ===\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load Layer 0 weights
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    // Input: Single token "Hello" = 15043
    let tokens = [15043.0]
    print("Input tokens: ", tokens, "\n")
    print("Number of tokens: ", len(tokens), "\n")

    // Embedding
    let h = embedding(embed_table, tokens)
    print("Embedding shape: ", shape(h), "\n")

    // Layer 0 - Attention
    let x_norm_0 = rms_norm(h, attn_norm_0)
    let Q_0 = linear(x_norm_0, W_q_0)
    let K_0 = linear(x_norm_0, W_k_0)
    let V_0 = linear(x_norm_0, W_v_0)

    let seq_len = shape(h)[0]
    let Q_0_heads = reshape(Q_0, [seq_len, 32.0, 64.0])
    let K_0_heads = reshape(K_0, [seq_len, 4.0, 64.0])
    let V_0_heads = reshape(V_0, [seq_len, 4.0, 64.0])

    let Q_0_rope = rope(Q_0_heads)
    let K_0_rope = rope(K_0_heads)

    // GQA expansion
    let K_0_exp = reshape(K_0_rope, [seq_len, 4.0, 1.0, 64.0])
    let K_0_bc = broadcast_to(K_0_exp, [seq_len, 4.0, 8.0, 64.0])
    let K_0_expanded = reshape(K_0_bc, [seq_len, 32.0, 64.0])

    let V_0_exp = reshape(V_0_heads, [seq_len, 4.0, 1.0, 64.0])
    let V_0_bc = broadcast_to(V_0_exp, [seq_len, 4.0, 8.0, 64.0])
    let V_0_expanded = reshape(V_0_bc, [seq_len, 32.0, 64.0])

    // Attention
    let scores_0 = einsum("ihd,jhd->ihj", Q_0_rope, K_0_expanded)
    let scaled_0 = scores_0 * 0.125
    let attn_0 = softmax(scaled_0, 2)
    let attn_out_0 = einsum("ihj,jhd->ihd", attn_0, V_0_expanded)

    let attn_flat_0 = reshape(attn_out_0, [seq_len, 2048.0])
    let attn_final_0 = linear(attn_flat_0, W_o_0)
    let h_after_attn_0 = h + attn_final_0

    // Layer 0 - FFN
    let x_ffn_0 = rms_norm(h_after_attn_0, ffn_norm_0)
    let gate_0 = linear(x_ffn_0, W_gate_0)
    let up_0 = linear(x_ffn_0, W_up_0)
    let silu_gate_0 = gate_0 * sigmoid(gate_0)
    let ffn_pre_0 = silu_gate_0 * up_0
    let ffn_out_0 = linear(ffn_pre_0, W_down_0)
    let h_0 = h_after_attn_0 + ffn_out_0

    print("Layer 0 output shape: ", shape(h_0), "\n")

    // Final norm and output projection (h_0 is already [1, 2048])
    let final_norm = rms_norm(h_0, output_norm)
    let logits = linear(final_norm, output_weight)
    print("Logits shape: ", shape(logits), "\n")

    // Get argmax (greedy sampling)
    let predicted_token = argmax(logits)
    print("\nPredicted token (argmax): ", predicted_token, "\n")

    // Decode
    let predicted_text = detokenize(tokenizer, [predicted_token])
    print("Predicted text: '", predicted_text, "'\n")

    print("\nNote: This is Layer 0 only output, not final 22-layer output")
}
