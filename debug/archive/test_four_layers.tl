// Test: 4-layer transformer for debugging
// Progressive testing approach

// SiLU activation function (inline version now works with fix)
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("TensorLogic 4-Layer Transformer Test")
    print("================================================================================")
    print("")

    // ========================================================================
    print("[1/5] Loading model and tokenizer...")
    // ========================================================================

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model and tokenizer loaded")
    print("")

    // ========================================================================
    print("[2/5] Loading weights for Layers 0, 1, 2, and 3...")
    // ========================================================================

    // Embedding and output
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Layer 0 weights
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    // Layer 1 weights
    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")

    // Layer 2 weights
    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")

    // Layer 3 weights
    let W_q_3 = get_tensor(model, "blk.3.attn_q.weight")
    let W_k_3 = get_tensor(model, "blk.3.attn_k.weight")
    let W_v_3 = get_tensor(model, "blk.3.attn_v.weight")
    let W_o_3 = get_tensor(model, "blk.3.attn_output.weight")
    let attn_norm_3 = get_tensor(model, "blk.3.attn_norm.weight")
    let W_gate_3 = get_tensor(model, "blk.3.ffn_gate.weight")
    let W_up_3 = get_tensor(model, "blk.3.ffn_up.weight")
    let W_down_3 = get_tensor(model, "blk.3.ffn_down.weight")
    let ffn_norm_3 = get_tensor(model, "blk.3.ffn_norm.weight")

    print("      ✓ Layers 0, 1, 2, and 3 weights loaded")
    print("")

    // ========================================================================
    print("[3/5] Creating prompt...")
    // ========================================================================

    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let chat_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("      Prompt:", chat_prompt)
    print("")

    // ========================================================================
    print("[4/5] Tokenizing...")
    // ========================================================================

    // No BOS token (false) to match HuggingFace
    let tokens = tokenize(tokenizer, chat_prompt, false)
    let num_tokens = len(tokens)

    print("      Tokens:", num_tokens)
    print("      Token IDs:", tokens)
    print("")

    // ========================================================================
    print("[5/5] Running forward pass (Layers 0, 1, 2, and 3)...")
    // ========================================================================

    // Embedding
    let embeddings = embedding(embed_table, tokens)
    let emb_sum = sum(embeddings)
    print("      Embedding sum:", emb_sum)

    // Layer 0: Attention
    let x_norm = rms_norm(embeddings, attn_norm_0)
    print("      [DEBUG] x_norm sum:", sum(x_norm))

    let Q = linear(x_norm, W_q_0)
    print("      [DEBUG] Q sum:", sum(Q))
    let K = linear(x_norm, W_k_0)
    print("      [DEBUG] K sum:", sum(K))
    let V = linear(x_norm, W_v_0)
    print("      [DEBUG] V sum:", sum(V))

    // GQA attention (simplified - using builtin)
    let Q_shape = shape(Q)
    let seq_len = Q_shape[0]

    let Q_heads = reshape(Q, [seq_len, 32.0, 64.0])
    let K_heads = reshape(K, [seq_len, 4.0, 64.0])
    let V_heads = reshape(V, [seq_len, 4.0, 64.0])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA expansion
    let K_group = reshape(K_rope, [seq_len, 4.0, 1.0, 64.0])
    let V_group = reshape(V_heads, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_group, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_group, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [seq_len, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)
    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Attention output projection
    let attn_reshaped = reshape(attn_output, [seq_len, 2048.0])
    let attn_out = linear(attn_reshaped, W_o_0)
    print("      [DEBUG] attn_out sum:", sum(attn_out))

    // Residual
    let hidden_1 = embeddings + attn_out
    print("      [DEBUG] hidden_1 (after attn) sum:", sum(hidden_1))

    // Layer 0: FFN
    let x_norm2 = rms_norm(hidden_1, ffn_norm_0)
    print("      [DEBUG] x_norm2 sum:", sum(x_norm2))

    let gate = linear(x_norm2, W_gate_0)
    let up = linear(x_norm2, W_up_0)
    let silu_gate = silu(gate)
    let gated = silu_gate * up
    let ffn_out = linear(gated, W_down_0)

    print("      [DEBUG] ffn_out sum:", sum(ffn_out))

    // Final residual
    let layer_0_output = hidden_1 + ffn_out

    let layer_0_sum = sum(layer_0_output)
    print("      Layer 0 output sum:", layer_0_sum)
    print("")

    // ========================================================================
    // Layer 1: Attention
    // ========================================================================
    let x_norm_l1 = rms_norm(layer_0_output, attn_norm_1)
    print("      [DEBUG] Layer 1 x_norm sum:", sum(x_norm_l1))

    let Q_l1 = linear(x_norm_l1, W_q_1)
    let K_l1 = linear(x_norm_l1, W_k_1)
    let V_l1 = linear(x_norm_l1, W_v_1)

    let Q_heads_l1 = reshape(Q_l1, [seq_len, 32.0, 64.0])
    let K_heads_l1 = reshape(K_l1, [seq_len, 4.0, 64.0])
    let V_heads_l1 = reshape(V_l1, [seq_len, 4.0, 64.0])

    let Q_rope_l1 = rope(Q_heads_l1)
    let K_rope_l1 = rope(K_heads_l1)

    // GQA expansion
    let K_group_l1 = reshape(K_rope_l1, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l1 = reshape(V_heads_l1, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l1 = broadcast_to(K_group_l1, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l1 = broadcast_to(V_group_l1, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l1 = reshape(K_broadcast_l1, [seq_len, 32.0, 64.0])
    let V_expanded_l1 = reshape(V_broadcast_l1, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l1 = einsum("ihd,jhd->ihj", Q_rope_l1, K_expanded_l1)
    let scaled_scores_l1 = scores_l1 * 0.125  // 1/sqrt(64)
    let attn_weights_l1 = softmax(scaled_scores_l1, 2)
    let attn_output_l1 = einsum("ihj,jhd->ihd", attn_weights_l1, V_expanded_l1)

    // Attention output projection
    let attn_reshaped_l1 = reshape(attn_output_l1, [seq_len, 2048.0])
    let attn_out_l1 = linear(attn_reshaped_l1, W_o_1)
    print("      [DEBUG] Layer 1 attn_out sum:", sum(attn_out_l1))

    // Residual
    let hidden_l1 = layer_0_output + attn_out_l1
    print("      [DEBUG] Layer 1 hidden (after attn) sum:", sum(hidden_l1))

    // Layer 1: FFN
    let x_norm2_l1 = rms_norm(hidden_l1, ffn_norm_1)
    print("      [DEBUG] Layer 1 x_norm2 sum:", sum(x_norm2_l1))

    let gate_l1 = linear(x_norm2_l1, W_gate_1)
    let up_l1 = linear(x_norm2_l1, W_up_1)
    let silu_gate_l1 = silu(gate_l1)
    let gated_l1 = silu_gate_l1 * up_l1
    let ffn_out_l1 = linear(gated_l1, W_down_1)

    print("      [DEBUG] Layer 1 ffn_out sum:", sum(ffn_out_l1))

    // Final residual
    let layer_1_output = hidden_l1 + ffn_out_l1

    let layer_1_sum = sum(layer_1_output)
    print("      Layer 1 output sum:", layer_1_sum)
    print("")

    // ========================================================================
    // Layer 2: Attention
    // ========================================================================
    let x_norm_l2 = rms_norm(layer_1_output, attn_norm_2)
    print("      [DEBUG] Layer 2 x_norm sum:", sum(x_norm_l2))

    let Q_l2 = linear(x_norm_l2, W_q_2)
    let K_l2 = linear(x_norm_l2, W_k_2)
    let V_l2 = linear(x_norm_l2, W_v_2)

    let Q_heads_l2 = reshape(Q_l2, [seq_len, 32.0, 64.0])
    let K_heads_l2 = reshape(K_l2, [seq_len, 4.0, 64.0])
    let V_heads_l2 = reshape(V_l2, [seq_len, 4.0, 64.0])

    let Q_rope_l2 = rope(Q_heads_l2)
    let K_rope_l2 = rope(K_heads_l2)

    // GQA expansion
    let K_group_l2 = reshape(K_rope_l2, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l2 = reshape(V_heads_l2, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l2 = broadcast_to(K_group_l2, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l2 = broadcast_to(V_group_l2, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l2 = reshape(K_broadcast_l2, [seq_len, 32.0, 64.0])
    let V_expanded_l2 = reshape(V_broadcast_l2, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l2 = einsum("ihd,jhd->ihj", Q_rope_l2, K_expanded_l2)
    let scaled_scores_l2 = scores_l2 * 0.125  // 1/sqrt(64)
    let attn_weights_l2 = softmax(scaled_scores_l2, 2)
    let attn_output_l2 = einsum("ihj,jhd->ihd", attn_weights_l2, V_expanded_l2)

    // Attention output projection
    let attn_reshaped_l2 = reshape(attn_output_l2, [seq_len, 2048.0])
    let attn_out_l2 = linear(attn_reshaped_l2, W_o_2)
    print("      [DEBUG] Layer 2 attn_out sum:", sum(attn_out_l2))

    // Residual
    let hidden_l2 = layer_1_output + attn_out_l2
    print("      [DEBUG] Layer 2 hidden (after attn) sum:", sum(hidden_l2))

    // Layer 2: FFN
    let x_norm2_l2 = rms_norm(hidden_l2, ffn_norm_2)
    print("      [DEBUG] Layer 2 x_norm2 sum:", sum(x_norm2_l2))

    let gate_l2 = linear(x_norm2_l2, W_gate_2)
    let up_l2 = linear(x_norm2_l2, W_up_2)
    let silu_gate_l2 = silu(gate_l2)
    let gated_l2 = silu_gate_l2 * up_l2
    let ffn_out_l2 = linear(gated_l2, W_down_2)

    print("      [DEBUG] Layer 2 ffn_out sum:", sum(ffn_out_l2))

    // Final residual
    let layer_2_output = hidden_l2 + ffn_out_l2

    let layer_2_sum = sum(layer_2_output)
    print("      Layer 2 output sum:", layer_2_sum)
    print("")

    // ========================================================================
    // Layer 3: Attention
    // ========================================================================
    let x_norm_l3 = rms_norm(layer_2_output, attn_norm_3)
    print("      [DEBUG] Layer 3 x_norm sum:", sum(x_norm_l3))

    let Q_l3 = linear(x_norm_l3, W_q_3)
    let K_l3 = linear(x_norm_l3, W_k_3)
    let V_l3 = linear(x_norm_l3, W_v_3)

    let Q_heads_l3 = reshape(Q_l3, [seq_len, 32.0, 64.0])
    let K_heads_l3 = reshape(K_l3, [seq_len, 4.0, 64.0])
    let V_heads_l3 = reshape(V_l3, [seq_len, 4.0, 64.0])

    let Q_rope_l3 = rope(Q_heads_l3)
    let K_rope_l3 = rope(K_heads_l3)

    // GQA expansion
    let K_group_l3 = reshape(K_rope_l3, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l3 = reshape(V_heads_l3, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l3 = broadcast_to(K_group_l3, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l3 = broadcast_to(V_group_l3, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l3 = reshape(K_broadcast_l3, [seq_len, 32.0, 64.0])
    let V_expanded_l3 = reshape(V_broadcast_l3, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l3 = einsum("ihd,jhd->ihj", Q_rope_l3, K_expanded_l3)
    let scaled_scores_l3 = scores_l3 * 0.125  // 1/sqrt(64)
    let attn_weights_l3 = softmax(scaled_scores_l3, 2)
    let attn_output_l3 = einsum("ihj,jhd->ihd", attn_weights_l3, V_expanded_l3)

    // Attention output projection
    let attn_reshaped_l3 = reshape(attn_output_l3, [seq_len, 2048.0])
    let attn_out_l3 = linear(attn_reshaped_l3, W_o_3)
    print("      [DEBUG] Layer 3 attn_out sum:", sum(attn_out_l3))

    // Residual
    let hidden_l3 = layer_2_output + attn_out_l3
    print("      [DEBUG] Layer 3 hidden (after attn) sum:", sum(hidden_l3))

    // Layer 3: FFN
    let x_norm2_l3 = rms_norm(hidden_l3, ffn_norm_3)
    print("      [DEBUG] Layer 3 x_norm2 sum:", sum(x_norm2_l3))

    let gate_l3 = linear(x_norm2_l3, W_gate_3)
    let up_l3 = linear(x_norm2_l3, W_up_3)
    let silu_gate_l3 = silu(gate_l3)
    let gated_l3 = silu_gate_l3 * up_l3
    let ffn_out_l3 = linear(gated_l3, W_down_3)

    print("      [DEBUG] Layer 3 ffn_out sum:", sum(ffn_out_l3))

    // Final residual
    let layer_3_output = hidden_l3 + ffn_out_l3

    let layer_3_sum = sum(layer_3_output)
    print("      Layer 3 output sum:", layer_3_sum)
    print("")

    // Final norm
    let final_norm = rms_norm(layer_3_output, output_norm)
    let final_norm_sum = sum(final_norm)
    print("      Final norm sum:", final_norm_sum)

    // Logits
    let logits = linear(final_norm, output_weight)
    let logits_sum = sum(logits)
    print("      Logits sum (all tokens):", logits_sum)

    // Extract last token logits for comparison with Candle
    let last_row = 37.0  // Last token index (38 tokens, 0-indexed)
    let last_token_logits = slice(logits, last_row, 0.0, 32000.0)
    let last_token_sum = sum(last_token_logits)
    print("      Last token logits sum:", last_token_sum)

    // Sample from logits (temperature_sample shows top logits in debug mode)
    let temperature = 0.0  // Greedy decoding
    let predicted_token = temperature_sample(logits, temperature)

    print("")
    print("================================================================================")
    print("4-Layer Test Summary:")
    print("  Embedding sum:", emb_sum)
    print("  Layer 0 output sum:", layer_0_sum)
    print("  Layer 1 output sum:", layer_1_sum)
    print("  Layer 2 output sum:", layer_2_sum)
    print("  Layer 3 output sum:", layer_3_sum)
    print("  Final norm sum:", final_norm_sum)
    print("  Last token logits sum:", last_token_sum)
    print("  Predicted token:", predicted_token)
    print("================================================================================")
    print("")

    print("================================================================================")
    print("✅ 4-Layer Test Complete")
    print("================================================================================")
}
