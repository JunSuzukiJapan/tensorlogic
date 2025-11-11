// Test: 10-layer transformer for debugging
// Progressive testing approach

// SiLU activation function (inline version now works with fix)
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("TensorLogic 10-Layer Transformer Test")
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
    print("[2/5] Loading weights for Layers 0-9...")
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

    // Layer 4 weights
    let W_q_4 = get_tensor(model, "blk.4.attn_q.weight")
    let W_k_4 = get_tensor(model, "blk.4.attn_k.weight")
    let W_v_4 = get_tensor(model, "blk.4.attn_v.weight")
    let W_o_4 = get_tensor(model, "blk.4.attn_output.weight")
    let attn_norm_4 = get_tensor(model, "blk.4.attn_norm.weight")
    let W_gate_4 = get_tensor(model, "blk.4.ffn_gate.weight")
    let W_up_4 = get_tensor(model, "blk.4.ffn_up.weight")
    let W_down_4 = get_tensor(model, "blk.4.ffn_down.weight")
    let ffn_norm_4 = get_tensor(model, "blk.4.ffn_norm.weight")

    // Layer 5 weights
    let W_q_5 = get_tensor(model, "blk.5.attn_q.weight")
    let W_k_5 = get_tensor(model, "blk.5.attn_k.weight")
    let W_v_5 = get_tensor(model, "blk.5.attn_v.weight")
    let W_o_5 = get_tensor(model, "blk.5.attn_output.weight")
    let attn_norm_5 = get_tensor(model, "blk.5.attn_norm.weight")
    let W_gate_5 = get_tensor(model, "blk.5.ffn_gate.weight")
    let W_up_5 = get_tensor(model, "blk.5.ffn_up.weight")
    let W_down_5 = get_tensor(model, "blk.5.ffn_down.weight")
    let ffn_norm_5 = get_tensor(model, "blk.5.ffn_norm.weight")

    // Layer 6 weights
    let W_q_6 = get_tensor(model, "blk.6.attn_q.weight")
    let W_k_6 = get_tensor(model, "blk.6.attn_k.weight")
    let W_v_6 = get_tensor(model, "blk.6.attn_v.weight")
    let W_o_6 = get_tensor(model, "blk.6.attn_output.weight")
    let attn_norm_6 = get_tensor(model, "blk.6.attn_norm.weight")
    let W_gate_6 = get_tensor(model, "blk.6.ffn_gate.weight")
    let W_up_6 = get_tensor(model, "blk.6.ffn_up.weight")
    let W_down_6 = get_tensor(model, "blk.6.ffn_down.weight")
    let ffn_norm_6 = get_tensor(model, "blk.6.ffn_norm.weight")

    // Layer 7 weights
    let W_q_7 = get_tensor(model, "blk.7.attn_q.weight")
    let W_k_7 = get_tensor(model, "blk.7.attn_k.weight")
    let W_v_7 = get_tensor(model, "blk.7.attn_v.weight")
    let W_o_7 = get_tensor(model, "blk.7.attn_output.weight")
    let attn_norm_7 = get_tensor(model, "blk.7.attn_norm.weight")
    let W_gate_7 = get_tensor(model, "blk.7.ffn_gate.weight")
    let W_up_7 = get_tensor(model, "blk.7.ffn_up.weight")
    let W_down_7 = get_tensor(model, "blk.7.ffn_down.weight")
    let ffn_norm_7 = get_tensor(model, "blk.7.ffn_norm.weight")

    // Layer 8 weights
    let W_q_8 = get_tensor(model, "blk.8.attn_q.weight")
    let W_k_8 = get_tensor(model, "blk.8.attn_k.weight")
    let W_v_8 = get_tensor(model, "blk.8.attn_v.weight")
    let W_o_8 = get_tensor(model, "blk.8.attn_output.weight")
    let attn_norm_8 = get_tensor(model, "blk.8.attn_norm.weight")
    let W_gate_8 = get_tensor(model, "blk.8.ffn_gate.weight")
    let W_up_8 = get_tensor(model, "blk.8.ffn_up.weight")
    let W_down_8 = get_tensor(model, "blk.8.ffn_down.weight")
    let ffn_norm_8 = get_tensor(model, "blk.8.ffn_norm.weight")

    // Layer 9 weights
    let W_q_9 = get_tensor(model, "blk.9.attn_q.weight")
    let W_k_9 = get_tensor(model, "blk.9.attn_k.weight")
    let W_v_9 = get_tensor(model, "blk.9.attn_v.weight")
    let W_o_9 = get_tensor(model, "blk.9.attn_output.weight")
    let attn_norm_9 = get_tensor(model, "blk.9.attn_norm.weight")
    let W_gate_9 = get_tensor(model, "blk.9.ffn_gate.weight")
    let W_up_9 = get_tensor(model, "blk.9.ffn_up.weight")
    let W_down_9 = get_tensor(model, "blk.9.ffn_down.weight")
    let ffn_norm_9 = get_tensor(model, "blk.9.ffn_norm.weight")

    print("      ✓ Layers 0-9 weights loaded")
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
    print("[5/5] Running forward pass (Layers 0-9)...")
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

    // ========================================================================
    // Layer 4: Attention
    // ========================================================================
    let x_norm_l4 = rms_norm(layer_3_output, attn_norm_4)
    print("      [DEBUG] Layer 4 x_norm sum:", sum(x_norm_l4))

    let Q_l4 = linear(x_norm_l4, W_q_4)
    let K_l4 = linear(x_norm_l4, W_k_4)
    let V_l4 = linear(x_norm_l4, W_v_4)

    let Q_heads_l4 = reshape(Q_l4, [seq_len, 32.0, 64.0])
    let K_heads_l4 = reshape(K_l4, [seq_len, 4.0, 64.0])
    let V_heads_l4 = reshape(V_l4, [seq_len, 4.0, 64.0])

    let Q_rope_l4 = rope(Q_heads_l4)
    let K_rope_l4 = rope(K_heads_l4)

    // GQA expansion
    let K_group_l4 = reshape(K_rope_l4, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l4 = reshape(V_heads_l4, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l4 = broadcast_to(K_group_l4, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l4 = broadcast_to(V_group_l4, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l4 = reshape(K_broadcast_l4, [seq_len, 32.0, 64.0])
    let V_expanded_l4 = reshape(V_broadcast_l4, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l4 = einsum("ihd,jhd->ihj", Q_rope_l4, K_expanded_l4)
    let scaled_scores_l4 = scores_l4 * 0.125  // 1/sqrt(64)
    let attn_weights_l4 = softmax(scaled_scores_l4, 2)
    let attn_output_l4 = einsum("ihj,jhd->ihd", attn_weights_l4, V_expanded_l4)

    // Attention output projection
    let attn_reshaped_l4 = reshape(attn_output_l4, [seq_len, 2048.0])
    let attn_out_l4 = linear(attn_reshaped_l4, W_o_4)
    print("      [DEBUG] Layer 4 attn_out sum:", sum(attn_out_l4))

    // Residual
    let hidden_l4 = layer_3_output + attn_out_l4
    print("      [DEBUG] Layer 4 hidden (after attn) sum:", sum(hidden_l4))

    // Layer 4: FFN
    let x_norm2_l4 = rms_norm(hidden_l4, ffn_norm_4)
    print("      [DEBUG] Layer 4 x_norm2 sum:", sum(x_norm2_l4))

    let gate_l4 = linear(x_norm2_l4, W_gate_4)
    let up_l4 = linear(x_norm2_l4, W_up_4)
    let silu_gate_l4 = silu(gate_l4)
    let gated_l4 = silu_gate_l4 * up_l4
    let ffn_out_l4 = linear(gated_l4, W_down_4)

    print("      [DEBUG] Layer 4 ffn_out sum:", sum(ffn_out_l4))

    // Final residual
    let layer_4_output = hidden_l4 + ffn_out_l4

    let layer_4_sum = sum(layer_4_output)
    print("      Layer 4 output sum:", layer_4_sum)
    print("")


    // ========================================================================
    // Layer 5: Attention
    // ========================================================================
    let x_norm_l5 = rms_norm(layer_4_output, attn_norm_5)
    print("      [DEBUG] Layer 5 x_norm sum:", sum(x_norm_l5))

    let Q_l5 = linear(x_norm_l5, W_q_5)
    let K_l5 = linear(x_norm_l5, W_k_5)
    let V_l5 = linear(x_norm_l5, W_v_5)

    let Q_heads_l5 = reshape(Q_l5, [seq_len, 32.0, 64.0])
    let K_heads_l5 = reshape(K_l5, [seq_len, 4.0, 64.0])
    let V_heads_l5 = reshape(V_l5, [seq_len, 4.0, 64.0])

    let Q_rope_l5 = rope(Q_heads_l5)
    let K_rope_l5 = rope(K_heads_l5)

    // GQA expansion
    let K_group_l5 = reshape(K_rope_l5, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l5 = reshape(V_heads_l5, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l5 = broadcast_to(K_group_l5, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l5 = broadcast_to(V_group_l5, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l5 = reshape(K_broadcast_l5, [seq_len, 32.0, 64.0])
    let V_expanded_l5 = reshape(V_broadcast_l5, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l5 = einsum("ihd,jhd->ihj", Q_rope_l5, K_expanded_l5)
    let scaled_scores_l5 = scores_l5 * 0.125  // 1/sqrt(64)
    let attn_weights_l5 = softmax(scaled_scores_l5, 2)
    let attn_output_l5 = einsum("ihj,jhd->ihd", attn_weights_l5, V_expanded_l5)

    // Attention output projection
    let attn_reshaped_l5 = reshape(attn_output_l5, [seq_len, 2048.0])
    let attn_out_l5 = linear(attn_reshaped_l5, W_o_5)
    print("      [DEBUG] Layer 5 attn_out sum:", sum(attn_out_l5))

    // Residual
    let hidden_l5 = layer_4_output + attn_out_l5
    print("      [DEBUG] Layer 5 hidden (after attn) sum:", sum(hidden_l5))

    // Layer 5: FFN
    let x_norm2_l5 = rms_norm(hidden_l5, ffn_norm_5)
    print("      [DEBUG] Layer 5 x_norm2 sum:", sum(x_norm2_l5))

    let gate_l5 = linear(x_norm2_l5, W_gate_5)
    let up_l5 = linear(x_norm2_l5, W_up_5)
    let silu_gate_l5 = silu(gate_l5)
    let gated_l5 = silu_gate_l5 * up_l5
    let ffn_out_l5 = linear(gated_l5, W_down_5)

    print("      [DEBUG] Layer 5 ffn_out sum:", sum(ffn_out_l5))

    // Final residual
    let layer_5_output = hidden_l5 + ffn_out_l5

    let layer_5_sum = sum(layer_5_output)
    print("      Layer 5 output sum:", layer_5_sum)
    print("")

    // ========================================================================
    // Layer 6: Attention
    // ========================================================================
    let x_norm_l6 = rms_norm(layer_5_output, attn_norm_6)
    print("      [DEBUG] Layer 6 x_norm sum:", sum(x_norm_l6))

    let Q_l6 = linear(x_norm_l6, W_q_6)
    let K_l6 = linear(x_norm_l6, W_k_6)
    let V_l6 = linear(x_norm_l6, W_v_6)

    let Q_heads_l6 = reshape(Q_l6, [seq_len, 32.0, 64.0])
    let K_heads_l6 = reshape(K_l6, [seq_len, 4.0, 64.0])
    let V_heads_l6 = reshape(V_l6, [seq_len, 4.0, 64.0])

    let Q_rope_l6 = rope(Q_heads_l6)
    let K_rope_l6 = rope(K_heads_l6)

    // GQA expansion
    let K_group_l6 = reshape(K_rope_l6, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l6 = reshape(V_heads_l6, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l6 = broadcast_to(K_group_l6, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l6 = broadcast_to(V_group_l6, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l6 = reshape(K_broadcast_l6, [seq_len, 32.0, 64.0])
    let V_expanded_l6 = reshape(V_broadcast_l6, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l6 = einsum("ihd,jhd->ihj", Q_rope_l6, K_expanded_l6)
    let scaled_scores_l6 = scores_l6 * 0.125  // 1/sqrt(64)
    let attn_weights_l6 = softmax(scaled_scores_l6, 2)
    let attn_output_l6 = einsum("ihj,jhd->ihd", attn_weights_l6, V_expanded_l6)

    // Attention output projection
    let attn_reshaped_l6 = reshape(attn_output_l6, [seq_len, 2048.0])
    let attn_out_l6 = linear(attn_reshaped_l6, W_o_6)
    print("      [DEBUG] Layer 6 attn_out sum:", sum(attn_out_l6))

    // Residual
    let hidden_l6 = layer_5_output + attn_out_l6
    print("      [DEBUG] Layer 6 hidden (after attn) sum:", sum(hidden_l6))

    // Layer 6: FFN
    let x_norm2_l6 = rms_norm(hidden_l6, ffn_norm_6)
    print("      [DEBUG] Layer 6 x_norm2 sum:", sum(x_norm2_l6))

    let gate_l6 = linear(x_norm2_l6, W_gate_6)
    let up_l6 = linear(x_norm2_l6, W_up_6)
    let silu_gate_l6 = silu(gate_l6)
    let gated_l6 = silu_gate_l6 * up_l6
    let ffn_out_l6 = linear(gated_l6, W_down_6)

    print("      [DEBUG] Layer 6 ffn_out sum:", sum(ffn_out_l6))

    // Final residual
    let layer_6_output = hidden_l6 + ffn_out_l6

    let layer_6_sum = sum(layer_6_output)
    print("      Layer 6 output sum:", layer_6_sum)
    print("")

    // ========================================================================
    // Layer 7: Attention
    // ========================================================================
    let x_norm_l7 = rms_norm(layer_6_output, attn_norm_7)
    print("      [DEBUG] Layer 7 x_norm sum:", sum(x_norm_l7))

    let Q_l7 = linear(x_norm_l7, W_q_7)
    let K_l7 = linear(x_norm_l7, W_k_7)
    let V_l7 = linear(x_norm_l7, W_v_7)

    let Q_heads_l7 = reshape(Q_l7, [seq_len, 32.0, 64.0])
    let K_heads_l7 = reshape(K_l7, [seq_len, 4.0, 64.0])
    let V_heads_l7 = reshape(V_l7, [seq_len, 4.0, 64.0])

    let Q_rope_l7 = rope(Q_heads_l7)
    let K_rope_l7 = rope(K_heads_l7)

    // GQA expansion
    let K_group_l7 = reshape(K_rope_l7, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l7 = reshape(V_heads_l7, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l7 = broadcast_to(K_group_l7, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l7 = broadcast_to(V_group_l7, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l7 = reshape(K_broadcast_l7, [seq_len, 32.0, 64.0])
    let V_expanded_l7 = reshape(V_broadcast_l7, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l7 = einsum("ihd,jhd->ihj", Q_rope_l7, K_expanded_l7)
    let scaled_scores_l7 = scores_l7 * 0.125  // 1/sqrt(64)
    let attn_weights_l7 = softmax(scaled_scores_l7, 2)
    let attn_output_l7 = einsum("ihj,jhd->ihd", attn_weights_l7, V_expanded_l7)

    // Attention output projection
    let attn_reshaped_l7 = reshape(attn_output_l7, [seq_len, 2048.0])
    let attn_out_l7 = linear(attn_reshaped_l7, W_o_7)
    print("      [DEBUG] Layer 7 attn_out sum:", sum(attn_out_l7))

    // Residual
    let hidden_l7 = layer_6_output + attn_out_l7
    print("      [DEBUG] Layer 7 hidden (after attn) sum:", sum(hidden_l7))

    // Layer 7: FFN
    let x_norm2_l7 = rms_norm(hidden_l7, ffn_norm_7)
    print("      [DEBUG] Layer 7 x_norm2 sum:", sum(x_norm2_l7))

    let gate_l7 = linear(x_norm2_l7, W_gate_7)
    let up_l7 = linear(x_norm2_l7, W_up_7)
    let silu_gate_l7 = silu(gate_l7)
    let gated_l7 = silu_gate_l7 * up_l7
    let ffn_out_l7 = linear(gated_l7, W_down_7)

    print("      [DEBUG] Layer 7 ffn_out sum:", sum(ffn_out_l7))

    // Final residual
    let layer_7_output = hidden_l7 + ffn_out_l7

    let layer_7_sum = sum(layer_7_output)
    print("      Layer 7 output sum:", layer_7_sum)
    print("")

    // ========================================================================
    // Layer 8: Attention
    // ========================================================================
    let x_norm_l8 = rms_norm(layer_7_output, attn_norm_8)
    print("      [DEBUG] Layer 8 x_norm sum:", sum(x_norm_l8))

    let Q_l8 = linear(x_norm_l8, W_q_8)
    let K_l8 = linear(x_norm_l8, W_k_8)
    let V_l8 = linear(x_norm_l8, W_v_8)

    let Q_heads_l8 = reshape(Q_l8, [seq_len, 32.0, 64.0])
    let K_heads_l8 = reshape(K_l8, [seq_len, 4.0, 64.0])
    let V_heads_l8 = reshape(V_l8, [seq_len, 4.0, 64.0])

    let Q_rope_l8 = rope(Q_heads_l8)
    let K_rope_l8 = rope(K_heads_l8)

    // GQA expansion
    let K_group_l8 = reshape(K_rope_l8, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l8 = reshape(V_heads_l8, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l8 = broadcast_to(K_group_l8, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l8 = broadcast_to(V_group_l8, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l8 = reshape(K_broadcast_l8, [seq_len, 32.0, 64.0])
    let V_expanded_l8 = reshape(V_broadcast_l8, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l8 = einsum("ihd,jhd->ihj", Q_rope_l8, K_expanded_l8)
    let scaled_scores_l8 = scores_l8 * 0.125  // 1/sqrt(64)
    let attn_weights_l8 = softmax(scaled_scores_l8, 2)
    let attn_output_l8 = einsum("ihj,jhd->ihd", attn_weights_l8, V_expanded_l8)

    // Attention output projection
    let attn_reshaped_l8 = reshape(attn_output_l8, [seq_len, 2048.0])
    let attn_out_l8 = linear(attn_reshaped_l8, W_o_8)
    print("      [DEBUG] Layer 8 attn_out sum:", sum(attn_out_l8))

    // Residual
    let hidden_l8 = layer_7_output + attn_out_l8
    print("      [DEBUG] Layer 8 hidden (after attn) sum:", sum(hidden_l8))

    // Layer 8: FFN
    let x_norm2_l8 = rms_norm(hidden_l8, ffn_norm_8)
    print("      [DEBUG] Layer 8 x_norm2 sum:", sum(x_norm2_l8))

    let gate_l8 = linear(x_norm2_l8, W_gate_8)
    let up_l8 = linear(x_norm2_l8, W_up_8)
    let silu_gate_l8 = silu(gate_l8)
    let gated_l8 = silu_gate_l8 * up_l8
    let ffn_out_l8 = linear(gated_l8, W_down_8)

    print("      [DEBUG] Layer 8 ffn_out sum:", sum(ffn_out_l8))

    // Final residual
    let layer_8_output = hidden_l8 + ffn_out_l8

    let layer_8_sum = sum(layer_8_output)
    print("      Layer 8 output sum:", layer_8_sum)
    print("")

    // ========================================================================
    // Layer 9: Attention
    // ========================================================================
    let x_norm_l9 = rms_norm(layer_8_output, attn_norm_9)
    print("      [DEBUG] Layer 9 x_norm sum:", sum(x_norm_l9))

    let Q_l9 = linear(x_norm_l9, W_q_9)
    let K_l9 = linear(x_norm_l9, W_k_9)
    let V_l9 = linear(x_norm_l9, W_v_9)

    let Q_heads_l9 = reshape(Q_l9, [seq_len, 32.0, 64.0])
    let K_heads_l9 = reshape(K_l9, [seq_len, 4.0, 64.0])
    let V_heads_l9 = reshape(V_l9, [seq_len, 4.0, 64.0])

    let Q_rope_l9 = rope(Q_heads_l9)
    let K_rope_l9 = rope(K_heads_l9)

    // GQA expansion
    let K_group_l9 = reshape(K_rope_l9, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l9 = reshape(V_heads_l9, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l9 = broadcast_to(K_group_l9, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l9 = broadcast_to(V_group_l9, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l9 = reshape(K_broadcast_l9, [seq_len, 32.0, 64.0])
    let V_expanded_l9 = reshape(V_broadcast_l9, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l9 = einsum("ihd,jhd->ihj", Q_rope_l9, K_expanded_l9)
    let scaled_scores_l9 = scores_l9 * 0.125  // 1/sqrt(64)
    let attn_weights_l9 = softmax(scaled_scores_l9, 2)
    let attn_output_l9 = einsum("ihj,jhd->ihd", attn_weights_l9, V_expanded_l9)

    // Attention output projection
    let attn_reshaped_l9 = reshape(attn_output_l9, [seq_len, 2048.0])
    let attn_out_l9 = linear(attn_reshaped_l9, W_o_9)
    print("      [DEBUG] Layer 9 attn_out sum:", sum(attn_out_l9))

    // Residual
    let hidden_l9 = layer_8_output + attn_out_l9
    print("      [DEBUG] Layer 9 hidden (after attn) sum:", sum(hidden_l9))

    // Layer 9: FFN
    let x_norm2_l9 = rms_norm(hidden_l9, ffn_norm_9)
    print("      [DEBUG] Layer 9 x_norm2 sum:", sum(x_norm2_l9))

    let gate_l9 = linear(x_norm2_l9, W_gate_9)
    let up_l9 = linear(x_norm2_l9, W_up_9)
    let silu_gate_l9 = silu(gate_l9)
    let gated_l9 = silu_gate_l9 * up_l9
    let ffn_out_l9 = linear(gated_l9, W_down_9)

    print("      [DEBUG] Layer 9 ffn_out sum:", sum(ffn_out_l9))

    // Final residual
    let layer_9_output = hidden_l9 + ffn_out_l9

    let layer_9_sum = sum(layer_9_output)
    print("      Layer 9 output sum:", layer_9_sum)
    print("")
    // Final norm
    let final_norm = rms_norm(layer_9_output, output_norm)
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
    print("10-Layer Test Summary:")
    print("  Embedding sum:", emb_sum)
    print("  Layer 0 output sum:", layer_0_sum)
    print("  Layer 1 output sum:", layer_1_sum)
    print("  Layer 2 output sum:", layer_2_sum)
    print("  Layer 3 output sum:", layer_3_sum)
    print("  Layer 4 output sum:", layer_4_sum)
    print("  Layer 5 output sum:", layer_5_sum)
    print("  Layer 6 output sum:", layer_6_sum)
    print("  Layer 7 output sum:", layer_7_sum)
    print("  Layer 8 output sum:", layer_8_sum)
    print("  Layer 9 output sum:", layer_9_sum)
    print("  Final norm sum:", final_norm_sum)
    print("  Last token logits sum:", last_token_sum)
    print("  Predicted token:", predicted_token)
    print("================================================================================")
    print("")

    print("================================================================================")
    print("✅ 10-Layer Test Complete")
    print("================================================================================")
}
