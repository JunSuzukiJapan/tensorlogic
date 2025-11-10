// Test: 22-layer transformer for debugging
// Progressive testing approach

// SiLU activation function (inline version now works with fix)
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("TensorLogic 22-Layer Transformer Test (with Causal Mask)")
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
    print("[2/5] Loading weights for Layers 0-21...")
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

    // Layer 10 weights
    let W_q_10 = get_tensor(model, "blk.10.attn_q.weight")
    let W_k_10 = get_tensor(model, "blk.10.attn_k.weight")
    let W_v_10 = get_tensor(model, "blk.10.attn_v.weight")
    let W_o_10 = get_tensor(model, "blk.10.attn_output.weight")
    let attn_norm_10 = get_tensor(model, "blk.10.attn_norm.weight")
    let W_gate_10 = get_tensor(model, "blk.10.ffn_gate.weight")
    let W_up_10 = get_tensor(model, "blk.10.ffn_up.weight")
    let W_down_10 = get_tensor(model, "blk.10.ffn_down.weight")
    let ffn_norm_10 = get_tensor(model, "blk.10.ffn_norm.weight")

    // Layer 11 weights
    let W_q_11 = get_tensor(model, "blk.11.attn_q.weight")
    let W_k_11 = get_tensor(model, "blk.11.attn_k.weight")
    let W_v_11 = get_tensor(model, "blk.11.attn_v.weight")
    let W_o_11 = get_tensor(model, "blk.11.attn_output.weight")
    let attn_norm_11 = get_tensor(model, "blk.11.attn_norm.weight")
    let W_gate_11 = get_tensor(model, "blk.11.ffn_gate.weight")
    let W_up_11 = get_tensor(model, "blk.11.ffn_up.weight")
    let W_down_11 = get_tensor(model, "blk.11.ffn_down.weight")
    let ffn_norm_11 = get_tensor(model, "blk.11.ffn_norm.weight")

    // Layer 12 weights
    let W_q_12 = get_tensor(model, "blk.12.attn_q.weight")
    let W_k_12 = get_tensor(model, "blk.12.attn_k.weight")
    let W_v_12 = get_tensor(model, "blk.12.attn_v.weight")
    let W_o_12 = get_tensor(model, "blk.12.attn_output.weight")
    let attn_norm_12 = get_tensor(model, "blk.12.attn_norm.weight")
    let W_gate_12 = get_tensor(model, "blk.12.ffn_gate.weight")
    let W_up_12 = get_tensor(model, "blk.12.ffn_up.weight")
    let W_down_12 = get_tensor(model, "blk.12.ffn_down.weight")
    let ffn_norm_12 = get_tensor(model, "blk.12.ffn_norm.weight")

    // Layer 13 weights
    let W_q_13 = get_tensor(model, "blk.13.attn_q.weight")
    let W_k_13 = get_tensor(model, "blk.13.attn_k.weight")
    let W_v_13 = get_tensor(model, "blk.13.attn_v.weight")
    let W_o_13 = get_tensor(model, "blk.13.attn_output.weight")
    let attn_norm_13 = get_tensor(model, "blk.13.attn_norm.weight")
    let W_gate_13 = get_tensor(model, "blk.13.ffn_gate.weight")
    let W_up_13 = get_tensor(model, "blk.13.ffn_up.weight")
    let W_down_13 = get_tensor(model, "blk.13.ffn_down.weight")
    let ffn_norm_13 = get_tensor(model, "blk.13.ffn_norm.weight")

    // Layer 14 weights
    let W_q_14 = get_tensor(model, "blk.14.attn_q.weight")
    let W_k_14 = get_tensor(model, "blk.14.attn_k.weight")
    let W_v_14 = get_tensor(model, "blk.14.attn_v.weight")
    let W_o_14 = get_tensor(model, "blk.14.attn_output.weight")
    let attn_norm_14 = get_tensor(model, "blk.14.attn_norm.weight")
    let W_gate_14 = get_tensor(model, "blk.14.ffn_gate.weight")
    let W_up_14 = get_tensor(model, "blk.14.ffn_up.weight")
    let W_down_14 = get_tensor(model, "blk.14.ffn_down.weight")
    let ffn_norm_14 = get_tensor(model, "blk.14.ffn_norm.weight")

    // Layer 15 weights
    let W_q_15 = get_tensor(model, "blk.15.attn_q.weight")
    let W_k_15 = get_tensor(model, "blk.15.attn_k.weight")
    let W_v_15 = get_tensor(model, "blk.15.attn_v.weight")
    let W_o_15 = get_tensor(model, "blk.15.attn_output.weight")
    let attn_norm_15 = get_tensor(model, "blk.15.attn_norm.weight")
    let W_gate_15 = get_tensor(model, "blk.15.ffn_gate.weight")
    let W_up_15 = get_tensor(model, "blk.15.ffn_up.weight")
    let W_down_15 = get_tensor(model, "blk.15.ffn_down.weight")
    let ffn_norm_15 = get_tensor(model, "blk.15.ffn_norm.weight")

    // Layer 16 weights
    let W_q_16 = get_tensor(model, "blk.16.attn_q.weight")
    let W_k_16 = get_tensor(model, "blk.16.attn_k.weight")
    let W_v_16 = get_tensor(model, "blk.16.attn_v.weight")
    let W_o_16 = get_tensor(model, "blk.16.attn_output.weight")
    let attn_norm_16 = get_tensor(model, "blk.16.attn_norm.weight")
    let W_gate_16 = get_tensor(model, "blk.16.ffn_gate.weight")
    let W_up_16 = get_tensor(model, "blk.16.ffn_up.weight")
    let W_down_16 = get_tensor(model, "blk.16.ffn_down.weight")
    let ffn_norm_16 = get_tensor(model, "blk.16.ffn_norm.weight")

    // Layer 17 weights
    let W_q_17 = get_tensor(model, "blk.17.attn_q.weight")
    let W_k_17 = get_tensor(model, "blk.17.attn_k.weight")
    let W_v_17 = get_tensor(model, "blk.17.attn_v.weight")
    let W_o_17 = get_tensor(model, "blk.17.attn_output.weight")
    let attn_norm_17 = get_tensor(model, "blk.17.attn_norm.weight")
    let W_gate_17 = get_tensor(model, "blk.17.ffn_gate.weight")
    let W_up_17 = get_tensor(model, "blk.17.ffn_up.weight")
    let W_down_17 = get_tensor(model, "blk.17.ffn_down.weight")
    let ffn_norm_17 = get_tensor(model, "blk.17.ffn_norm.weight")

    // Layer 18 weights
    let W_q_18 = get_tensor(model, "blk.18.attn_q.weight")
    let W_k_18 = get_tensor(model, "blk.18.attn_k.weight")
    let W_v_18 = get_tensor(model, "blk.18.attn_v.weight")
    let W_o_18 = get_tensor(model, "blk.18.attn_output.weight")
    let attn_norm_18 = get_tensor(model, "blk.18.attn_norm.weight")
    let W_gate_18 = get_tensor(model, "blk.18.ffn_gate.weight")
    let W_up_18 = get_tensor(model, "blk.18.ffn_up.weight")
    let W_down_18 = get_tensor(model, "blk.18.ffn_down.weight")
    let ffn_norm_18 = get_tensor(model, "blk.18.ffn_norm.weight")

    // Layer 19 weights
    let W_q_19 = get_tensor(model, "blk.19.attn_q.weight")
    let W_k_19 = get_tensor(model, "blk.19.attn_k.weight")
    let W_v_19 = get_tensor(model, "blk.19.attn_v.weight")
    let W_o_19 = get_tensor(model, "blk.19.attn_output.weight")
    let attn_norm_19 = get_tensor(model, "blk.19.attn_norm.weight")
    let W_gate_19 = get_tensor(model, "blk.19.ffn_gate.weight")
    let W_up_19 = get_tensor(model, "blk.19.ffn_up.weight")
    let W_down_19 = get_tensor(model, "blk.19.ffn_down.weight")
    let ffn_norm_19 = get_tensor(model, "blk.19.ffn_norm.weight")

    // Layer 20 weights
    let W_q_20 = get_tensor(model, "blk.20.attn_q.weight")
    let W_k_20 = get_tensor(model, "blk.20.attn_k.weight")
    let W_v_20 = get_tensor(model, "blk.20.attn_v.weight")
    let W_o_20 = get_tensor(model, "blk.20.attn_output.weight")
    let attn_norm_20 = get_tensor(model, "blk.20.attn_norm.weight")
    let W_gate_20 = get_tensor(model, "blk.20.ffn_gate.weight")
    let W_up_20 = get_tensor(model, "blk.20.ffn_up.weight")
    let W_down_20 = get_tensor(model, "blk.20.ffn_down.weight")
    let ffn_norm_20 = get_tensor(model, "blk.20.ffn_norm.weight")

    // Layer 21 weights
    let W_q_21 = get_tensor(model, "blk.21.attn_q.weight")
    let W_k_21 = get_tensor(model, "blk.21.attn_k.weight")
    let W_v_21 = get_tensor(model, "blk.21.attn_v.weight")
    let W_o_21 = get_tensor(model, "blk.21.attn_output.weight")
    let attn_norm_21 = get_tensor(model, "blk.21.attn_norm.weight")
    let W_gate_21 = get_tensor(model, "blk.21.ffn_gate.weight")
    let W_up_21 = get_tensor(model, "blk.21.ffn_up.weight")
    let W_down_21 = get_tensor(model, "blk.21.ffn_down.weight")
    let ffn_norm_21 = get_tensor(model, "blk.21.ffn_norm.weight")

    print("      ✓ Layers 0-21 weights loaded")
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
    print("[5/5] Running forward pass (Layers 0-21)...")
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

    // Attention computation with causal masking
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)

    // Apply causal mask to prevent attending to future tokens
    let seq_len_int = to_int(seq_len)
    let mask_2d = causal_mask(seq_len_int)  // [seq_len, seq_len]
    let mask_shape_3d = [seq_len, 1.0, seq_len]
    let mask_3d = reshape(mask_2d, mask_shape_3d)  // [seq_len, 1, seq_len]
    let mask_broadcast_shape = [seq_len, 32.0, seq_len]
    let mask = broadcast_to(mask_3d, mask_broadcast_shape)  // [seq_len, 32, seq_len]
    let masked_scores = apply_attention_mask(scaled_scores, mask)

    let attn_weights = softmax(masked_scores, 2)
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

    // ========================================================================
    // Layer 10: Attention
    // ========================================================================
    let x_norm_l10 = rms_norm(layer_9_output, attn_norm_10)
    print("      [DEBUG] Layer 10 x_norm sum:", sum(x_norm_l10))

    let Q_l10 = linear(x_norm_l10, W_q_10)
    let K_l10 = linear(x_norm_l10, W_k_10)
    let V_l10 = linear(x_norm_l10, W_v_10)

    let Q_heads_l10 = reshape(Q_l10, [seq_len, 32.0, 64.0])
    let K_heads_l10 = reshape(K_l10, [seq_len, 4.0, 64.0])
    let V_heads_l10 = reshape(V_l10, [seq_len, 4.0, 64.0])

    let Q_rope_l10 = rope(Q_heads_l10)
    let K_rope_l10 = rope(K_heads_l10)

    // GQA expansion
    let K_group_l10 = reshape(K_rope_l10, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l10 = reshape(V_heads_l10, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l10 = broadcast_to(K_group_l10, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l10 = broadcast_to(V_group_l10, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l10 = reshape(K_broadcast_l10, [seq_len, 32.0, 64.0])
    let V_expanded_l10 = reshape(V_broadcast_l10, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l10 = einsum("ihd,jhd->ihj", Q_rope_l10, K_expanded_l10)
    let scaled_scores_l10 = scores_l10 * 0.125  // 1/sqrt(64)
    let attn_weights_l10 = softmax(scaled_scores_l10, 2)
    let attn_output_l10 = einsum("ihj,jhd->ihd", attn_weights_l10, V_expanded_l10)

    // Attention output projection
    let attn_reshaped_l10 = reshape(attn_output_l10, [seq_len, 2048.0])
    let attn_out_l10 = linear(attn_reshaped_l10, W_o_10)
    print("      [DEBUG] Layer 10 attn_out sum:", sum(attn_out_l10))

    // Residual
    let hidden_l10 = layer_9_output + attn_out_l10
    print("      [DEBUG] Layer 10 hidden (after attn) sum:", sum(hidden_l10))

    // Layer 10: FFN
    let x_norm2_l10 = rms_norm(hidden_l10, ffn_norm_10)
    print("      [DEBUG] Layer 10 x_norm2 sum:", sum(x_norm2_l10))

    let gate_l10 = linear(x_norm2_l10, W_gate_10)
    let up_l10 = linear(x_norm2_l10, W_up_10)
    let silu_gate_l10 = silu(gate_l10)
    let gated_l10 = silu_gate_l10 * up_l10
    let ffn_out_l10 = linear(gated_l10, W_down_10)

    print("      [DEBUG] Layer 10 ffn_out sum:", sum(ffn_out_l10))

    // Final residual
    let layer_10_output = hidden_l10 + ffn_out_l10

    let layer_10_sum = sum(layer_10_output)
    print("      Layer 10 output sum:", layer_10_sum)
    print("")

    // ========================================================================
    // Layer 11: Attention
    // ========================================================================
    let x_norm_l11 = rms_norm(layer_10_output, attn_norm_11)
    print("      [DEBUG] Layer 11 x_norm sum:", sum(x_norm_l11))

    let Q_l11 = linear(x_norm_l11, W_q_11)
    let K_l11 = linear(x_norm_l11, W_k_11)
    let V_l11 = linear(x_norm_l11, W_v_11)

    let Q_heads_l11 = reshape(Q_l11, [seq_len, 32.0, 64.0])
    let K_heads_l11 = reshape(K_l11, [seq_len, 4.0, 64.0])
    let V_heads_l11 = reshape(V_l11, [seq_len, 4.0, 64.0])

    let Q_rope_l11 = rope(Q_heads_l11)
    let K_rope_l11 = rope(K_heads_l11)

    // GQA expansion
    let K_group_l11 = reshape(K_rope_l11, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l11 = reshape(V_heads_l11, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l11 = broadcast_to(K_group_l11, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l11 = broadcast_to(V_group_l11, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l11 = reshape(K_broadcast_l11, [seq_len, 32.0, 64.0])
    let V_expanded_l11 = reshape(V_broadcast_l11, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l11 = einsum("ihd,jhd->ihj", Q_rope_l11, K_expanded_l11)
    let scaled_scores_l11 = scores_l11 * 0.125  // 1/sqrt(64)
    let attn_weights_l11 = softmax(scaled_scores_l11, 2)
    let attn_output_l11 = einsum("ihj,jhd->ihd", attn_weights_l11, V_expanded_l11)

    // Attention output projection
    let attn_reshaped_l11 = reshape(attn_output_l11, [seq_len, 2048.0])
    let attn_out_l11 = linear(attn_reshaped_l11, W_o_11)
    print("      [DEBUG] Layer 11 attn_out sum:", sum(attn_out_l11))

    // Residual
    let hidden_l11 = layer_10_output + attn_out_l11
    print("      [DEBUG] Layer 11 hidden (after attn) sum:", sum(hidden_l11))

    // Layer 11: FFN
    let x_norm2_l11 = rms_norm(hidden_l11, ffn_norm_11)
    print("      [DEBUG] Layer 11 x_norm2 sum:", sum(x_norm2_l11))

    let gate_l11 = linear(x_norm2_l11, W_gate_11)
    let up_l11 = linear(x_norm2_l11, W_up_11)
    let silu_gate_l11 = silu(gate_l11)
    let gated_l11 = silu_gate_l11 * up_l11
    let ffn_out_l11 = linear(gated_l11, W_down_11)

    print("      [DEBUG] Layer 11 ffn_out sum:", sum(ffn_out_l11))

    // Final residual
    let layer_11_output = hidden_l11 + ffn_out_l11

    let layer_11_sum = sum(layer_11_output)
    print("      Layer 11 output sum:", layer_11_sum)
    print("")

    // ========================================================================
    // Layer 12: Attention
    // ========================================================================
    let x_norm_l12 = rms_norm(layer_11_output, attn_norm_12)
    print("      [DEBUG] Layer 12 x_norm sum:", sum(x_norm_l12))

    let Q_l12 = linear(x_norm_l12, W_q_12)
    let K_l12 = linear(x_norm_l12, W_k_12)
    let V_l12 = linear(x_norm_l12, W_v_12)

    let Q_heads_l12 = reshape(Q_l12, [seq_len, 32.0, 64.0])
    let K_heads_l12 = reshape(K_l12, [seq_len, 4.0, 64.0])
    let V_heads_l12 = reshape(V_l12, [seq_len, 4.0, 64.0])

    let Q_rope_l12 = rope(Q_heads_l12)
    let K_rope_l12 = rope(K_heads_l12)

    // GQA expansion
    let K_group_l12 = reshape(K_rope_l12, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l12 = reshape(V_heads_l12, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l12 = broadcast_to(K_group_l12, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l12 = broadcast_to(V_group_l12, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l12 = reshape(K_broadcast_l12, [seq_len, 32.0, 64.0])
    let V_expanded_l12 = reshape(V_broadcast_l12, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l12 = einsum("ihd,jhd->ihj", Q_rope_l12, K_expanded_l12)
    let scaled_scores_l12 = scores_l12 * 0.125  // 1/sqrt(64)
    let attn_weights_l12 = softmax(scaled_scores_l12, 2)
    let attn_output_l12 = einsum("ihj,jhd->ihd", attn_weights_l12, V_expanded_l12)

    // Attention output projection
    let attn_reshaped_l12 = reshape(attn_output_l12, [seq_len, 2048.0])
    let attn_out_l12 = linear(attn_reshaped_l12, W_o_12)
    print("      [DEBUG] Layer 12 attn_out sum:", sum(attn_out_l12))

    // Residual
    let hidden_l12 = layer_11_output + attn_out_l12
    print("      [DEBUG] Layer 12 hidden (after attn) sum:", sum(hidden_l12))

    // Layer 12: FFN
    let x_norm2_l12 = rms_norm(hidden_l12, ffn_norm_12)
    print("      [DEBUG] Layer 12 x_norm2 sum:", sum(x_norm2_l12))

    let gate_l12 = linear(x_norm2_l12, W_gate_12)
    let up_l12 = linear(x_norm2_l12, W_up_12)
    let silu_gate_l12 = silu(gate_l12)
    let gated_l12 = silu_gate_l12 * up_l12
    let ffn_out_l12 = linear(gated_l12, W_down_12)

    print("      [DEBUG] Layer 12 ffn_out sum:", sum(ffn_out_l12))

    // Final residual
    let layer_12_output = hidden_l12 + ffn_out_l12

    let layer_12_sum = sum(layer_12_output)
    print("      Layer 12 output sum:", layer_12_sum)
    print("")

    // ========================================================================
    // Layer 13: Attention
    // ========================================================================
    let x_norm_l13 = rms_norm(layer_12_output, attn_norm_13)
    print("      [DEBUG] Layer 13 x_norm sum:", sum(x_norm_l13))

    let Q_l13 = linear(x_norm_l13, W_q_13)
    let K_l13 = linear(x_norm_l13, W_k_13)
    let V_l13 = linear(x_norm_l13, W_v_13)

    let Q_heads_l13 = reshape(Q_l13, [seq_len, 32.0, 64.0])
    let K_heads_l13 = reshape(K_l13, [seq_len, 4.0, 64.0])
    let V_heads_l13 = reshape(V_l13, [seq_len, 4.0, 64.0])

    let Q_rope_l13 = rope(Q_heads_l13)
    let K_rope_l13 = rope(K_heads_l13)

    // GQA expansion
    let K_group_l13 = reshape(K_rope_l13, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l13 = reshape(V_heads_l13, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l13 = broadcast_to(K_group_l13, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l13 = broadcast_to(V_group_l13, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l13 = reshape(K_broadcast_l13, [seq_len, 32.0, 64.0])
    let V_expanded_l13 = reshape(V_broadcast_l13, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l13 = einsum("ihd,jhd->ihj", Q_rope_l13, K_expanded_l13)
    let scaled_scores_l13 = scores_l13 * 0.125  // 1/sqrt(64)
    let attn_weights_l13 = softmax(scaled_scores_l13, 2)
    let attn_output_l13 = einsum("ihj,jhd->ihd", attn_weights_l13, V_expanded_l13)

    // Attention output projection
    let attn_reshaped_l13 = reshape(attn_output_l13, [seq_len, 2048.0])
    let attn_out_l13 = linear(attn_reshaped_l13, W_o_13)
    print("      [DEBUG] Layer 13 attn_out sum:", sum(attn_out_l13))

    // Residual
    let hidden_l13 = layer_12_output + attn_out_l13
    print("      [DEBUG] Layer 13 hidden (after attn) sum:", sum(hidden_l13))

    // Layer 13: FFN
    let x_norm2_l13 = rms_norm(hidden_l13, ffn_norm_13)
    print("      [DEBUG] Layer 13 x_norm2 sum:", sum(x_norm2_l13))

    let gate_l13 = linear(x_norm2_l13, W_gate_13)
    let up_l13 = linear(x_norm2_l13, W_up_13)
    let silu_gate_l13 = silu(gate_l13)
    let gated_l13 = silu_gate_l13 * up_l13
    let ffn_out_l13 = linear(gated_l13, W_down_13)

    print("      [DEBUG] Layer 13 ffn_out sum:", sum(ffn_out_l13))

    // Final residual
    let layer_13_output = hidden_l13 + ffn_out_l13

    let layer_13_sum = sum(layer_13_output)
    print("      Layer 13 output sum:", layer_13_sum)
    print("")

    // ========================================================================
    // Layer 14: Attention
    // ========================================================================
    let x_norm_l14 = rms_norm(layer_13_output, attn_norm_14)
    print("      [DEBUG] Layer 14 x_norm sum:", sum(x_norm_l14))

    let Q_l14 = linear(x_norm_l14, W_q_14)
    let K_l14 = linear(x_norm_l14, W_k_14)
    let V_l14 = linear(x_norm_l14, W_v_14)

    let Q_heads_l14 = reshape(Q_l14, [seq_len, 32.0, 64.0])
    let K_heads_l14 = reshape(K_l14, [seq_len, 4.0, 64.0])
    let V_heads_l14 = reshape(V_l14, [seq_len, 4.0, 64.0])

    let Q_rope_l14 = rope(Q_heads_l14)
    let K_rope_l14 = rope(K_heads_l14)

    // GQA expansion
    let K_group_l14 = reshape(K_rope_l14, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l14 = reshape(V_heads_l14, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l14 = broadcast_to(K_group_l14, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l14 = broadcast_to(V_group_l14, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l14 = reshape(K_broadcast_l14, [seq_len, 32.0, 64.0])
    let V_expanded_l14 = reshape(V_broadcast_l14, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l14 = einsum("ihd,jhd->ihj", Q_rope_l14, K_expanded_l14)
    let scaled_scores_l14 = scores_l14 * 0.125  // 1/sqrt(64)
    let attn_weights_l14 = softmax(scaled_scores_l14, 2)
    let attn_output_l14 = einsum("ihj,jhd->ihd", attn_weights_l14, V_expanded_l14)

    // Attention output projection
    let attn_reshaped_l14 = reshape(attn_output_l14, [seq_len, 2048.0])
    let attn_out_l14 = linear(attn_reshaped_l14, W_o_14)
    print("      [DEBUG] Layer 14 attn_out sum:", sum(attn_out_l14))

    // Residual
    let hidden_l14 = layer_13_output + attn_out_l14
    print("      [DEBUG] Layer 14 hidden (after attn) sum:", sum(hidden_l14))

    // Layer 14: FFN
    let x_norm2_l14 = rms_norm(hidden_l14, ffn_norm_14)
    print("      [DEBUG] Layer 14 x_norm2 sum:", sum(x_norm2_l14))

    let gate_l14 = linear(x_norm2_l14, W_gate_14)
    let up_l14 = linear(x_norm2_l14, W_up_14)
    let silu_gate_l14 = silu(gate_l14)
    let gated_l14 = silu_gate_l14 * up_l14
    let ffn_out_l14 = linear(gated_l14, W_down_14)

    print("      [DEBUG] Layer 14 ffn_out sum:", sum(ffn_out_l14))

    // Final residual
    let layer_14_output = hidden_l14 + ffn_out_l14

    let layer_14_sum = sum(layer_14_output)
    print("      Layer 14 output sum:", layer_14_sum)
    print("")

    // ========================================================================
    // Layer 15: Attention
    // ========================================================================
    let x_norm_l15 = rms_norm(layer_14_output, attn_norm_15)
    print("      [DEBUG] Layer 15 x_norm sum:", sum(x_norm_l15))

    let Q_l15 = linear(x_norm_l15, W_q_15)
    let K_l15 = linear(x_norm_l15, W_k_15)
    let V_l15 = linear(x_norm_l15, W_v_15)

    let Q_heads_l15 = reshape(Q_l15, [seq_len, 32.0, 64.0])
    let K_heads_l15 = reshape(K_l15, [seq_len, 4.0, 64.0])
    let V_heads_l15 = reshape(V_l15, [seq_len, 4.0, 64.0])

    let Q_rope_l15 = rope(Q_heads_l15)
    let K_rope_l15 = rope(K_heads_l15)

    // GQA expansion
    let K_group_l15 = reshape(K_rope_l15, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l15 = reshape(V_heads_l15, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l15 = broadcast_to(K_group_l15, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l15 = broadcast_to(V_group_l15, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l15 = reshape(K_broadcast_l15, [seq_len, 32.0, 64.0])
    let V_expanded_l15 = reshape(V_broadcast_l15, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l15 = einsum("ihd,jhd->ihj", Q_rope_l15, K_expanded_l15)
    let scaled_scores_l15 = scores_l15 * 0.125  // 1/sqrt(64)
    let attn_weights_l15 = softmax(scaled_scores_l15, 2)
    let attn_output_l15 = einsum("ihj,jhd->ihd", attn_weights_l15, V_expanded_l15)

    // Attention output projection
    let attn_reshaped_l15 = reshape(attn_output_l15, [seq_len, 2048.0])
    let attn_out_l15 = linear(attn_reshaped_l15, W_o_15)
    print("      [DEBUG] Layer 15 attn_out sum:", sum(attn_out_l15))

    // Residual
    let hidden_l15 = layer_14_output + attn_out_l15
    print("      [DEBUG] Layer 15 hidden (after attn) sum:", sum(hidden_l15))

    // Layer 15: FFN
    let x_norm2_l15 = rms_norm(hidden_l15, ffn_norm_15)
    print("      [DEBUG] Layer 15 x_norm2 sum:", sum(x_norm2_l15))

    let gate_l15 = linear(x_norm2_l15, W_gate_15)
    let up_l15 = linear(x_norm2_l15, W_up_15)
    let silu_gate_l15 = silu(gate_l15)
    let gated_l15 = silu_gate_l15 * up_l15
    let ffn_out_l15 = linear(gated_l15, W_down_15)

    print("      [DEBUG] Layer 15 ffn_out sum:", sum(ffn_out_l15))

    // Final residual
    let layer_15_output = hidden_l15 + ffn_out_l15

    let layer_15_sum = sum(layer_15_output)
    print("      Layer 15 output sum:", layer_15_sum)
    print("")

    // ========================================================================
    // Layer 16: Attention
    // ========================================================================
    let x_norm_l16 = rms_norm(layer_15_output, attn_norm_16)
    print("      [DEBUG] Layer 16 x_norm sum:", sum(x_norm_l16))

    let Q_l16 = linear(x_norm_l16, W_q_16)
    let K_l16 = linear(x_norm_l16, W_k_16)
    let V_l16 = linear(x_norm_l16, W_v_16)

    let Q_heads_l16 = reshape(Q_l16, [seq_len, 32.0, 64.0])
    let K_heads_l16 = reshape(K_l16, [seq_len, 4.0, 64.0])
    let V_heads_l16 = reshape(V_l16, [seq_len, 4.0, 64.0])

    let Q_rope_l16 = rope(Q_heads_l16)
    let K_rope_l16 = rope(K_heads_l16)

    // GQA expansion
    let K_group_l16 = reshape(K_rope_l16, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l16 = reshape(V_heads_l16, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l16 = broadcast_to(K_group_l16, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l16 = broadcast_to(V_group_l16, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l16 = reshape(K_broadcast_l16, [seq_len, 32.0, 64.0])
    let V_expanded_l16 = reshape(V_broadcast_l16, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l16 = einsum("ihd,jhd->ihj", Q_rope_l16, K_expanded_l16)
    let scaled_scores_l16 = scores_l16 * 0.125  // 1/sqrt(64)
    let attn_weights_l16 = softmax(scaled_scores_l16, 2)
    let attn_output_l16 = einsum("ihj,jhd->ihd", attn_weights_l16, V_expanded_l16)

    // Attention output projection
    let attn_reshaped_l16 = reshape(attn_output_l16, [seq_len, 2048.0])
    let attn_out_l16 = linear(attn_reshaped_l16, W_o_16)
    print("      [DEBUG] Layer 16 attn_out sum:", sum(attn_out_l16))

    // Residual
    let hidden_l16 = layer_15_output + attn_out_l16
    print("      [DEBUG] Layer 16 hidden (after attn) sum:", sum(hidden_l16))

    // Layer 16: FFN
    let x_norm2_l16 = rms_norm(hidden_l16, ffn_norm_16)
    print("      [DEBUG] Layer 16 x_norm2 sum:", sum(x_norm2_l16))

    let gate_l16 = linear(x_norm2_l16, W_gate_16)
    let up_l16 = linear(x_norm2_l16, W_up_16)
    let silu_gate_l16 = silu(gate_l16)
    let gated_l16 = silu_gate_l16 * up_l16
    let ffn_out_l16 = linear(gated_l16, W_down_16)

    print("      [DEBUG] Layer 16 ffn_out sum:", sum(ffn_out_l16))

    // Final residual
    let layer_16_output = hidden_l16 + ffn_out_l16

    let layer_16_sum = sum(layer_16_output)
    print("      Layer 16 output sum:", layer_16_sum)
    print("")

    // ========================================================================
    // Layer 17: Attention
    // ========================================================================
    let x_norm_l17 = rms_norm(layer_16_output, attn_norm_17)
    print("      [DEBUG] Layer 17 x_norm sum:", sum(x_norm_l17))

    let Q_l17 = linear(x_norm_l17, W_q_17)
    let K_l17 = linear(x_norm_l17, W_k_17)
    let V_l17 = linear(x_norm_l17, W_v_17)

    let Q_heads_l17 = reshape(Q_l17, [seq_len, 32.0, 64.0])
    let K_heads_l17 = reshape(K_l17, [seq_len, 4.0, 64.0])
    let V_heads_l17 = reshape(V_l17, [seq_len, 4.0, 64.0])

    let Q_rope_l17 = rope(Q_heads_l17)
    let K_rope_l17 = rope(K_heads_l17)

    // GQA expansion
    let K_group_l17 = reshape(K_rope_l17, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l17 = reshape(V_heads_l17, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l17 = broadcast_to(K_group_l17, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l17 = broadcast_to(V_group_l17, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l17 = reshape(K_broadcast_l17, [seq_len, 32.0, 64.0])
    let V_expanded_l17 = reshape(V_broadcast_l17, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l17 = einsum("ihd,jhd->ihj", Q_rope_l17, K_expanded_l17)
    let scaled_scores_l17 = scores_l17 * 0.125  // 1/sqrt(64)
    let attn_weights_l17 = softmax(scaled_scores_l17, 2)
    let attn_output_l17 = einsum("ihj,jhd->ihd", attn_weights_l17, V_expanded_l17)

    // Attention output projection
    let attn_reshaped_l17 = reshape(attn_output_l17, [seq_len, 2048.0])
    let attn_out_l17 = linear(attn_reshaped_l17, W_o_17)
    print("      [DEBUG] Layer 17 attn_out sum:", sum(attn_out_l17))

    // Residual
    let hidden_l17 = layer_16_output + attn_out_l17
    print("      [DEBUG] Layer 17 hidden (after attn) sum:", sum(hidden_l17))

    // Layer 17: FFN
    let x_norm2_l17 = rms_norm(hidden_l17, ffn_norm_17)
    print("      [DEBUG] Layer 17 x_norm2 sum:", sum(x_norm2_l17))

    let gate_l17 = linear(x_norm2_l17, W_gate_17)
    let up_l17 = linear(x_norm2_l17, W_up_17)
    let silu_gate_l17 = silu(gate_l17)
    let gated_l17 = silu_gate_l17 * up_l17
    let ffn_out_l17 = linear(gated_l17, W_down_17)

    print("      [DEBUG] Layer 17 ffn_out sum:", sum(ffn_out_l17))

    // Final residual
    let layer_17_output = hidden_l17 + ffn_out_l17

    let layer_17_sum = sum(layer_17_output)
    print("      Layer 17 output sum:", layer_17_sum)
    print("")

    // ========================================================================
    // Layer 18: Attention
    // ========================================================================
    let x_norm_l18 = rms_norm(layer_17_output, attn_norm_18)
    print("      [DEBUG] Layer 18 x_norm sum:", sum(x_norm_l18))

    let Q_l18 = linear(x_norm_l18, W_q_18)
    let K_l18 = linear(x_norm_l18, W_k_18)
    let V_l18 = linear(x_norm_l18, W_v_18)

    let Q_heads_l18 = reshape(Q_l18, [seq_len, 32.0, 64.0])
    let K_heads_l18 = reshape(K_l18, [seq_len, 4.0, 64.0])
    let V_heads_l18 = reshape(V_l18, [seq_len, 4.0, 64.0])

    let Q_rope_l18 = rope(Q_heads_l18)
    let K_rope_l18 = rope(K_heads_l18)

    // GQA expansion
    let K_group_l18 = reshape(K_rope_l18, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l18 = reshape(V_heads_l18, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l18 = broadcast_to(K_group_l18, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l18 = broadcast_to(V_group_l18, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l18 = reshape(K_broadcast_l18, [seq_len, 32.0, 64.0])
    let V_expanded_l18 = reshape(V_broadcast_l18, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l18 = einsum("ihd,jhd->ihj", Q_rope_l18, K_expanded_l18)
    let scaled_scores_l18 = scores_l18 * 0.125  // 1/sqrt(64)
    let attn_weights_l18 = softmax(scaled_scores_l18, 2)
    let attn_output_l18 = einsum("ihj,jhd->ihd", attn_weights_l18, V_expanded_l18)

    // Attention output projection
    let attn_reshaped_l18 = reshape(attn_output_l18, [seq_len, 2048.0])
    let attn_out_l18 = linear(attn_reshaped_l18, W_o_18)
    print("      [DEBUG] Layer 18 attn_out sum:", sum(attn_out_l18))

    // Residual
    let hidden_l18 = layer_17_output + attn_out_l18
    print("      [DEBUG] Layer 18 hidden (after attn) sum:", sum(hidden_l18))

    // Layer 18: FFN
    let x_norm2_l18 = rms_norm(hidden_l18, ffn_norm_18)
    print("      [DEBUG] Layer 18 x_norm2 sum:", sum(x_norm2_l18))

    let gate_l18 = linear(x_norm2_l18, W_gate_18)
    let up_l18 = linear(x_norm2_l18, W_up_18)
    let silu_gate_l18 = silu(gate_l18)
    let gated_l18 = silu_gate_l18 * up_l18
    let ffn_out_l18 = linear(gated_l18, W_down_18)

    print("      [DEBUG] Layer 18 ffn_out sum:", sum(ffn_out_l18))

    // Final residual
    let layer_18_output = hidden_l18 + ffn_out_l18

    let layer_18_sum = sum(layer_18_output)
    print("      Layer 18 output sum:", layer_18_sum)
    print("")

    // ========================================================================
    // Layer 19: Attention
    // ========================================================================
    let x_norm_l19 = rms_norm(layer_18_output, attn_norm_19)
    print("      [DEBUG] Layer 19 x_norm sum:", sum(x_norm_l19))

    let Q_l19 = linear(x_norm_l19, W_q_19)
    let K_l19 = linear(x_norm_l19, W_k_19)
    let V_l19 = linear(x_norm_l19, W_v_19)

    let Q_heads_l19 = reshape(Q_l19, [seq_len, 32.0, 64.0])
    let K_heads_l19 = reshape(K_l19, [seq_len, 4.0, 64.0])
    let V_heads_l19 = reshape(V_l19, [seq_len, 4.0, 64.0])

    let Q_rope_l19 = rope(Q_heads_l19)
    let K_rope_l19 = rope(K_heads_l19)

    // GQA expansion
    let K_group_l19 = reshape(K_rope_l19, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l19 = reshape(V_heads_l19, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l19 = broadcast_to(K_group_l19, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l19 = broadcast_to(V_group_l19, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l19 = reshape(K_broadcast_l19, [seq_len, 32.0, 64.0])
    let V_expanded_l19 = reshape(V_broadcast_l19, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l19 = einsum("ihd,jhd->ihj", Q_rope_l19, K_expanded_l19)
    let scaled_scores_l19 = scores_l19 * 0.125  // 1/sqrt(64)
    let attn_weights_l19 = softmax(scaled_scores_l19, 2)
    let attn_output_l19 = einsum("ihj,jhd->ihd", attn_weights_l19, V_expanded_l19)

    // Attention output projection
    let attn_reshaped_l19 = reshape(attn_output_l19, [seq_len, 2048.0])
    let attn_out_l19 = linear(attn_reshaped_l19, W_o_19)
    print("      [DEBUG] Layer 19 attn_out sum:", sum(attn_out_l19))

    // Residual
    let hidden_l19 = layer_18_output + attn_out_l19
    print("      [DEBUG] Layer 19 hidden (after attn) sum:", sum(hidden_l19))

    // Layer 19: FFN
    let x_norm2_l19 = rms_norm(hidden_l19, ffn_norm_19)
    print("      [DEBUG] Layer 19 x_norm2 sum:", sum(x_norm2_l19))

    let gate_l19 = linear(x_norm2_l19, W_gate_19)
    let up_l19 = linear(x_norm2_l19, W_up_19)
    let silu_gate_l19 = silu(gate_l19)
    let gated_l19 = silu_gate_l19 * up_l19
    let ffn_out_l19 = linear(gated_l19, W_down_19)

    print("      [DEBUG] Layer 19 ffn_out sum:", sum(ffn_out_l19))

    // Final residual
    let layer_19_output = hidden_l19 + ffn_out_l19

    let layer_19_sum = sum(layer_19_output)
    print("      Layer 19 output sum:", layer_19_sum)
    print("")

    // ========================================================================
    // Layer 20: Attention
    // ========================================================================
    let x_norm_l20 = rms_norm(layer_19_output, attn_norm_20)
    print("      [DEBUG] Layer 20 x_norm sum:", sum(x_norm_l20))

    let Q_l20 = linear(x_norm_l20, W_q_20)
    let K_l20 = linear(x_norm_l20, W_k_20)
    let V_l20 = linear(x_norm_l20, W_v_20)

    let Q_heads_l20 = reshape(Q_l20, [seq_len, 32.0, 64.0])
    let K_heads_l20 = reshape(K_l20, [seq_len, 4.0, 64.0])
    let V_heads_l20 = reshape(V_l20, [seq_len, 4.0, 64.0])

    let Q_rope_l20 = rope(Q_heads_l20)
    let K_rope_l20 = rope(K_heads_l20)

    // GQA expansion
    let K_group_l20 = reshape(K_rope_l20, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l20 = reshape(V_heads_l20, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l20 = broadcast_to(K_group_l20, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l20 = broadcast_to(V_group_l20, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l20 = reshape(K_broadcast_l20, [seq_len, 32.0, 64.0])
    let V_expanded_l20 = reshape(V_broadcast_l20, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l20 = einsum("ihd,jhd->ihj", Q_rope_l20, K_expanded_l20)
    let scaled_scores_l20 = scores_l20 * 0.125  // 1/sqrt(64)
    let attn_weights_l20 = softmax(scaled_scores_l20, 2)
    let attn_output_l20 = einsum("ihj,jhd->ihd", attn_weights_l20, V_expanded_l20)

    // Attention output projection
    let attn_reshaped_l20 = reshape(attn_output_l20, [seq_len, 2048.0])
    let attn_out_l20 = linear(attn_reshaped_l20, W_o_20)
    print("      [DEBUG] Layer 20 attn_out sum:", sum(attn_out_l20))

    // Residual
    let hidden_l20 = layer_19_output + attn_out_l20
    print("      [DEBUG] Layer 20 hidden (after attn) sum:", sum(hidden_l20))

    // Layer 20: FFN
    let x_norm2_l20 = rms_norm(hidden_l20, ffn_norm_20)
    print("      [DEBUG] Layer 20 x_norm2 sum:", sum(x_norm2_l20))

    let gate_l20 = linear(x_norm2_l20, W_gate_20)
    let up_l20 = linear(x_norm2_l20, W_up_20)
    let silu_gate_l20 = silu(gate_l20)
    let gated_l20 = silu_gate_l20 * up_l20
    let ffn_out_l20 = linear(gated_l20, W_down_20)

    print("      [DEBUG] Layer 20 ffn_out sum:", sum(ffn_out_l20))

    // Final residual
    let layer_20_output = hidden_l20 + ffn_out_l20

    let layer_20_sum = sum(layer_20_output)
    print("      Layer 20 output sum:", layer_20_sum)
    print("")

    // ========================================================================
    // Layer 21: Attention
    // ========================================================================
    let x_norm_l21 = rms_norm(layer_20_output, attn_norm_21)
    print("      [DEBUG] Layer 21 x_norm sum:", sum(x_norm_l21))

    let Q_l21 = linear(x_norm_l21, W_q_21)
    let K_l21 = linear(x_norm_l21, W_k_21)
    let V_l21 = linear(x_norm_l21, W_v_21)

    let Q_heads_l21 = reshape(Q_l21, [seq_len, 32.0, 64.0])
    let K_heads_l21 = reshape(K_l21, [seq_len, 4.0, 64.0])
    let V_heads_l21 = reshape(V_l21, [seq_len, 4.0, 64.0])

    let Q_rope_l21 = rope(Q_heads_l21)
    let K_rope_l21 = rope(K_heads_l21)

    // GQA expansion
    let K_group_l21 = reshape(K_rope_l21, [seq_len, 4.0, 1.0, 64.0])
    let V_group_l21 = reshape(V_heads_l21, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast_l21 = broadcast_to(K_group_l21, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast_l21 = broadcast_to(V_group_l21, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded_l21 = reshape(K_broadcast_l21, [seq_len, 32.0, 64.0])
    let V_expanded_l21 = reshape(V_broadcast_l21, [seq_len, 32.0, 64.0])

    // Attention computation (no causal mask for reference comparison)
    let scores_l21 = einsum("ihd,jhd->ihj", Q_rope_l21, K_expanded_l21)
    let scaled_scores_l21 = scores_l21 * 0.125  // 1/sqrt(64)
    let attn_weights_l21 = softmax(scaled_scores_l21, 2)
    let attn_output_l21 = einsum("ihj,jhd->ihd", attn_weights_l21, V_expanded_l21)

    // Attention output projection
    let attn_reshaped_l21 = reshape(attn_output_l21, [seq_len, 2048.0])
    let attn_out_l21 = linear(attn_reshaped_l21, W_o_21)
    print("      [DEBUG] Layer 21 attn_out sum:", sum(attn_out_l21))

    // Residual
    let hidden_l21 = layer_20_output + attn_out_l21
    print("      [DEBUG] Layer 21 hidden (after attn) sum:", sum(hidden_l21))

    // Layer 21: FFN
    let x_norm2_l21 = rms_norm(hidden_l21, ffn_norm_21)
    print("      [DEBUG] Layer 21 x_norm2 sum:", sum(x_norm2_l21))

    let gate_l21 = linear(x_norm2_l21, W_gate_21)
    let up_l21 = linear(x_norm2_l21, W_up_21)
    let silu_gate_l21 = silu(gate_l21)
    let gated_l21 = silu_gate_l21 * up_l21
    let ffn_out_l21 = linear(gated_l21, W_down_21)

    print("      [DEBUG] Layer 21 ffn_out sum:", sum(ffn_out_l21))

    // Final residual
    let layer_21_output = hidden_l21 + ffn_out_l21

    let layer_21_sum = sum(layer_21_output)
    print("      Layer 21 output sum:", layer_21_sum)
    print("")
    // Final norm
    let final_norm = rms_norm(layer_21_output, output_norm)
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
    print("22-Layer Test Summary:")
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
    print("  Layer 10 output sum:", layer_10_sum)
    print("  Layer 11 output sum:", layer_11_sum)
    print("  Layer 12 output sum:", layer_12_sum)
    print("  Layer 13 output sum:", layer_13_sum)
    print("  Layer 14 output sum:", layer_14_sum)
    print("  Layer 15 output sum:", layer_15_sum)
    print("  Layer 16 output sum:", layer_16_sum)
    print("  Layer 17 output sum:", layer_17_sum)
    print("  Layer 18 output sum:", layer_18_sum)
    print("  Layer 19 output sum:", layer_19_sum)
    print("  Layer 20 output sum:", layer_20_sum)
    print("  Layer 21 output sum:", layer_21_sum)
    print("  Final norm sum:", final_norm_sum)
    print("  Last token logits sum:", last_token_sum)
    print("  Predicted token:", predicted_token)
    print("================================================================================")
    print("")

    print("================================================================================")
    print("✅ 22-Layer Test Complete")
    print("================================================================================")
}
