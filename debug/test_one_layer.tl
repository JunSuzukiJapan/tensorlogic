// Test: 1-layer transformer for debugging GPU sync issues
// Compare with candle_1layer_reference.json

// SiLU activation function (inline version now works with fix)
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("TensorLogic 1-Layer Transformer Test")
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
    print("[2/5] Loading weights for Layer 0 only...")
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

    print("      ✓ Layer 0 weights loaded")
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
    print("[5/5] Running forward pass (Layer 0 only)...")
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

    // Final norm
    let final_norm = rms_norm(layer_0_output, output_norm)
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
    print("Comparison with Candle Reference:")
    print("  Candle Embedding sum: 3.66015625    | TensorLogic: 3.673828125")
    print("  Candle Layer 0 sum:   774.0         | TensorLogic: 5.1328125")
    print("  Candle Final norm:    774.0         | TensorLogic:", final_norm_sum)
    print("  Candle Last token logits sum: -34784.0 | TensorLogic:", last_token_sum)
    print("================================================================================")
    print("")

    print("")
    print("      Predicted token:", predicted_token)
    print("")

    print("================================================================================")
    print("✅ 1-Layer Test Complete")
    print("================================================================================")
}
