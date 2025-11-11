// ============================================================================
// TinyLlama 1.1B Chat Demo - Loop-Based Generation
// ============================================================================
//
// This demo implements conversational AI using loop-based generation:
//   - TinyLlama 1.1B Chat model (GGUF format)
//   - Full 22-layer Transformer with loop processing
//   - Autoregressive generation up to 10 tokens
//   - EOS detection for early stopping
//   - Incremental decoding (display tokens as generated)
//
// Model Architecture:
//   - Embedding dimension: 2048 (d_model)
//   - Number of layers: 22
//   - Query heads: 32, KV heads: 4 (Grouped Query Attention)
//   - FFN hidden size: 5632
//   - Vocabulary size: 32000
//
// Key Innovation:
//   - Dynamic layer access: model.blk[i]
//   - Loop-based processing eliminates manual expansion
//   - Scalable to any number of layers and tokens
//
// Usage:
//   cargo build --release
//   timeout 180 ./target/release/tl run examples/chat_demo_22layers_loop.tl
//
// ============================================================================

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

// SiLU (Swish) activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

// SwiGLU Feed-Forward Network
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

// Grouped Query Attention (GQA) for TinyLlama
// 32 query heads, 4 KV heads (8:1 group ratio)
fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    // Dynamic seq_len from Q shape
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    // Reshape to heads: [seq, heads, head_dim]
    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    // Apply RoPE to Q and K (NOT V)
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA: Expand K, V from 4 to 32 heads (group size = 8)
    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)

    // Apply causal mask (prevent attending to future positions)
    let mask_2d = causal_mask(seq_len_f)
    let shape_mask_3d = [seq_len_f, 1.0, seq_len_f]
    let mask_3d = reshape(mask_2d, shape_mask_3d)
    let shape_mask_broadcast = [seq_len_f, 32.0, seq_len_f]
    let mask_broadcast = broadcast_to(mask_3d, shape_mask_broadcast)
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Flatten and project
    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    linear(attn_reshaped, W_o)
}


// ============================================================================
// Main: Loop-Based Chat Demo
// ============================================================================
main {
    print("================================================================================")
    print("🤖 TinyLlama 1.1B Chat Demo - Loop-Based Generation")
    print("================================================================================")
    print("")

    // ----------------------------------------------------------------------------
    // [1] Load Model and Tokenizer
    // ----------------------------------------------------------------------------
    print("[1/6] Loading model and tokenizer...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("      ✓ Model loaded:", model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [2] Load Fixed Weights (non-layer specific)
    // ----------------------------------------------------------------------------
    print("[2/6] Loading model weights...")

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Embedding and output weights loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [3] Prepare Input
    // ----------------------------------------------------------------------------
    print("[3/6] Preparing chat prompt...")

    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello!"

    // TinyLlama chat template
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("      System:", system_prompt)
    print("      User:", user_message)
    print("      ✓ Prompt formatted")
    print("")

    // Tokenize prompt
    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    print("      Prompt tokens:", prompt_tokens)
    print("")

    // ----------------------------------------------------------------------------
    // [4] Generation Loop - Up to 10 tokens with EOS detection
    // ----------------------------------------------------------------------------
    print("[4/6] Generating response (max 10 tokens)...")
    print("")
    print("      Assistant: ")

    let eos_token_id = 2
    let temperature = 0.1  // Lower temperature for more deterministic output
    let max_tokens = 10

    let gen_tokens = prompt_tokens
    let generated_count = 0

    // Generate up to 10 tokens (range_i requires constant)
    for token_step in range_i(10) {
        print("")
        print("[DEBUG] Token step:", token_step)

        // Get embeddings for all tokens so far
        let current_emb = embedding(emb_weight, gen_tokens)
        let emb_shape = shape(current_emb)
        print("[DEBUG] Embedding shape:", emb_shape)

        // Debug: Print first token's first 10 embedding values
        print("[DEBUG] Embedding[0, 0:10]:")
        print("  ", current_emb[0, 0], current_emb[0, 1], current_emb[0, 2], current_emb[0, 3], current_emb[0, 4])
        print("  ", current_emb[0, 5], current_emb[0, 6], current_emb[0, 7], current_emb[0, 8], current_emb[0, 9])

        // Process through all 22 layers using loop
        let x = current_emb
        for layer_idx in range_i(22) {
            // Only debug first layer, first token
            if layer_idx == 0 && token_step == 0 {
                print("[DEBUG LAYER 0] Before RMSNorm:")
                print("  x[0, 0:5]:", x[0, 0], x[0, 1], x[0, 2], x[0, 3], x[0, 4])
            }
            // Dynamically load layer weights using model.blk[layer_idx]
            let attn_norm = model.blk[layer_idx].attn_norm.weight
            let W_q = model.blk[layer_idx].attn_q.weight
            let W_k = model.blk[layer_idx].attn_k.weight
            let W_v = model.blk[layer_idx].attn_v.weight
            let W_o = model.blk[layer_idx].attn_output.weight

            let ffn_norm = model.blk[layer_idx].ffn_norm.weight
            let W_gate = model.blk[layer_idx].ffn_gate.weight
            let W_up = model.blk[layer_idx].ffn_up.weight
            let W_down = model.blk[layer_idx].ffn_down.weight

            // Pre-attention RMSNorm
            let x_norm1 = rms_norm(x, attn_norm)

            // Debug RMSNorm output for first layer, first token
            if layer_idx == 0 && token_step == 0 {
                print("[DEBUG LAYER 0] After RMSNorm:")
                print("  x_norm1[0, 0:5]:", x_norm1[0, 0], x_norm1[0, 1], x_norm1[0, 2], x_norm1[0, 3], x_norm1[0, 4])
            }

            // QKV projections
            let Q = linear(x_norm1, W_q)
            let K = linear(x_norm1, W_k)
            let V = linear(x_norm1, W_v)

            // GQA Attention
            let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

            // Residual
            let x1 = x + attn_out

            // Pre-FFN RMSNorm
            let x_norm2 = rms_norm(x1, ffn_norm)

            // SwiGLU FFN
            let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

            // Final residual - update x for next layer
            x = x1 + ffn_out
        }

        let final_output = x
        let final_shape = shape(final_output)
        print("[DEBUG] Final output shape:", final_shape)

        // Extract last token position [seq_len, d_model] -> [d_model]
        let last_pos_1d = slice_last(final_output, 0)

        // Reshape to 2D [1, d_model] for RMSNorm and linear
        let last_pos = reshape(last_pos_1d, [1.0, 2048.0])

        // Final layer norm and projection to vocabulary
        let normed = rms_norm(last_pos, output_norm_weight)
        let logits = linear(normed, output_weight)
        let logits_shape = shape(logits)
        print("[DEBUG] Logits shape:", logits_shape)

        // Debug: Show argmax (most likely token)
        let argmax_token = argmax(logits, 1, false)
        print("[DEBUG] Argmax token ID:", argmax_token)

        // Debug: Show first 20 logit values
        print("[DEBUG] First 20 logits:")
        print("  logits[0]:", logits[0, 0])
        print("  logits[1]:", logits[0, 1])
        print("  logits[2]:", logits[0, 2])
        print("  logits[3]:", logits[0, 3])
        print("  logits[4]:", logits[0, 4])
        print("  logits[5]:", logits[0, 5])
        print("  logits[6]:", logits[0, 6])
        print("  logits[7]:", logits[0, 7])
        print("  logits[8]:", logits[0, 8])
        print("  logits[9]:", logits[0, 9])
        print("  logits[10]:", logits[0, 10])
        print("  logits[11]:", logits[0, 11])
        print("  logits[12]:", logits[0, 12])
        print("  logits[13]:", logits[0, 13])
        print("  logits[14]:", logits[0, 14])
        print("  logits[15]:", logits[0, 15])
        print("  logits[16]:", logits[0, 16])
        print("  logits[17]:", logits[0, 17])
        print("  logits[18]:", logits[0, 18])
        print("  logits[19]:", logits[0, 19])

        // Sample next token using temperature sampling
        let next_token_id = temperature_sample(logits, temperature)
        print("[DEBUG] Sampled token ID:", next_token_id)

        // Check for EOS
        if next_token_id == eos_token_id {
            print("")
            print("")
            print("      [EOS detected after", generated_count, "tokens]")
            break
        }

        // Append token to sequence (append expects integer, not tensor)
        gen_tokens = append(gen_tokens, next_token_id)
        generated_count = generated_count + 1
        print("[DEBUG] Total tokens in sequence:", generated_count)

        // Incremental decode and display (just the new token)
        let full_text = detokenize(tokenizer, gen_tokens, false)
        print("[OUTPUT]", full_text)
    }

    print("")
    print("")

    // ----------------------------------------------------------------------------
    // [5] Display Results
    // ----------------------------------------------------------------------------
    print("[5/6] Generation complete!")
    print("")
    print("      Tokens generated:", generated_count)
    print("      Final sequence:", gen_tokens)
    print("")

    // ----------------------------------------------------------------------------
    // [6] Full Text
    // ----------------------------------------------------------------------------
    print("[6/6] Full conversation:")
    print("")

    let final_text = detokenize(tokenizer, gen_tokens, false)
    print(final_text)
    print("")

    print("================================================================================")
    print("✅ Chat demo completed successfully!")
    print("================================================================================")
}
