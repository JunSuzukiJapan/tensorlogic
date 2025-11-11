// ============================================================================
// TinyLlama 1.1B Chat Demo with KV Cache - Fast Inference
// ============================================================================
//
// This demo implements KV cache for efficient autoregressive generation:
//   - Prefill: Process all prompt tokens at once, store K/V
//   - Decode: Reuse cached K/V and embed ONLY the new token (optimized!)
//
// OPTIMIZATION: embedding() now supports single token IDs (Integer type),
//               so we can embed just the new token in decode phase instead
//               of re-embedding all tokens. This provides the full speedup
//               expected from KV cache implementations.
//
// Model Architecture:
//   - Embedding dimension: 2048 (d_model)
//   - Number of layers: 22
//   - Query heads: 32, KV heads: 4 (Grouped Query Attention)
//   - Vocabulary size: 32000
//
// Usage:
//   cargo build --release
//   timeout 180 ./target/release/tl run debug/chat_demo_with_kvcache.tl
//
// ============================================================================

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
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
    linear(intermediate, W_down)
}

// GQA Attention WITHOUT cache (for prefill phase)
fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    // Get sequence length from Q
    let Q_shape_tensor = shape(Q)
    let seq_len_int = Q_shape_tensor[0]
    let seq_len_f = seq_len_int * 1.0

    // Reshape to heads [seq, heads, head_dim]
    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let K_heads = reshape(K, [seq_len_f, 4.0, 64.0])
    let V_heads = reshape(V, [seq_len_f, 4.0, 64.0])

    // Apply RoPE
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA: Expand K, V from 4 to 32 heads
    let K_with_group = reshape(K_rope, [seq_len_f, 4.0, 1.0, 64.0])
    let V_with_group = reshape(V_heads, [seq_len_f, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_with_group, [seq_len_f, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_with_group, [seq_len_f, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [seq_len_f, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len_f, 32.0, 64.0])

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)

    // Apply causal mask
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len_f, 1.0, seq_len_f])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len_f, 32.0, seq_len_f])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Flatten and project
    let attn_reshaped = reshape(attn_output, [seq_len_f, 2048.0])
    linear(attn_reshaped, W_o)
}

main {
    print("================================================================================")
    print("🤖 TinyLlama Chat Demo with KV Cache")
    print("================================================================================")
    print("")

    // [1] Load model and tokenizer
    print("[1/6] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model and tokenizer loaded")
    print("")

    // [2] Load weights
    print("[2/6] Loading weights...")
    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Embedding and output weights loaded")
    print("")

    // [3] Create KV cache for 22 layers
    print("[3/6] Creating KV cache...")
    let cache = KVCache::new_f16(22)
    print("      ✓ KV cache created")
    print("")

    // [4] Prepare prompt
    print("[4/6] Preparing prompt...")
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello!"
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    print("      Prompt tokens:", len(prompt_tokens))
    print("")

    // [5] Prefill phase - process all prompt tokens and store in cache
    print("[5/6] Prefill phase (processing prompt and building cache)...")
    let emb_prefill = embedding(emb_weight, prompt_tokens)

    let x_prefill = emb_prefill
    for layer_idx in range_i(22) {
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
        let x_norm1 = rms_norm(x_prefill, attn_norm)

        // QKV projections
        let Q = linear(x_norm1, W_q)
        let K = linear(x_norm1, W_k)
        let V = linear(x_norm1, W_v)

        // Prepare K and V for caching (reshape to heads and apply RoPE to K)
        let seq_shape = shape(K)
        let seq_len_int = seq_shape[0]
        let seq_len_f = seq_len_int * 1.0

        let K_heads = reshape(K, [seq_len_f, 4.0, 64.0])
        let V_heads = reshape(V, [seq_len_f, 4.0, 64.0])
        let K_rope = rope(K_heads)

        // Store K and V in cache (prefill)
        cache.set(layer_idx, K_rope, V_heads)

        // GQA Attention (using non-cached function for prefill)
        let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

        // Residual
        let x1 = x_prefill + attn_out

        // Pre-FFN RMSNorm
        let x_norm2 = rms_norm(x1, ffn_norm)

        // SwiGLU FFN
        let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

        // Final residual
        x_prefill = x1 + ffn_out
    }

    // Extract last token logits and sample first token
    let last_pos_1d = slice_last(x_prefill, 0)
    let last_pos = reshape(last_pos_1d, [1.0, 2048.0])
    let normed = rms_norm(last_pos, output_norm_weight)
    let logits = linear(normed, output_weight)

    let temperature = 0.8
    let first_token = temperature_sample(logits, temperature)

    let gen_tokens = append(prompt_tokens, first_token)
    print("      ✓ Prefill complete, cache populated, first token sampled")
    print("")

    // [6] Decode phase - generate tokens one by one using cache
    print("[6/6] Decode phase (generating tokens with KV cache)...")
    print("")
    print("      Assistant: ")

    let eos_token_id = 2
    let max_new_tokens = 2
    let generated_count = 1

    // Output first token
    let decoded_so_far = detokenize(tokenizer, gen_tokens, false)
    print("      ", decoded_so_far)

    // Track current token for decode
    let current_token = first_token

    // Generate remaining tokens using cache
    for token_step in range_i(max_new_tokens) {
        // Get embedding for current token only (NEW: single token support!)
        let emb_new = embedding(emb_weight, current_token)

        // Process through all layers (using cached K/V)
        let x_decode = emb_new
        for layer_idx in range_i(22) {
            let attn_norm = model.blk[layer_idx].attn_norm.weight
            let W_q = model.blk[layer_idx].attn_q.weight
            let W_k = model.blk[layer_idx].attn_k.weight
            let W_v = model.blk[layer_idx].attn_v.weight
            let W_o = model.blk[layer_idx].attn_output.weight
            let ffn_norm = model.blk[layer_idx].ffn_norm.weight
            let W_gate = model.blk[layer_idx].ffn_gate.weight
            let W_up = model.blk[layer_idx].ffn_up.weight
            let W_down = model.blk[layer_idx].ffn_down.weight

            // === DECODE ATTENTION WITH CACHE ===
            let x_norm1 = rms_norm(x_decode, attn_norm)

            // Compute Q for new token only
            let Q = linear(x_norm1, W_q)
            let Q_heads = reshape(Q, [1.0, 32.0, 64.0])
            let Q_rope = rope(Q_heads)

            // Compute K and V for new token
            let K_new = linear(x_norm1, W_k)
            let V_new = linear(x_norm1, W_v)

            let K_new_heads = reshape(K_new, [1.0, 4.0, 64.0])
            let V_new_heads = reshape(V_new, [1.0, 4.0, 64.0])
            let K_new_rope = rope(K_new_heads)

            // Append new K/V to cache (decode)
            cache.append(layer_idx, K_new_rope, V_new_heads)

            // Get full K/V from cache
            let K_full = cache.get_k(layer_idx)
            let V_full = cache.get_v(layer_idx)

            // Get full sequence length
            let K_full_shape = shape(K_full)
            let seq_full_int = K_full_shape[0]
            let seq_full_f = seq_full_int * 1.0

            // GQA: Expand K, V from 4 to 32 heads
            let K_with_group = reshape(K_full, [seq_full_f, 4.0, 1.0, 64.0])
            let V_with_group = reshape(V_full, [seq_full_f, 4.0, 1.0, 64.0])

            let K_broadcast = broadcast_to(K_with_group, [seq_full_f, 4.0, 8.0, 64.0])
            let V_broadcast = broadcast_to(V_with_group, [seq_full_f, 4.0, 8.0, 64.0])

            let K_expanded = reshape(K_broadcast, [seq_full_f, 32.0, 64.0])
            let V_expanded = reshape(V_broadcast, [seq_full_f, 32.0, 64.0])

            // Attention (Q attends to all cached K/V)
            let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
            let scaled_scores = scores * 0.125

            // Causal mask for single query position
            let mask_2d = causal_mask(1)
            let mask_3d = reshape(mask_2d, [1.0, 1.0, 1.0])
            let mask_broadcast = broadcast_to(mask_3d, [1.0, 32.0, seq_full_f])
            let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

            let attn_weights = softmax(masked_scores, 2)
            let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

            let attn_reshaped = reshape(attn_output, [1.0, 2048.0])
            let attn_out = linear(attn_reshaped, W_o)
            // === END DECODE ATTENTION ===

            let x1 = x_decode + attn_out

            let x_norm2 = rms_norm(x1, ffn_norm)
            let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

            x_decode = x1 + ffn_out
        }

        // Get logits and sample
        let normed_decode = rms_norm(x_decode, output_norm_weight)
        let logits_decode = linear(normed_decode, output_weight)
        let next_token = temperature_sample(logits_decode, temperature)

        // Check for EOS
        if next_token == eos_token_id {
            print("")
            print("")
            print("      [EOS detected after", generated_count, "tokens]")
            break
        }

        // Append token and decode
        gen_tokens = append(gen_tokens, next_token)
        generated_count = generated_count + 1
        current_token = next_token

        let decoded_text = detokenize(tokenizer, gen_tokens, false)
        print("      ", decoded_text)
    }

    print("")
    print("")

    // Summary
    print("===============================================")
    print("✅ Chat demo with KV cache completed!")
    print("   Tokens generated:", generated_count)
    print("===============================================")
}
