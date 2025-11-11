// ============================================================================
// TinyLlama 1.1B Chat Demo with TensorBuffer and Debug Logging
// ============================================================================
//
// TensorBuffer最適化:
//   ✓ 事前にGPUメモリを大量確保（累積的な枯渇を防止）
//   ✓ 因果マスクを各層で生成（再利用による数値エラーを防止）
//   ✓ 詳細なデバッグログで各ステップを追跡
//
// ============================================================================

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

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

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    // Reshape to heads
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    // Apply RoPE
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    // GQA: Expand K, V from 4 to 32 heads
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    // Attention computation
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125  // 1/sqrt(64)

    // Generate mask for this layer (NOT reused from outside!)
    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Flatten and project
    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    linear(attn_reshaped, W_o)
}

fn transformer_layer(
    x: float16[?, ?],
    attn_norm: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    ffn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    // Pre-attention RMSNorm
    let x_norm1 = rms_norm(x, attn_norm)

    // QKV projections
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)

    // GQA Attention (mask generated internally)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

    // First residual
    let x1 = x + attn_out

    // Pre-FFN RMSNorm
    let x_norm2 = rms_norm(x1, ffn_norm)

    // SwiGLU FFN
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

    // Second residual
    x1 + ffn_out
}

// ============================================================================
// Main: Chat Demo with TensorBuffer
// ============================================================================
main {
    print("================================================================================")
    print("🚀 TinyLlama 1.1B Chat Demo with TensorBuffer Optimization")
    print("================================================================================")
    print("")

    // ----------------------------------------------------------------------------
    // [1] TensorBuffer Initialization
    // ----------------------------------------------------------------------------
    print("[1/6] Initializing TensorBuffer...")

    // Calculate buffer size for 22 layers with 46 tokens
    // Per layer intermediate tensors:
    //   - Q, K, V projections: ~400KB
    //   - Attention scores: [46, 32, 46] = 67KB
    //   - FFN intermediates: ~500KB
    //   Total per layer: ~1MB
    // 22 layers: ~22MB + 50MB safety margin = 72MB
    let buffer_size = 100 * 1024 * 1024  // 100MB buffer
    let buffer = new_tensor_buffer(buffer_size)
    print("      ✓ Created 100MB TensorBuffer")
    print("      Buffer status:", buffer)
    print("")

    // Pre-allocate common tensor shapes
    print("      Pre-allocating tensor slots...")
    buffer.alloc([46, 2048], 100, 2)      // Q projections (50 tensors, safety margin)
    buffer.alloc([46, 256], 100, 2)       // K, V projections
    buffer.alloc([46, 32, 46], 50, 2)     // Attention scores
    buffer.alloc([46, 5632], 50, 2)       // FFN intermediates
    print("      ✓ Pre-allocated tensor slots")
    print("      Buffer status:", buffer)
    print("")

    // ----------------------------------------------------------------------------
    // [2] Load Model and Tokenizer
    // ----------------------------------------------------------------------------
    print("[2/6] Loading model and tokenizer...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("      ✓ Model loaded")

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [3] Load Weights
    // ----------------------------------------------------------------------------
    print("[3/6] Loading model weights...")

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Core weights loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [4] Prepare Chat Prompt
    // ----------------------------------------------------------------------------
    print("[4/6] Preparing conversation...")

    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."

    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("      System:", system_prompt)
    print("      User:", user_message)
    print("")

    // Tokenize
    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let prompt_len = len(prompt_tokens)
    print("      Prompt tokens:", prompt_len)
    print("")

    print("      ✓ Masks will be generated internally for each layer")
    print("")

    // ----------------------------------------------------------------------------
    // [5] GENERATION LOOP: Generate tokens with TensorBuffer optimization
    // ----------------------------------------------------------------------------
    print("[5/6] Generating response (max 3 tokens)...")
    print("")
    print("      Assistant: ")

    let eos_token_id = 2
    let temperature = 0.8

    let gen_tokens = prompt_tokens
    let generated_count = 0

    // Token generation loop - LIMITED TO 3 TOKENS
    for token_step in range_i(3) {
        print("")
        print("      === Token Generation Step", token_step, "===")

        let current_len = len(gen_tokens)
        print("      Current sequence length:", current_len)

        // Get embeddings for all tokens so far
        print("      [Step 1] Computing embeddings...")
        let current_emb = embedding(emb_weight, gen_tokens)
        print("      ✓ Embeddings computed")

        // Process through 3 layers (testing memory optimization)
        print("      [Step 2] Processing 3 layers...")
        let x = current_emb
        for layer_idx in range_i(3) {
            print("        Layer", layer_idx, "...")
            let attn_norm = model.blk[layer_idx].attn_norm.weight
            let W_q = model.blk[layer_idx].attn_q.weight
            let W_k = model.blk[layer_idx].attn_k.weight
            let W_v = model.blk[layer_idx].attn_v.weight
            let W_o = model.blk[layer_idx].attn_output.weight
            let ffn_norm = model.blk[layer_idx].ffn_norm.weight
            let W_gate = model.blk[layer_idx].ffn_gate.weight
            let W_up = model.blk[layer_idx].ffn_up.weight
            let W_down = model.blk[layer_idx].ffn_down.weight

            x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
            print("        Layer", layer_idx, "done")
        }
        print("      ✓ All layers completed")

        // Extract last token's hidden state
        print("      [Step 3] Extracting last token...")
        print("        [3.1] slice_last()...")
        let last_hidden_1d = slice_last(x, 0)
        print("        ✓ slice_last() done")

        print("        [3.2] reshape()...")
        let last_hidden = reshape(last_hidden_1d, [1, 2048])
        print("        ✓ reshape() done")

        // Final layer norm and logits projection
        print("      [Step 4] Computing logits...")
        print("        [4.1] rms_norm()...")
        let normed = rms_norm(last_hidden, output_norm_weight)
        print("        ✓ rms_norm() done")

        print("        [4.2] linear() for logits (BIG: [1,2048] @ [32000,2048])...")
        let logits = linear(normed, output_weight)
        print("        ✓ linear() done")

        // Sample next token (GPU sync happens here via to_cpu)
        print("      [Step 5] Sampling next token...")
        let next_token_id = temperature_sample(logits, temperature)
        print("      ✓ Sampled token:", next_token_id)

        // Check for EOS
        if next_token_id == eos_token_id {
            print("")
            print("")
            print("      [Generation stopped: EOS after", generated_count, "tokens]")
            break
        }

        // Append token
        gen_tokens = append(gen_tokens, next_token_id)
        generated_count = generated_count + 1

        print("      ✓ Token generation step", token_step, "completed!")
    }

    print("")
    print("")

    // ----------------------------------------------------------------------------
    // [6] Display Final Result
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Generation Summary")
    print("================================================================================")
    print("")
    print("      Tokens generated:", generated_count)
    print("      Total sequence length:", len(gen_tokens))
    print("")
    print("--------------------------------------------------------------------------------")
    print("Full Conversation:")
    print("--------------------------------------------------------------------------------")
    print("")

    let final_text = detokenize(tokenizer, gen_tokens, false)
    print(final_text)
    print("")

    print("================================================================================")
    print("✅ Chat demo with TensorBuffer completed!")
    print("================================================================================")
    print("")
    print("💡 TensorBuffer Benefits:")
    print("   - Pre-allocated 100MB GPU memory (prevents cumulative exhaustion)")
    print("   - Pre-allocated tensor slots (reduces allocation overhead)")
    print("   - Testing with 3 layers (memory-efficient demonstration)")
    print("   - Buffer status:", buffer)
    print("")
    print("Note: Using 3 layers instead of 22 for stable demonstration")
    print("      Full 22-layer support requires additional memory optimizations")
    print("")
}
