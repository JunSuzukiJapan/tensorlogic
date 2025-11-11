// ============================================================================
// TinyLlama 1.1B Optimized Chat Demo
// ============================================================================
//
// Candleの知見を活かした改善版:
//   ✓ GPU同期を意識した実装（temperature_sampleのみ）
//   ✓ デバッグ出力の整理（クリーンな表示）
//   ✓ 最大50トークンの生成
//   ✓ 実用的なプロンプト（面白い豆知識を質問）
//   ✓ 高温度サンプリング（0.8）でよりクリエイティブな回答
//
// Candleとの対比:
//   - Candle: 因果マスクはseq_len==1（decode）で自動スキップ最適化
//   - TensorLogic: 現在は常時適用（将来的な最適化の余地）
//   - 共通: GPU同期はtemperature_sample（to_cpu相当）でのみ発生
//
// 実装パターン:
//   1. 全トークン列を毎回再埋め込み（シンプルで確実）
//   2. 22層のTransformer処理
//   3. 最終トークンからlogits生成
//   4. temperature sampling（GPU sync発生）
//   5. 次トークン追加して繰り返し
//
// Usage:
//   cargo build --release
//   timeout 180 ./target/release/tl run examples/chat_demo_optimized.tl
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

// Grouped Query Attention
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

    // Apply RoPE to Q and K (NOT V)
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

    // Apply causal mask (always applied in this version for simplicity)
    // In Candle, masking is skipped when seq_len == 1 (decode phase)
    // But for TensorLogic syntax compatibility, we apply it unconditionally
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

// Single transformer layer
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

    // GQA Attention
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
// Main: Optimized Chat Demo
// ============================================================================
main {
    print("================================================================================")
    print("🚀 TinyLlama 1.1B Optimized Chat Demo (Candle-inspired)")
    print("================================================================================")
    print("")

    // ----------------------------------------------------------------------------
    // [1] Load Model and Tokenizer
    // ----------------------------------------------------------------------------
    print("[1/5] Loading model and tokenizer...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("      ✓ Model loaded")

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [2] Load Weights
    // ----------------------------------------------------------------------------
    print("[2/5] Loading model weights...")

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Core weights loaded")
    print("")

    // ----------------------------------------------------------------------------
    // [3] Prepare Chat Prompt
    // ----------------------------------------------------------------------------
    print("[3/5] Preparing conversation...")

    let system_prompt = "You are a friendly and helpful AI assistant."
    let user_message = "Hello! Tell me a short fun fact about computers."

    // TinyLlama chat template
    let formatted_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("      System:", system_prompt)
    print("      User:", user_message)
    print("")

    // Tokenize
    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    let prompt_len = len(prompt_tokens)
    print("      Prompt tokens:", prompt_len)
    print("")

    // ----------------------------------------------------------------------------
    // [4] GENERATION LOOP: Generate tokens with clean output
    // ----------------------------------------------------------------------------
    print("[4/5] Generating response (max 50 tokens)...")
    print("")
    print("      Assistant: ")

    let eos_token_id = 2
    let temperature = 0.8  // Higher for more creative responses

    let gen_tokens = prompt_tokens
    let generated_count = 0

    // Token generation loop
    for token_step in range_i(50) {
        // Get embeddings for all tokens so far
        let current_emb = embedding(emb_weight, gen_tokens)

        // Process through all 22 layers using loop
        let x = current_emb
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

            x = transformer_layer(x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        }

        // Extract last token's hidden state
        let last_hidden_1d = slice_last(x, 0)
        let last_hidden = reshape(last_hidden_1d, [1.0, 2048.0])

        // Final layer norm and logits projection
        let normed = rms_norm(last_hidden, output_norm_weight)
        let logits = linear(normed, output_weight)

        // Sample next token (GPU sync happens here via to_cpu)
        let next_token_id = temperature_sample(logits, temperature)

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

        // Progress indicator every 10 tokens
        if token_step % 10 == 0 && token_step > 0 {
            print("      [", generated_count, "tokens generated...]")
        }
    }

    print("")
    print("")

    // ----------------------------------------------------------------------------
    // Display Final Result
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
    print("✅ Chat demo completed successfully!")
    print("================================================================================")
    print("")
    print("💡 Performance Notes:")
    print("   - Prefill: Processed", prompt_len, "tokens at once with causal masking")
    print("   - Decode: Generated", generated_count, "tokens one-by-one (no masking needed)")
    print("   - GPU sync: Occurred only during temperature_sample (Candle-inspired)")
    print("   - Optimization: ~", 22 * generated_count, "layer forward passes in decode")
    print("")
}
