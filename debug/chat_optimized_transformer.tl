// ============================================================================
// Optimized Transformer Chat Demo with GQA + GGUF Loading
// ============================================================================
//
// This implementation combines:
//   1. Explicit Grouped Query Attention (GQA) from optimized_transformer.tl
//   2. GGUF model loading and KV cache from chat_full_22layers_f32.tl
//   3. SwiGLU activation for modern LLM architecture
//   4. Autoregressive generation with 50 token limit
//
// Model: TinyLlama 1.1B Chat (f32)
// Architecture: 22 layers, 32 Q heads, 4 KV heads, 2048 hidden_dim
// ============================================================================

// ----------------------------------------------------------------------------
// Helper: SiLU (Swish) Activation
// ----------------------------------------------------------------------------
fn silu(x: float32[?, ?]) -> float32[?, ?] {
    x * sigmoid(x)
}

// ----------------------------------------------------------------------------
// SwiGLU Feed-Forward Network
// ----------------------------------------------------------------------------
fn swiglu_ffn(
    x: float32[?, ?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?]
) -> float32[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    linear(mul_result, W_down)
}

// ----------------------------------------------------------------------------
// Helper: Apply RoPE to K for caching
// ----------------------------------------------------------------------------
fn apply_rope_k(K: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])  // 4 KV heads, 64 head_dim
    let K_r = rope(K_h, pos)
    reshape(K_r, [seq_len, 256.0])  // 4 * 64 = 256
}

// ----------------------------------------------------------------------------
// Helper: Apply RoPE to Q
// ----------------------------------------------------------------------------
fn apply_rope_q(Q: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let Q_h = reshape(Q, [seq_len, 32.0, 64.0])  // 32 Q heads, 64 head_dim
    let Q_r = rope(Q_h, pos)
    reshape(Q_r, [seq_len, 2048.0])  // 32 * 64 = 2048
}

// ----------------------------------------------------------------------------
// Multi-Head Attention with KV Cache (using builtin)
// ----------------------------------------------------------------------------
// NOTE: Using Rust builtin implementation with GQA support
// The builtin version is optimized with:
// - Grouped Query Attention (4 KV heads, 32 Q heads)
// - Efficient K/V expansion via broadcast (repeat_kv_for_gqa)
// - No need for cache_len parameter (calculated internally)

// ----------------------------------------------------------------------------
// Transformer Layer with GQA and SwiGLU
// ----------------------------------------------------------------------------
fn transformer_layer(
    x: float32[?, ?],
    W_attn_norm: float32[?],
    W_q: float32[?, ?],
    W_o: float32[?, ?],
    W_ffn_norm: float32[?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?],
    position: float,
    seq_len: float
) -> float32[?, ?] {
    // ===== Self-Attention Block =====
    let normed = rms_norm(x, W_attn_norm)
    let Q_raw = linear(normed, W_q)
    let Q = apply_rope_q(Q_raw, seq_len, position)

    // Use builtin attention_with_cache (optimized with GQA support)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out

    // ===== Feed-Forward Block =====
    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    after_attn + ffn_out
}

// ============================================================================
// Main: Chat Demo with Optimized Architecture
// ============================================================================
main {
    print("============================================================================")
    print("Optimized Transformer Chat Demo (GQA + SwiGLU + GGUF)")
    print("============================================================================")
    print("")

    let EOS_TOKEN = 2

    // ========================================================================
    // [1/3] Load Model
    // ========================================================================
    print("[1/3] Loading TinyLlama 1.1B Chat (f32)...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    // Load all 22 layers
    let L0 = model.blk[0]
    let L1 = model.blk[1]
    let L2 = model.blk[2]
    let L3 = model.blk[3]
    let L4 = model.blk[4]
    let L5 = model.blk[5]
    let L6 = model.blk[6]
    let L7 = model.blk[7]
    let L8 = model.blk[8]
    let L9 = model.blk[9]
    let L10 = model.blk[10]
    let L11 = model.blk[11]
    let L12 = model.blk[12]
    let L13 = model.blk[13]
    let L14 = model.blk[14]
    let L15 = model.blk[15]
    let L16 = model.blk[16]
    let L17 = model.blk[17]
    let L18 = model.blk[18]
    let L19 = model.blk[19]
    let L20 = model.blk[20]
    let L21 = model.blk[21]

    print("      ✓ All 22 layers loaded")
    print("")

    // ========================================================================
    // [2/3] Prepare Prompt
    // ========================================================================
    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)
    print("      User: {}", user_msg)
    print("")

    // ========================================================================
    // [3/3] Generate Response
    // ========================================================================
    print("[3/3] Generating response (max 50 tokens)...")
    print("")
    print("      Assistant: ", "")

    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]  // Get actual sequence length dynamically
    print("[DEBUG] Actual sequence length from tokens: {}", seq_len)

    // ========================================================================
    // PREFILL PHASE: Build initial KV caches for all 22 layers
    // ========================================================================
    let K0_raw = linear(x, L0.attn_k.weight)
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)
    let K1_raw = linear(x, L1.attn_k.weight)
    let K1 = apply_rope_k(K1_raw, seq_len, 0.0)
    let V1 = linear(x, L1.attn_v.weight)
    let K2_raw = linear(x, L2.attn_k.weight)
    let K2 = apply_rope_k(K2_raw, seq_len, 0.0)
    let V2 = linear(x, L2.attn_v.weight)
    let K3_raw = linear(x, L3.attn_k.weight)
    let K3 = apply_rope_k(K3_raw, seq_len, 0.0)
    let V3 = linear(x, L3.attn_v.weight)
    let K4_raw = linear(x, L4.attn_k.weight)
    let K4 = apply_rope_k(K4_raw, seq_len, 0.0)
    let V4 = linear(x, L4.attn_v.weight)
    let K5_raw = linear(x, L5.attn_k.weight)
    let K5 = apply_rope_k(K5_raw, seq_len, 0.0)
    let V5 = linear(x, L5.attn_v.weight)
    let K6_raw = linear(x, L6.attn_k.weight)
    let K6 = apply_rope_k(K6_raw, seq_len, 0.0)
    let V6 = linear(x, L6.attn_v.weight)
    let K7_raw = linear(x, L7.attn_k.weight)
    let K7 = apply_rope_k(K7_raw, seq_len, 0.0)
    let V7 = linear(x, L7.attn_v.weight)
    let K8_raw = linear(x, L8.attn_k.weight)
    let K8 = apply_rope_k(K8_raw, seq_len, 0.0)
    let V8 = linear(x, L8.attn_v.weight)
    let K9_raw = linear(x, L9.attn_k.weight)
    let K9 = apply_rope_k(K9_raw, seq_len, 0.0)
    let V9 = linear(x, L9.attn_v.weight)
    let K10_raw = linear(x, L10.attn_k.weight)
    let K10 = apply_rope_k(K10_raw, seq_len, 0.0)
    let V10 = linear(x, L10.attn_v.weight)
    let K11_raw = linear(x, L11.attn_k.weight)
    let K11 = apply_rope_k(K11_raw, seq_len, 0.0)
    let V11 = linear(x, L11.attn_v.weight)
    let K12_raw = linear(x, L12.attn_k.weight)
    let K12 = apply_rope_k(K12_raw, seq_len, 0.0)
    let V12 = linear(x, L12.attn_v.weight)
    let K13_raw = linear(x, L13.attn_k.weight)
    let K13 = apply_rope_k(K13_raw, seq_len, 0.0)
    let V13 = linear(x, L13.attn_v.weight)
    let K14_raw = linear(x, L14.attn_k.weight)
    let K14 = apply_rope_k(K14_raw, seq_len, 0.0)
    let V14 = linear(x, L14.attn_v.weight)
    let K15_raw = linear(x, L15.attn_k.weight)
    let K15 = apply_rope_k(K15_raw, seq_len, 0.0)
    let V15 = linear(x, L15.attn_v.weight)
    let K16_raw = linear(x, L16.attn_k.weight)
    let K16 = apply_rope_k(K16_raw, seq_len, 0.0)
    let V16 = linear(x, L16.attn_v.weight)
    let K17_raw = linear(x, L17.attn_k.weight)
    let K17 = apply_rope_k(K17_raw, seq_len, 0.0)
    let V17 = linear(x, L17.attn_v.weight)
    let K18_raw = linear(x, L18.attn_k.weight)
    let K18 = apply_rope_k(K18_raw, seq_len, 0.0)
    let V18 = linear(x, L18.attn_v.weight)
    let K19_raw = linear(x, L19.attn_k.weight)
    let K19 = apply_rope_k(K19_raw, seq_len, 0.0)
    let V19 = linear(x, L19.attn_v.weight)
    let K20_raw = linear(x, L20.attn_k.weight)
    let K20 = apply_rope_k(K20_raw, seq_len, 0.0)
    let V20 = linear(x, L20.attn_v.weight)
    let K21_raw = linear(x, L21.attn_k.weight)
    let K21 = apply_rope_k(K21_raw, seq_len, 0.0)
    let V21 = linear(x, L21.attn_v.weight)

    // ========================================================================
    // PREFILL: Run prompt through all 22 transformer layers
    // ========================================================================
    let h = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h = transformer_layer(h, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h = transformer_layer(h, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)
    let h = transformer_layer(h, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3, 0.0, seq_len)
    let h = transformer_layer(h, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, K4, V4, 0.0, seq_len)
    let h = transformer_layer(h, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, K5, V5, 0.0, seq_len)
    let h = transformer_layer(h, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, K6, V6, 0.0, seq_len)
    let h = transformer_layer(h, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, K7, V7, 0.0, seq_len)
    let h = transformer_layer(h, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, K8, V8, 0.0, seq_len)
    let h = transformer_layer(h, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, K9, V9, 0.0, seq_len)
    let h = transformer_layer(h, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, K10, V10, 0.0, seq_len)
    let h = transformer_layer(h, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, K11, V11, 0.0, seq_len)
    let h = transformer_layer(h, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, K12, V12, 0.0, seq_len)
    let h = transformer_layer(h, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, K13, V13, 0.0, seq_len)
    let h = transformer_layer(h, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, K14, V14, 0.0, seq_len)
    let h = transformer_layer(h, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, K15, V15, 0.0, seq_len)
    let h = transformer_layer(h, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, K16, V16, 0.0, seq_len)
    let h = transformer_layer(h, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, K17, V17, 0.0, seq_len)
    let h = transformer_layer(h, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, K18, V18, 0.0, seq_len)
    let h = transformer_layer(h, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, K19, V19, 0.0, seq_len)
    let h = transformer_layer(h, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, K20, V20, 0.0, seq_len)
    let h = transformer_layer(h, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, K21, V21, 0.0, seq_len)

    // Final normalization and output projection
    print("[DEBUG] h shape before norm: {}", shape(h))
    let final_norm = rms_norm(h, output_norm)
    print("[DEBUG] final_norm shape: {}", shape(final_norm))

    // 🚨 CRITICAL FIX: Extract ONLY the last token position before projection
    // Candle does: x.i((.., seq_len - 1, ..))
    // Without this, we project ALL tokens [35, 2048] → [35, 32000] causing zero logits
    let last_token = slice_last(final_norm, 0)  // [35, 2048] → [2048]
    print("[DEBUG] last_token shape after slice_last: {}", shape(last_token))

    let last_token_2d = reshape(last_token, [1.0, 2048.0])  // [2048] → [1, 2048]
    print("[DEBUG] last_token_2d shape: {}", shape(last_token_2d))

    // CRITICAL: TinyLlama uses weight tying (output.weight == tok_embd.weight)
    let logits = linear(last_token_2d, tok_embd)  // [1, 2048] @ [2048, 32000] → [1, 32000]

    print("[DEBUG PREFILL] logits shape: {}", shape(logits))

    // ========================================================================
    // AUTOREGRESSIVE GENERATION LOOP
    // ========================================================================
    let temperature = 0.0  // Greedy decoding
    let current_logits = logits
    let continue_generation = true
    let token_count = 0
    let current_pos = seq_len

    // Initialize KV cache
    let kv_cache = KVCache::new_f32(22)
    kv_cache.set(0, K0, V0)
    kv_cache.set(1, K1, V1)
    kv_cache.set(2, K2, V2)
    kv_cache.set(3, K3, V3)
    kv_cache.set(4, K4, V4)
    kv_cache.set(5, K5, V5)
    kv_cache.set(6, K6, V6)
    kv_cache.set(7, K7, V7)
    kv_cache.set(8, K8, V8)
    kv_cache.set(9, K9, V9)
    kv_cache.set(10, K10, V10)
    kv_cache.set(11, K11, V11)
    kv_cache.set(12, K12, V12)
    kv_cache.set(13, K13, V13)
    kv_cache.set(14, K14, V14)
    kv_cache.set(15, K15, V15)
    kv_cache.set(16, K16, V16)
    kv_cache.set(17, K17, V17)
    kv_cache.set(18, K18, V18)
    kv_cache.set(19, K19, V19)
    kv_cache.set(20, K20, V20)
    kv_cache.set(21, K21, V21)

    // Token-by-token generation (max 50 tokens)
    for i in range(50) {
        if continue_generation {
            // Sample next token
            let token_id = temperature_sample(current_logits, temperature)
            let text = detokenize_single(tokenizer, token_id, false)
            print(text, "")

            token_count = token_count + 1

            // Check for EOS
            if token_id == EOS_TOKEN {
                continue_generation = false
                print(" <EOS>")
            }

            // ========================================================
            // DECODE STEP: Update KV caches and process new token
            // ========================================================
            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)

            // Layer 0
            let nK0_raw = linear(new_token_emb, L0.attn_k.weight)
            let nK0 = apply_rope_k(nK0_raw, 1.0, current_pos)
            let nV0 = linear(new_token_emb, L0.attn_v.weight)
            kv_cache.append(0, nK0, nV0)
            let KV0 = kv_cache.get_k(0)
            let KV0_V = kv_cache.get_v(0)
            let nh0 = transformer_layer(new_token_emb, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, KV0, KV0_V, current_pos, 1.0)

            // Layer 1
            let nK1_raw = linear(nh0, L1.attn_k.weight)
            let nK1 = apply_rope_k(nK1_raw, 1.0, current_pos)
            let nV1 = linear(nh0, L1.attn_v.weight)
            kv_cache.append(1, nK1, nV1)
            let KV1 = kv_cache.get_k(1)
            let KV1_V = kv_cache.get_v(1)
            let nh = transformer_layer(nh0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, KV1, KV1_V, current_pos, 1.0)

            // Layer 2
            let nK2_raw = linear(nh, L2.attn_k.weight)
            let nK2 = apply_rope_k(nK2_raw, 1.0, current_pos)
            let nV2 = linear(nh, L2.attn_v.weight)
            kv_cache.append(2, nK2, nV2)
            let KV2 = kv_cache.get_k(2)
            let KV2_V = kv_cache.get_v(2)
            let nh = transformer_layer(nh, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, KV2, KV2_V, current_pos, 1.0)

            // Layer 3
            let nK3_raw = linear(nh, L3.attn_k.weight)
            let nK3 = apply_rope_k(nK3_raw, 1.0, current_pos)
            let nV3 = linear(nh, L3.attn_v.weight)
            kv_cache.append(3, nK3, nV3)
            let KV3 = kv_cache.get_k(3)
            let KV3_V = kv_cache.get_v(3)
            let nh = transformer_layer(nh, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, KV3, KV3_V, current_pos, 1.0)

            // Layer 4
            let nK4_raw = linear(nh, L4.attn_k.weight)
            let nK4 = apply_rope_k(nK4_raw, 1.0, current_pos)
            let nV4 = linear(nh, L4.attn_v.weight)
            kv_cache.append(4, nK4, nV4)
            let KV4 = kv_cache.get_k(4)
            let KV4_V = kv_cache.get_v(4)
            let nh = transformer_layer(nh, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, KV4, KV4_V, current_pos, 1.0)

            // Layer 5
            let nK5_raw = linear(nh, L5.attn_k.weight)
            let nK5 = apply_rope_k(nK5_raw, 1.0, current_pos)
            let nV5 = linear(nh, L5.attn_v.weight)
            kv_cache.append(5, nK5, nV5)
            let KV5 = kv_cache.get_k(5)
            let KV5_V = kv_cache.get_v(5)
            let nh = transformer_layer(nh, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, KV5, KV5_V, current_pos, 1.0)

            // Layer 6
            let nK6_raw = linear(nh, L6.attn_k.weight)
            let nK6 = apply_rope_k(nK6_raw, 1.0, current_pos)
            let nV6 = linear(nh, L6.attn_v.weight)
            kv_cache.append(6, nK6, nV6)
            let KV6 = kv_cache.get_k(6)
            let KV6_V = kv_cache.get_v(6)
            let nh = transformer_layer(nh, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, KV6, KV6_V, current_pos, 1.0)

            // Layer 7
            let nK7_raw = linear(nh, L7.attn_k.weight)
            let nK7 = apply_rope_k(nK7_raw, 1.0, current_pos)
            let nV7 = linear(nh, L7.attn_v.weight)
            kv_cache.append(7, nK7, nV7)
            let KV7 = kv_cache.get_k(7)
            let KV7_V = kv_cache.get_v(7)
            let nh = transformer_layer(nh, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, KV7, KV7_V, current_pos, 1.0)

            // Layer 8
            let nK8_raw = linear(nh, L8.attn_k.weight)
            let nK8 = apply_rope_k(nK8_raw, 1.0, current_pos)
            let nV8 = linear(nh, L8.attn_v.weight)
            kv_cache.append(8, nK8, nV8)
            let KV8 = kv_cache.get_k(8)
            let KV8_V = kv_cache.get_v(8)
            let nh = transformer_layer(nh, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, KV8, KV8_V, current_pos, 1.0)

            // Layer 9
            let nK9_raw = linear(nh, L9.attn_k.weight)
            let nK9 = apply_rope_k(nK9_raw, 1.0, current_pos)
            let nV9 = linear(nh, L9.attn_v.weight)
            kv_cache.append(9, nK9, nV9)
            let KV9 = kv_cache.get_k(9)
            let KV9_V = kv_cache.get_v(9)
            let nh = transformer_layer(nh, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, KV9, KV9_V, current_pos, 1.0)

            // Layer 10
            let nK10_raw = linear(nh, L10.attn_k.weight)
            let nK10 = apply_rope_k(nK10_raw, 1.0, current_pos)
            let nV10 = linear(nh, L10.attn_v.weight)
            kv_cache.append(10, nK10, nV10)
            let KV10 = kv_cache.get_k(10)
            let KV10_V = kv_cache.get_v(10)
            let nh = transformer_layer(nh, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, KV10, KV10_V, current_pos, 1.0)

            // Layer 11
            let nK11_raw = linear(nh, L11.attn_k.weight)
            let nK11 = apply_rope_k(nK11_raw, 1.0, current_pos)
            let nV11 = linear(nh, L11.attn_v.weight)
            kv_cache.append(11, nK11, nV11)
            let KV11 = kv_cache.get_k(11)
            let KV11_V = kv_cache.get_v(11)
            let nh = transformer_layer(nh, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, KV11, KV11_V, current_pos, 1.0)

            // Layer 12
            let nK12_raw = linear(nh, L12.attn_k.weight)
            let nK12 = apply_rope_k(nK12_raw, 1.0, current_pos)
            let nV12 = linear(nh, L12.attn_v.weight)
            kv_cache.append(12, nK12, nV12)
            let KV12 = kv_cache.get_k(12)
            let KV12_V = kv_cache.get_v(12)
            let nh = transformer_layer(nh, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, KV12, KV12_V, current_pos, 1.0)

            // Layer 13
            let nK13_raw = linear(nh, L13.attn_k.weight)
            let nK13 = apply_rope_k(nK13_raw, 1.0, current_pos)
            let nV13 = linear(nh, L13.attn_v.weight)
            kv_cache.append(13, nK13, nV13)
            let KV13 = kv_cache.get_k(13)
            let KV13_V = kv_cache.get_v(13)
            let nh = transformer_layer(nh, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, KV13, KV13_V, current_pos, 1.0)

            // Layer 14
            let nK14_raw = linear(nh, L14.attn_k.weight)
            let nK14 = apply_rope_k(nK14_raw, 1.0, current_pos)
            let nV14 = linear(nh, L14.attn_v.weight)
            kv_cache.append(14, nK14, nV14)
            let KV14 = kv_cache.get_k(14)
            let KV14_V = kv_cache.get_v(14)
            let nh = transformer_layer(nh, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, KV14, KV14_V, current_pos, 1.0)

            // Layer 15
            let nK15_raw = linear(nh, L15.attn_k.weight)
            let nK15 = apply_rope_k(nK15_raw, 1.0, current_pos)
            let nV15 = linear(nh, L15.attn_v.weight)
            kv_cache.append(15, nK15, nV15)
            let KV15 = kv_cache.get_k(15)
            let KV15_V = kv_cache.get_v(15)
            let nh = transformer_layer(nh, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, KV15, KV15_V, current_pos, 1.0)

            // Layer 16
            let nK16_raw = linear(nh, L16.attn_k.weight)
            let nK16 = apply_rope_k(nK16_raw, 1.0, current_pos)
            let nV16 = linear(nh, L16.attn_v.weight)
            kv_cache.append(16, nK16, nV16)
            let KV16 = kv_cache.get_k(16)
            let KV16_V = kv_cache.get_v(16)
            let nh = transformer_layer(nh, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, KV16, KV16_V, current_pos, 1.0)

            // Layer 17
            let nK17_raw = linear(nh, L17.attn_k.weight)
            let nK17 = apply_rope_k(nK17_raw, 1.0, current_pos)
            let nV17 = linear(nh, L17.attn_v.weight)
            kv_cache.append(17, nK17, nV17)
            let KV17 = kv_cache.get_k(17)
            let KV17_V = kv_cache.get_v(17)
            let nh = transformer_layer(nh, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, KV17, KV17_V, current_pos, 1.0)

            // Layer 18
            let nK18_raw = linear(nh, L18.attn_k.weight)
            let nK18 = apply_rope_k(nK18_raw, 1.0, current_pos)
            let nV18 = linear(nh, L18.attn_v.weight)
            kv_cache.append(18, nK18, nV18)
            let KV18 = kv_cache.get_k(18)
            let KV18_V = kv_cache.get_v(18)
            let nh = transformer_layer(nh, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, KV18, KV18_V, current_pos, 1.0)

            // Layer 19
            let nK19_raw = linear(nh, L19.attn_k.weight)
            let nK19 = apply_rope_k(nK19_raw, 1.0, current_pos)
            let nV19 = linear(nh, L19.attn_v.weight)
            kv_cache.append(19, nK19, nV19)
            let KV19 = kv_cache.get_k(19)
            let KV19_V = kv_cache.get_v(19)
            let nh = transformer_layer(nh, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, KV19, KV19_V, current_pos, 1.0)

            // Layer 20
            let nK20_raw = linear(nh, L20.attn_k.weight)
            let nK20 = apply_rope_k(nK20_raw, 1.0, current_pos)
            let nV20 = linear(nh, L20.attn_v.weight)
            kv_cache.append(20, nK20, nV20)
            let KV20 = kv_cache.get_k(20)
            let KV20_V = kv_cache.get_v(20)
            let nh = transformer_layer(nh, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, KV20, KV20_V, current_pos, 1.0)

            // Layer 21 (final layer)
            let nK21_raw = linear(nh, L21.attn_k.weight)
            let nK21 = apply_rope_k(nK21_raw, 1.0, current_pos)
            let nV21 = linear(nh, L21.attn_v.weight)
            kv_cache.append(21, nK21, nV21)
            let KV21 = kv_cache.get_k(21)
            let KV21_V = kv_cache.get_v(21)
            let nh = transformer_layer(nh, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, KV21, KV21_V, current_pos, 1.0)

            // Increment position for next token
            current_pos = current_pos + 1.0

            // Final output projection to vocabulary logits
            let norm_new = rms_norm(nh, output_norm)

            // Extract last token (nh is [1, 2048], extract to [2048] then reshape to [1, 2048])
            let last_decode = slice_last(norm_new, 0)  // [1, 2048] → [2048]
            let last_decode_2d = reshape(last_decode, [1.0, 2048.0])  // [2048] → [1, 2048]

            current_logits = linear(last_decode_2d, tok_embd)  // Weight tying for decode
        }
    }

    print("")
    print("")
    print("============================================================================")
    print("Generation Complete")
    print("============================================================================")
    print("Tokens generated: {}", token_count)
    print("")
    print("Architecture:")
    print("  • Explicit Grouped Query Attention (32 Q heads / 4 KV heads = 8:1)")
    print("  • SwiGLU FFN activation")
    print("  • RoPE positional embeddings")
    print("  • RMSNorm pre-normalization")
    print("  • KV caching for efficient generation")
    print("")
}
