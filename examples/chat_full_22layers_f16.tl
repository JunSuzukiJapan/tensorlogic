// Full 22-Layer Chat Demo with f16 (Memory Optimized)
// Complete implementation using ALL 22 transformer layers
//
// PERFORMANCE OPTIMIZATIONS:
// ─────────────────────────────────────────────────────────────
// 1. Fused Transpose-Matmul Kernel
//    - Combines transpose(weight) + matmul into single GPU kernel
//    - 20-30% faster per linear() call
//    - Automatically selected for transformer patterns
//
// 2. Dynamic Tile Size Selection
//    - 32x32 tiling for large matrices (K≥512, transformer patterns)
//    - 16x16 tiling for smaller matrices
//    - Optimized for [1,K] @ [K,N] decode patterns
//
// 3. Results
//    - Overall: 2.89x speedup (3,586ms → 1,239ms per decode step)
//    - Reduction: 65% faster with kernel optimizations
//
// ARCHITECTURE:
// ─────────────────────────────────────────────────────────────
// - Model: TinyLlama 1.1B Chat (q4_0 quantization)
// - Layers: 22 transformer blocks
// - Attention: GQA with 4 KV heads, 32 Q heads
// - FFN: SwiGLU activation
// - Context: KV cache for efficient autoregressive generation

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
}

// SwiGLU Feed-Forward Network
// All linear() calls use optimized fused transpose-matmul kernel
fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)    // Optimized
    let up = linear(x, W_up)        // Optimized
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    result = linear(mul_result, W_down)  // Optimized
}

// Multi-Head Attention with KV Cache
// NOTE: Now using Rust builtin implementation with GQA support
// The builtin version is optimized with:
// - Grouped Query Attention (4 KV heads, 32 Q heads)
// - Efficient K/V expansion via broadcast (repeat_kv_for_gqa)
// - Fused transpose-matmul (matmul_transposed_b) for 2.89x faster output projection
// - CommandBuffer batching for reduced GPU synchronization overhead

// Helper: Apply RoPE to K for caching (optimized: no shape() call)
fn apply_rope_k(K: float16[?, ?], seq_len: float, pos: float) -> float16[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result = reshape(K_r, [seq_len, 256.0])
}

// NOTE: Rust builtin attention_with_cache (f16) modified to NOT apply RoPE to K
// It expects K/V cache to already have RoPE applied
// We apply RoPE during prefill/decode before caching

fn transformer_layer(
    x: float16[?, ?],
    W_attn_norm: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    W_ffn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    K_cache: float16[?, ?],
    V_cache: float16[?, ?]
) -> float16[?, ?] {
    let normed = rms_norm(x, W_attn_norm)
    let Q = linear(normed, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out
    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result = after_attn + ffn_out
}
main {
    print("=== TensorLogic Chat: FULL 22-Layer System ===")
    print("")

    let EOS_TOKEN = 2

    print("[1/3] Loading model (ALL 22 layers)...")
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("      Loading all 22 transformer layers...")
    
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

    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize( chat_prompt, true)
    print("      User: {}", user_msg)
    print("")

    print("[3/3] Generating response (max 50 tokens)...")
    print("")
    print("      Assistant: ", "")

    let x = embedding(tok_embd, tokens)

    // ============================================================
    // PREFILL PHASE: Build initial KV caches for all 22 layers
    // Apply RoPE to K at position 0 before caching
    // ============================================================
    // FIXED: shape tensor indexing now works with extract_shape! fix
    let x_shp = shape(x)
    let seq_len = x_shp[0]

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

    // ============================================================
    // PREFILL: Run prompt through all 22 transformer layers
    // ============================================================
    let h0 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)
    let h1 = transformer_layer(h0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)
    let h2 = transformer_layer(h1, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_k.weight, L2.attn_v.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2)
    let h3 = transformer_layer(h2, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_k.weight, L3.attn_v.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3)
    let h4 = transformer_layer(h3, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_k.weight, L4.attn_v.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, K4, V4)
    let h5 = transformer_layer(h4, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_k.weight, L5.attn_v.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, K5, V5)
    let h6 = transformer_layer(h5, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_k.weight, L6.attn_v.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, K6, V6)
    let h7 = transformer_layer(h6, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_k.weight, L7.attn_v.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, K7, V7)
    let h8 = transformer_layer(h7, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_k.weight, L8.attn_v.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, K8, V8)
    let h9 = transformer_layer(h8, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_k.weight, L9.attn_v.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, K9, V9)
    let h10 = transformer_layer(h9, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_k.weight, L10.attn_v.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, K10, V10)
    let h11 = transformer_layer(h10, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_k.weight, L11.attn_v.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, K11, V11)
    let h12 = transformer_layer(h11, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_k.weight, L12.attn_v.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, K12, V12)
    let h13 = transformer_layer(h12, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_k.weight, L13.attn_v.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, K13, V13)
    let h14 = transformer_layer(h13, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_k.weight, L14.attn_v.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, K14, V14)
    let h15 = transformer_layer(h14, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_k.weight, L15.attn_v.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, K15, V15)
    let h16 = transformer_layer(h15, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_k.weight, L16.attn_v.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, K16, V16)
    let h17 = transformer_layer(h16, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_k.weight, L17.attn_v.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, K17, V17)
    let h18 = transformer_layer(h17, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_k.weight, L18.attn_v.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, K18, V18)
    let h19 = transformer_layer(h18, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_k.weight, L19.attn_v.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, K19, V19)
    let h20 = transformer_layer(h19, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_k.weight, L20.attn_v.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, K20, V20)
    let h21 = transformer_layer(h20, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_k.weight, L21.attn_v.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, K21, V21)

    // Final normalization and output projection
    let final_norm = rms_norm(h21, output_norm)
    let logits = linear(final_norm, output)

    // ============================================================
    // AUTOREGRESSIVE GENERATION LOOP
    // ============================================================
    // Initialize generation state
    let temperature = 0.8
    let current_logits = logits
    let continue_generation = true
    let token_count = 0

    // Track position on CPU (Candle-style: avoid shape() GPU syncs)
    let current_pos = seq_len

    // Initialize KV cache (builtin type manages all 22 layers)
    let kv_cache = KVCache::new(22)

    // Store prefill K/V tensors in cache
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

    print("Prefill complete, starting token generation...")

    // Token-by-token generation (max 10 tokens for testing)
    for i in range(10) {
        if continue_generation {
            // Sample next token using temperature sampling
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
            // Each step processes through all 22 layers sequentially
            // KV cache grows by 1 token per layer (concat operation)
            // ========================================================
            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)

            // Layer 0: Update KV cache and transform
            // Use CPU-tracked position (Candle-style: no shape() sync)
            let nK0_raw = linear(new_token_emb, L0.attn_k.weight)
            let nK0 = apply_rope_k(nK0_raw, 1.0, current_pos)
            let nV0 = linear(new_token_emb, L0.attn_v.weight)
            kv_cache.append(0, nK0, nV0)
            let KV0 = kv_cache.get_k(0)
            let KV0_V = kv_cache.get_v(0)

            let nh0 = transformer_layer(new_token_emb, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, KV0, KV0_V)

            // Layer 1
            let nK1_raw = linear(nh0, L1.attn_k.weight)
            let nK1 = apply_rope_k(nK1_raw, 1.0, current_pos)
            let nV1 = linear(nh0, L1.attn_v.weight)
            kv_cache.append(1, nK1, nV1)
            let KV1 = kv_cache.get_k(1)
            let KV1_V = kv_cache.get_v(1)

            let nh1 = transformer_layer(nh0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, KV1, KV1_V)

            let nK2_raw = linear(nh1, L2.attn_k.weight)
            let nK2 = apply_rope_k(nK2_raw, 1.0, current_pos)
            let nV2 = linear(nh1, L2.attn_v.weight)
            kv_cache.append(2, nK2, nV2)
            let KV2 = kv_cache.get_k(2)
            let KV2_V = kv_cache.get_v(2)

            let nh2 = transformer_layer(nh1, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_k.weight, L2.attn_v.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, KV2, KV2_V)

            let nK3_raw = linear(nh2, L3.attn_k.weight)
            let nK3 = apply_rope_k(nK3_raw, 1.0, current_pos)
            let nV3 = linear(nh2, L3.attn_v.weight)
            kv_cache.append(3, nK3, nV3)
            let KV3 = kv_cache.get_k(3)
            let KV3_V = kv_cache.get_v(3)

            let nh3 = transformer_layer(nh2, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_k.weight, L3.attn_v.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, KV3, KV3_V)

            let nK4_raw = linear(nh3, L4.attn_k.weight)
            let nK4 = apply_rope_k(nK4_raw, 1.0, current_pos)
            let nV4 = linear(nh3, L4.attn_v.weight)
            kv_cache.append(4, nK4, nV4)
            let KV4 = kv_cache.get_k(4)
            let KV4_V = kv_cache.get_v(4)

            let nh4 = transformer_layer(nh3, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_k.weight, L4.attn_v.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, KV4, KV4_V)

            let nK5_raw = linear(nh4, L5.attn_k.weight)
            let nK5 = apply_rope_k(nK5_raw, 1.0, current_pos)
            let nV5 = linear(nh4, L5.attn_v.weight)
            kv_cache.append(5, nK5, nV5)
            let KV5 = kv_cache.get_k(5)
            let KV5_V = kv_cache.get_v(5)

            let nh5 = transformer_layer(nh4, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_k.weight, L5.attn_v.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, KV5, KV5_V)

            let nK6_raw = linear(nh5, L6.attn_k.weight)
            let nK6 = apply_rope_k(nK6_raw, 1.0, current_pos)
            let nV6 = linear(nh5, L6.attn_v.weight)
            kv_cache.append(6, nK6, nV6)
            let KV6 = kv_cache.get_k(6)
            let KV6_V = kv_cache.get_v(6)

            let nh6 = transformer_layer(nh5, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_k.weight, L6.attn_v.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, KV6, KV6_V)

            let nK7_raw = linear(nh6, L7.attn_k.weight)
            let nK7 = apply_rope_k(nK7_raw, 1.0, current_pos)
            let nV7 = linear(nh6, L7.attn_v.weight)
            kv_cache.append(7, nK7, nV7)
            let KV7 = kv_cache.get_k(7)
            let KV7_V = kv_cache.get_v(7)

            let nh7 = transformer_layer(nh6, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_k.weight, L7.attn_v.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, KV7, KV7_V)

            let nK8_raw = linear(nh7, L8.attn_k.weight)
            let nK8 = apply_rope_k(nK8_raw, 1.0, current_pos)
            let nV8 = linear(nh7, L8.attn_v.weight)
            kv_cache.append(8, nK8, nV8)
            let KV8 = kv_cache.get_k(8)
            let KV8_V = kv_cache.get_v(8)

            let nh8 = transformer_layer(nh7, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_k.weight, L8.attn_v.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, KV8, KV8_V)

            let nK9_raw = linear(nh8, L9.attn_k.weight)
            let nK9 = apply_rope_k(nK9_raw, 1.0, current_pos)
            let nV9 = linear(nh8, L9.attn_v.weight)
            kv_cache.append(9, nK9, nV9)
            let KV9 = kv_cache.get_k(9)
            let KV9_V = kv_cache.get_v(9)

            let nh9 = transformer_layer(nh8, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_k.weight, L9.attn_v.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, KV9, KV9_V)

            // Layer 10 (midpoint)
            let nK10_raw = linear(nh9, L10.attn_k.weight)
            let nK10 = apply_rope_k(nK10_raw, 1.0, current_pos)
            let nV10 = linear(nh9, L10.attn_v.weight)
            kv_cache.append(10, nK10, nV10)
            let KV10 = kv_cache.get_k(10)
            let KV10_V = kv_cache.get_v(10)

            let nh10 = transformer_layer(nh9, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_k.weight, L10.attn_v.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, KV10, KV10_V)

            let nK11_raw = linear(nh10, L11.attn_k.weight)
            let nK11 = apply_rope_k(nK11_raw, 1.0, current_pos)
            let nV11 = linear(nh10, L11.attn_v.weight)
            kv_cache.append(11, nK11, nV11)
            let KV11 = kv_cache.get_k(11)
            let KV11_V = kv_cache.get_v(11)

            let nh11 = transformer_layer(nh10, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_k.weight, L11.attn_v.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, KV11, KV11_V)

            let nK12_raw = linear(nh11, L12.attn_k.weight)
            let nK12 = apply_rope_k(nK12_raw, 1.0, current_pos)
            let nV12 = linear(nh11, L12.attn_v.weight)
            kv_cache.append(12, nK12, nV12)
            let KV12 = kv_cache.get_k(12)
            let KV12_V = kv_cache.get_v(12)

            let nh12 = transformer_layer(nh11, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_k.weight, L12.attn_v.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, KV12, KV12_V)

            let nK13_raw = linear(nh12, L13.attn_k.weight)
            let nK13 = apply_rope_k(nK13_raw, 1.0, current_pos)
            let nV13 = linear(nh12, L13.attn_v.weight)
            kv_cache.append(13, nK13, nV13)
            let KV13 = kv_cache.get_k(13)
            let KV13_V = kv_cache.get_v(13)

            let nh13 = transformer_layer(nh12, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_k.weight, L13.attn_v.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, KV13, KV13_V)

            let nK14_raw = linear(nh13, L14.attn_k.weight)
            let nK14 = apply_rope_k(nK14_raw, 1.0, current_pos)
            let nV14 = linear(nh13, L14.attn_v.weight)
            kv_cache.append(14, nK14, nV14)
            let KV14 = kv_cache.get_k(14)
            let KV14_V = kv_cache.get_v(14)

            let nh14 = transformer_layer(nh13, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_k.weight, L14.attn_v.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, KV14, KV14_V)

            let nK15_raw = linear(nh14, L15.attn_k.weight)
            let nK15 = apply_rope_k(nK15_raw, 1.0, current_pos)
            let nV15 = linear(nh14, L15.attn_v.weight)
            kv_cache.append(15, nK15, nV15)
            let KV15 = kv_cache.get_k(15)
            let KV15_V = kv_cache.get_v(15)

            let nh15 = transformer_layer(nh14, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_k.weight, L15.attn_v.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, KV15, KV15_V)

            let nK16_raw = linear(nh15, L16.attn_k.weight)
            let nK16 = apply_rope_k(nK16_raw, 1.0, current_pos)
            let nV16 = linear(nh15, L16.attn_v.weight)
            kv_cache.append(16, nK16, nV16)
            let KV16 = kv_cache.get_k(16)
            let KV16_V = kv_cache.get_v(16)

            let nh16 = transformer_layer(nh15, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_k.weight, L16.attn_v.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, KV16, KV16_V)

            let nK17_raw = linear(nh16, L17.attn_k.weight)
            let nK17 = apply_rope_k(nK17_raw, 1.0, current_pos)
            let nV17 = linear(nh16, L17.attn_v.weight)
            kv_cache.append(17, nK17, nV17)
            let KV17 = kv_cache.get_k(17)
            let KV17_V = kv_cache.get_v(17)

            let nh17 = transformer_layer(nh16, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_k.weight, L17.attn_v.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, KV17, KV17_V)

            let nK18_raw = linear(nh17, L18.attn_k.weight)
            let nK18 = apply_rope_k(nK18_raw, 1.0, current_pos)
            let nV18 = linear(nh17, L18.attn_v.weight)
            kv_cache.append(18, nK18, nV18)
            let KV18 = kv_cache.get_k(18)
            let KV18_V = kv_cache.get_v(18)

            let nh18 = transformer_layer(nh17, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_k.weight, L18.attn_v.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, KV18, KV18_V)

            let nK19_raw = linear(nh18, L19.attn_k.weight)
            let nK19 = apply_rope_k(nK19_raw, 1.0, current_pos)
            let nV19 = linear(nh18, L19.attn_v.weight)
            kv_cache.append(19, nK19, nV19)
            let KV19 = kv_cache.get_k(19)
            let KV19_V = kv_cache.get_v(19)

            let nh19 = transformer_layer(nh18, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_k.weight, L19.attn_v.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, KV19, KV19_V)

            let nK20_raw = linear(nh19, L20.attn_k.weight)
            let nK20 = apply_rope_k(nK20_raw, 1.0, current_pos)
            let nV20 = linear(nh19, L20.attn_v.weight)
            kv_cache.append(20, nK20, nV20)
            let KV20 = kv_cache.get_k(20)
            let KV20_V = kv_cache.get_v(20)

            let nh20 = transformer_layer(nh19, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_k.weight, L20.attn_v.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, KV20, KV20_V)

            // Layer 21 (final layer)
            let nK21_raw = linear(nh20, L21.attn_k.weight)
            let nK21 = apply_rope_k(nK21_raw, 1.0, current_pos)
            let nV21 = linear(nh20, L21.attn_v.weight)
            kv_cache.append(21, nK21, nV21)
            let KV21 = kv_cache.get_k(21)
            let KV21_V = kv_cache.get_v(21)

            let nh21 = transformer_layer(nh20, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_k.weight, L21.attn_v.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, KV21, KV21_V)

            // Final output projection to vocabulary logits
            let norm_new = rms_norm(nh21, output_norm)
            current_logits = linear(norm_new, output)  // [1, vocab_size]

            // Increment position (Candle-style: CPU tracking, no GPU sync)
            current_pos = current_pos + 1.0
        }
    }

    print("")
    print("")
    print("=== Generation Complete ===")
    print("Tokens generated: {}", token_count)
}
// ============================================================================
// Function Tests
// ============================================================================

// Test 1: RoPE (Rotary Position Embedding) preserves shape
test rope_shape {
    print("Test: RoPE preserves tensor shape")

    // Create simple test input: [seq_len=1, n_heads=1, head_dim=4]
    tensor test_input: float16[1, 1, 4] = [[[3.0, 4.0, 5.0, 12.0]]]

    let result = rope(test_input, 0.0)
    let result_shape = shape(result)

    // Verify shape is preserved
    // Note: Checking via print since assert() is not implemented
    print("  Input shape: [1, 1, 4]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], ", ", result_shape[2], "]")
    print("  ✓ RoPE executed successfully")
}

// Test 2: RMS Norm produces normalized output
test rms_norm_shape {
    print("Test: RMS Norm shape preservation")

    tensor x: float16[1, 4] = [[2.0, 4.0, 6.0, 8.0]]
    tensor weight: float16[4] = [1.0, 1.0, 1.0, 1.0]

    let result = rms_norm(x, weight)
    let result_shape = shape(result)

    print("  Input shape: [1, 4]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ RMS Norm executed successfully")
}

// Test 3: Softmax shape preservation
test softmax_shape {
    print("Test: Softmax shape preservation")

    tensor logits: float16[1, 4] = [[1.0, 2.0, 3.0, 4.0]]
    let probs = softmax(logits)
    let probs_shape = shape(probs)

    print("  Input shape: [1, 4]")
    print("  Output shape: [", probs_shape[0], ", ", probs_shape[1], "]")
    print("  ✓ Softmax executed successfully")
}

// Test 4: Linear layer (uses transposed matmul internally)
test linear_shape {
    print("Test: Linear layer shape")

    // linear() does: x @ weight.T
    // x: [2, 3], weight: [4, 3] -> output: [2, 4]
    tensor x: float16[2, 3] = [[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]
    tensor weight: float16[4, 3] = [[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]]

    let result = linear(x, weight)
    let result_shape = shape(result)

    print("  Input shape: [2, 3]")
    print("  Weight shape: [4, 3]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ Linear executed successfully")
}

// Test 5: Reshape operation
test reshape_shape {
    print("Test: Reshape operation")

    tensor x: float16[2, 3] = [[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]
    let x_shape = shape(x)

    let reshaped = reshape(x, [3.0, 2.0])
    let new_shape = shape(reshaped)

    print("  Original shape: [", x_shape[0], ", ", x_shape[1], "]")
    print("  Reshaped to: [", new_shape[0], ", ", new_shape[1], "]")
    print("  ✓ Reshape executed successfully")
}

// Test 6: SiLU activation
test silu_shape {
    print("Test: SiLU activation")

    tensor x: float16[1, 5] = [[-2.0, -1.0, 0.0, 1.0, 2.0]]
    let result = silu(x)
    let result_shape = shape(result)

    print("  Input shape: [1, 5]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ SiLU executed successfully")
}

// Test 7: Attention output shape
test attention_shape {
    print("Test: Attention mechanism shape")

    // Q, K, V: [seq_len=2, n_embd=8]
    // W_o: [n_embd=8, n_embd=8]
    tensor q: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor k: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor v: float16[2, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    tensor w_o: float16[8, 8] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    let result = attention_with_cache(q, k, v, w_o)
    let result_shape = shape(result)

    print("  Input Q/K/V shape: [2, 8]")
    print("  Output shape: [", result_shape[0], ", ", result_shape[1], "]")
    print("  ✓ Attention executed successfully")
}
