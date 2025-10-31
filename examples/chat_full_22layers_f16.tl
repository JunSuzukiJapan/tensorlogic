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
    result := x * sigmoid(x)
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
    result := linear(mul_result, W_down)  // Optimized
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
    result := reshape(K_r, [seq_len, 256.0])
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
    result := after_attn + ffn_out
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
    // Get sequence length once (optimization: avoid repeated shape() calls)
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
    
    // Initialize KV caches
    let KV0 = K0
    let KV0_V = V0
    let KV1 = K1
    let KV1_V = V1
    let KV2 = K2
    let KV2_V = V2
    let KV3 = K3
    let KV3_V = V3
    let KV4 = K4
    let KV4_V = V4
    let KV5 = K5
    let KV5_V = V5
    let KV6 = K6
    let KV6_V = V6
    let KV7 = K7
    let KV7_V = V7
    let KV8 = K8
    let KV8_V = V8
    let KV9 = K9
    let KV9_V = V9
    let KV10 = K10
    let KV10_V = V10
    let KV11 = K11
    let KV11_V = V11
    let KV12 = K12
    let KV12_V = V12
    let KV13 = K13
    let KV13_V = V13
    let KV14 = K14
    let KV14_V = V14
    let KV15 = K15
    let KV15_V = V15
    let KV16 = K16
    let KV16_V = V16
    let KV17 = K17
    let KV17_V = V17
    let KV18 = K18
    let KV18_V = V18
    let KV19 = K19
    let KV19_V = V19
    let KV20 = K20
    let KV20_V = V20
    let KV21 = K21
    let KV21_V = V21

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
            let KV0_shp = shape(KV0)
            let pos0 = KV0_shp[0]
            let nK0_raw = linear(new_token_emb, L0.attn_k.weight)
            let nK0 = apply_rope_k(nK0_raw, 1.0, pos0)
            let nV0 = linear(new_token_emb, L0.attn_v.weight)
            KV0 = concat(KV0, nK0, 0.0)
            KV0_V = concat(KV0_V, nV0, 0.0)

            let nh0 = transformer_layer(new_token_emb, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, KV0, KV0_V)

            // Layer 1
            let KV1_shp = shape(KV1)
            let pos1 = KV1_shp[0]
            let nK1_raw = linear(nh0, L1.attn_k.weight)
            let nK1 = apply_rope_k(nK1_raw, 1.0, pos1)
            let nV1 = linear(nh0, L1.attn_v.weight)
            KV1 = concat(KV1, nK1, 0.0)
            KV1_V = concat(KV1_V, nV1, 0.0)
            
            let nh1 = transformer_layer(nh0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, KV1, KV1_V)
            
            let KV2_shp = shape(KV2)
            let pos2 = KV2_shp[0]
            let nK2_raw = linear(nh1, L2.attn_k.weight)
            let nK2 = apply_rope_k(nK2_raw, 1.0, pos2)
            let nV2 = linear(nh1, L2.attn_v.weight)
            KV2 = concat(KV2, nK2, 0.0)
            KV2_V = concat(KV2_V, nV2, 0.0)

            let nh2 = transformer_layer(nh1, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_k.weight, L2.attn_v.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, KV2, KV2_V)

            let KV3_shp = shape(KV3)
            let pos3 = KV3_shp[0]
            let nK3_raw = linear(nh2, L3.attn_k.weight)
            let nK3 = apply_rope_k(nK3_raw, 1.0, pos3)
            let nV3 = linear(nh2, L3.attn_v.weight)
            KV3 = concat(KV3, nK3, 0.0)
            KV3_V = concat(KV3_V, nV3, 0.0)

            let nh3 = transformer_layer(nh2, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_k.weight, L3.attn_v.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, KV3, KV3_V)

            let KV4_shp = shape(KV4)
            let pos4 = KV4_shp[0]
            let nK4_raw = linear(nh3, L4.attn_k.weight)
            let nK4 = apply_rope_k(nK4_raw, 1.0, pos4)
            let nV4 = linear(nh3, L4.attn_v.weight)
            KV4 = concat(KV4, nK4, 0.0)
            KV4_V = concat(KV4_V, nV4, 0.0)

            let nh4 = transformer_layer(nh3, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_k.weight, L4.attn_v.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, KV4, KV4_V)

            let KV5_shp = shape(KV5)
            let pos5 = KV5_shp[0]
            let nK5_raw = linear(nh4, L5.attn_k.weight)
            let nK5 = apply_rope_k(nK5_raw, 1.0, pos5)
            let nV5 = linear(nh4, L5.attn_v.weight)
            KV5 = concat(KV5, nK5, 0.0)
            KV5_V = concat(KV5_V, nV5, 0.0)
            
            let nh5 = transformer_layer(nh4, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_k.weight, L5.attn_v.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, KV5, KV5_V)
            
            let KV6_shp = shape(KV6)
            let pos6 = KV6_shp[0]
            let nK6_raw = linear(nh5, L6.attn_k.weight)
            let nK6 = apply_rope_k(nK6_raw, 1.0, pos6)
            let nV6 = linear(nh5, L6.attn_v.weight)
            KV6 = concat(KV6, nK6, 0.0)
            KV6_V = concat(KV6_V, nV6, 0.0)

            let nh6 = transformer_layer(nh5, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_k.weight, L6.attn_v.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, KV6, KV6_V)

            let KV7_shp = shape(KV7)
            let pos7 = KV7_shp[0]
            let nK7_raw = linear(nh6, L7.attn_k.weight)
            let nK7 = apply_rope_k(nK7_raw, 1.0, pos7)
            let nV7 = linear(nh6, L7.attn_v.weight)
            KV7 = concat(KV7, nK7, 0.0)
            KV7_V = concat(KV7_V, nV7, 0.0)

            let nh7 = transformer_layer(nh6, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_k.weight, L7.attn_v.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, KV7, KV7_V)

            let KV8_shp = shape(KV8)
            let pos8 = KV8_shp[0]
            let nK8_raw = linear(nh7, L8.attn_k.weight)
            let nK8 = apply_rope_k(nK8_raw, 1.0, pos8)
            let nV8 = linear(nh7, L8.attn_v.weight)
            KV8 = concat(KV8, nK8, 0.0)
            KV8_V = concat(KV8_V, nV8, 0.0)

            let nh8 = transformer_layer(nh7, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_k.weight, L8.attn_v.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, KV8, KV8_V)

            let KV9_shp = shape(KV9)
            let pos9 = KV9_shp[0]
            let nK9_raw = linear(nh8, L9.attn_k.weight)
            let nK9 = apply_rope_k(nK9_raw, 1.0, pos9)
            let nV9 = linear(nh8, L9.attn_v.weight)
            KV9 = concat(KV9, nK9, 0.0)
            KV9_V = concat(KV9_V, nV9, 0.0)

            let nh9 = transformer_layer(nh8, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_k.weight, L9.attn_v.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, KV9, KV9_V)

            // Layer 10 (midpoint)
            let KV10_shp = shape(KV10)
            let pos10 = KV10_shp[0]
            let nK10_raw = linear(nh9, L10.attn_k.weight)
            let nK10 = apply_rope_k(nK10_raw, 1.0, pos10)
            let nV10 = linear(nh9, L10.attn_v.weight)
            KV10 = concat(KV10, nK10, 0.0)
            KV10_V = concat(KV10_V, nV10, 0.0)
            
            let nh10 = transformer_layer(nh9, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_k.weight, L10.attn_v.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, KV10, KV10_V)
            
            let KV11_shp = shape(KV11)
            let pos11 = KV11_shp[0]
            let nK11_raw = linear(nh10, L11.attn_k.weight)
            let nK11 = apply_rope_k(nK11_raw, 1.0, pos11)
            let nV11 = linear(nh10, L11.attn_v.weight)
            KV11 = concat(KV11, nK11, 0.0)
            KV11_V = concat(KV11_V, nV11, 0.0)

            let nh11 = transformer_layer(nh10, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_k.weight, L11.attn_v.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, KV11, KV11_V)

            let KV12_shp = shape(KV12)
            let pos12 = KV12_shp[0]
            let nK12_raw = linear(nh11, L12.attn_k.weight)
            let nK12 = apply_rope_k(nK12_raw, 1.0, pos12)
            let nV12 = linear(nh11, L12.attn_v.weight)
            KV12 = concat(KV12, nK12, 0.0)
            KV12_V = concat(KV12_V, nV12, 0.0)

            let nh12 = transformer_layer(nh11, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_k.weight, L12.attn_v.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, KV12, KV12_V)

            let KV13_shp = shape(KV13)
            let pos13 = KV13_shp[0]
            let nK13_raw = linear(nh12, L13.attn_k.weight)
            let nK13 = apply_rope_k(nK13_raw, 1.0, pos13)
            let nV13 = linear(nh12, L13.attn_v.weight)
            KV13 = concat(KV13, nK13, 0.0)
            KV13_V = concat(KV13_V, nV13, 0.0)

            let nh13 = transformer_layer(nh12, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_k.weight, L13.attn_v.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, KV13, KV13_V)

            let KV14_shp = shape(KV14)
            let pos14 = KV14_shp[0]
            let nK14_raw = linear(nh13, L14.attn_k.weight)
            let nK14 = apply_rope_k(nK14_raw, 1.0, pos14)
            let nV14 = linear(nh13, L14.attn_v.weight)
            KV14 = concat(KV14, nK14, 0.0)
            KV14_V = concat(KV14_V, nV14, 0.0)

            let nh14 = transformer_layer(nh13, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_k.weight, L14.attn_v.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, KV14, KV14_V)

            let KV15_shp = shape(KV15)
            let pos15 = KV15_shp[0]
            let nK15_raw = linear(nh14, L15.attn_k.weight)
            let nK15 = apply_rope_k(nK15_raw, 1.0, pos15)
            let nV15 = linear(nh14, L15.attn_v.weight)
            KV15 = concat(KV15, nK15, 0.0)
            KV15_V = concat(KV15_V, nV15, 0.0)
            
            let nh15 = transformer_layer(nh14, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_k.weight, L15.attn_v.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, KV15, KV15_V)
            
            let KV16_shp = shape(KV16)
            let pos16 = KV16_shp[0]
            let nK16_raw = linear(nh15, L16.attn_k.weight)
            let nK16 = apply_rope_k(nK16_raw, 1.0, pos16)
            let nV16 = linear(nh15, L16.attn_v.weight)
            KV16 = concat(KV16, nK16, 0.0)
            KV16_V = concat(KV16_V, nV16, 0.0)

            let nh16 = transformer_layer(nh15, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_k.weight, L16.attn_v.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, KV16, KV16_V)

            let KV17_shp = shape(KV17)
            let pos17 = KV17_shp[0]
            let nK17_raw = linear(nh16, L17.attn_k.weight)
            let nK17 = apply_rope_k(nK17_raw, 1.0, pos17)
            let nV17 = linear(nh16, L17.attn_v.weight)
            KV17 = concat(KV17, nK17, 0.0)
            KV17_V = concat(KV17_V, nV17, 0.0)

            let nh17 = transformer_layer(nh16, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_k.weight, L17.attn_v.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, KV17, KV17_V)

            let KV18_shp = shape(KV18)
            let pos18 = KV18_shp[0]
            let nK18_raw = linear(nh17, L18.attn_k.weight)
            let nK18 = apply_rope_k(nK18_raw, 1.0, pos18)
            let nV18 = linear(nh17, L18.attn_v.weight)
            KV18 = concat(KV18, nK18, 0.0)
            KV18_V = concat(KV18_V, nV18, 0.0)

            let nh18 = transformer_layer(nh17, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_k.weight, L18.attn_v.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, KV18, KV18_V)

            let KV19_shp = shape(KV19)
            let pos19 = KV19_shp[0]
            let nK19_raw = linear(nh18, L19.attn_k.weight)
            let nK19 = apply_rope_k(nK19_raw, 1.0, pos19)
            let nV19 = linear(nh18, L19.attn_v.weight)
            KV19 = concat(KV19, nK19, 0.0)
            KV19_V = concat(KV19_V, nV19, 0.0)

            let nh19 = transformer_layer(nh18, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_k.weight, L19.attn_v.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, KV19, KV19_V)

            let KV20_shp = shape(KV20)
            let pos20 = KV20_shp[0]
            let nK20_raw = linear(nh19, L20.attn_k.weight)
            let nK20 = apply_rope_k(nK20_raw, 1.0, pos20)
            let nV20 = linear(nh19, L20.attn_v.weight)
            KV20 = concat(KV20, nK20, 0.0)
            KV20_V = concat(KV20_V, nV20, 0.0)

            let nh20 = transformer_layer(nh19, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_k.weight, L20.attn_v.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, KV20, KV20_V)

            // Layer 21 (final layer)
            let KV21_shp = shape(KV21)
            let pos21 = KV21_shp[0]
            let nK21_raw = linear(nh20, L21.attn_k.weight)
            let nK21 = apply_rope_k(nK21_raw, 1.0, pos21)
            let nV21 = linear(nh20, L21.attn_v.weight)
            KV21 = concat(KV21, nK21, 0.0)
            KV21_V = concat(KV21_V, nV21, 0.0)
            
            let nh21 = transformer_layer(nh20, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_k.weight, L21.attn_v.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, KV21, KV21_V)

            // Final output projection to vocabulary logits
            let norm_new = rms_norm(nh21, output_norm)
            current_logits = linear(norm_new, output)  // [1, vocab_size]
        }
    }

    print("")
    print("")
    print("=== Generation Complete ===")
    print("Tokens generated: {}", token_count)
}
