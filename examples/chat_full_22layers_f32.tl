// Full 22-Layer Chat Demo with f32
// Complete implementation using ALL 22 transformer layers

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    result := x * sigmoid(x)
}

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
    result := linear(mul_result, W_down)
}

fn attention_with_cache(
    Q: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?],
    W_o: float32[?, ?]
) -> float32[?, ?] {
    let Q_shape = shape(Q)
    let seq_len_f = Q_shape[0]
    let K_shape = shape(K_cache)
    let cache_len_f = K_shape[0]

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let Q_rope = rope(Q_heads)

    let K_heads = reshape(K_cache, [cache_len_f, 4.0, 64.0])
    let K_rope = rope(K_heads)
    let V_heads = reshape(V_cache, [cache_len_f, 4.0, 64.0])

    let K_exp = reshape(K_rope, [cache_len_f, 4.0, 1.0, 64.0])
    let K_broadcast = broadcast_to(K_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let K_expanded = reshape(K_broadcast, [cache_len_f, 32.0, 64.0])

    let V_exp = reshape(V_heads, [cache_len_f, 4.0, 1.0, 64.0])
    let V_broadcast = broadcast_to(V_exp, [cache_len_f, 4.0, 8.0, 64.0])
    let V_expanded = reshape(V_broadcast, [cache_len_f, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled = scores * 0.125
    let attn = softmax(scaled, 2)
    let out = einsum("ihj,jhd->ihd", attn, V_expanded)

    let reshaped = reshape(out, [seq_len_f, 2048.0])
    result := linear(reshaped, W_o)
}

fn transformer_layer(
    x: float32[?, ?],
    W_attn_norm: float32[?],
    W_q: float32[?, ?],
    W_k: float32[?, ?],
    W_v: float32[?, ?],
    W_o: float32[?, ?],
    W_ffn_norm: float32[?],
    W_gate: float32[?, ?],
    W_up: float32[?, ?],
    W_down: float32[?, ?],
    K_cache: float32[?, ?],
    V_cache: float32[?, ?]
) -> float32[?, ?] {
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
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
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
    // ============================================================
    let K0 = linear(x, L0.attn_k.weight)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = linear(x, L1.attn_k.weight)
    let V1 = linear(x, L1.attn_v.weight)
    let K2 = linear(x, L2.attn_k.weight)
    let V2 = linear(x, L2.attn_v.weight)
    let K3 = linear(x, L3.attn_k.weight)
    let V3 = linear(x, L3.attn_v.weight)
    let K4 = linear(x, L4.attn_k.weight)
    let V4 = linear(x, L4.attn_v.weight)
    let K5 = linear(x, L5.attn_k.weight)
    let V5 = linear(x, L5.attn_v.weight)
    let K6 = linear(x, L6.attn_k.weight)
    let V6 = linear(x, L6.attn_v.weight)
    let K7 = linear(x, L7.attn_k.weight)
    let V7 = linear(x, L7.attn_v.weight)
    let K8 = linear(x, L8.attn_k.weight)
    let V8 = linear(x, L8.attn_v.weight)
    let K9 = linear(x, L9.attn_k.weight)
    let V9 = linear(x, L9.attn_v.weight)
    let K10 = linear(x, L10.attn_k.weight)
    let V10 = linear(x, L10.attn_v.weight)
    let K11 = linear(x, L11.attn_k.weight)
    let V11 = linear(x, L11.attn_v.weight)
    let K12 = linear(x, L12.attn_k.weight)
    let V12 = linear(x, L12.attn_v.weight)
    let K13 = linear(x, L13.attn_k.weight)
    let V13 = linear(x, L13.attn_v.weight)
    let K14 = linear(x, L14.attn_k.weight)
    let V14 = linear(x, L14.attn_v.weight)
    let K15 = linear(x, L15.attn_k.weight)
    let V15 = linear(x, L15.attn_v.weight)
    let K16 = linear(x, L16.attn_k.weight)
    let V16 = linear(x, L16.attn_v.weight)
    let K17 = linear(x, L17.attn_k.weight)
    let V17 = linear(x, L17.attn_v.weight)
    let K18 = linear(x, L18.attn_k.weight)
    let V18 = linear(x, L18.attn_v.weight)
    let K19 = linear(x, L19.attn_k.weight)
    let V19 = linear(x, L19.attn_v.weight)
    let K20 = linear(x, L20.attn_k.weight)
    let V20 = linear(x, L20.attn_v.weight)
    let K21 = linear(x, L21.attn_k.weight)
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

    // Token-by-token generation (max 50 tokens)
    for i in range(50) {
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
            // ========================================================
            let token_ids_single = int_to_tokenids(token_id)
            let new_token_emb = embedding(tok_embd, token_ids_single)
            
            let nK0 = linear(new_token_emb, L0.attn_k.weight)
            let nV0 = linear(new_token_emb, L0.attn_v.weight)
            KV0 = concat(KV0, nK0, 0.0)
            KV0_V = concat(KV0_V, nV0, 0.0)
            
            let nh0 = transformer_layer(new_token_emb, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, KV0, KV0_V)
            
            let nK1 = linear(nh0, L1.attn_k.weight)
            let nV1 = linear(nh0, L1.attn_v.weight)
            KV1 = concat(KV1, nK1, 0.0)
            KV1_V = concat(KV1_V, nV1, 0.0)
            
            let nh1 = transformer_layer(nh0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, KV1, KV1_V)
            
            let nK2 = linear(nh1, L2.attn_k.weight)
            let nV2 = linear(nh1, L2.attn_v.weight)
            KV2 = concat(KV2, nK2, 0.0)
            KV2_V = concat(KV2_V, nV2, 0.0)
            
            let nh2 = transformer_layer(nh1, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_k.weight, L2.attn_v.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, KV2, KV2_V)
            
            let nK3 = linear(nh2, L3.attn_k.weight)
            let nV3 = linear(nh2, L3.attn_v.weight)
            KV3 = concat(KV3, nK3, 0.0)
            KV3_V = concat(KV3_V, nV3, 0.0)
            
            let nh3 = transformer_layer(nh2, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_k.weight, L3.attn_v.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, KV3, KV3_V)
            
            let nK4 = linear(nh3, L4.attn_k.weight)
            let nV4 = linear(nh3, L4.attn_v.weight)
            KV4 = concat(KV4, nK4, 0.0)
            KV4_V = concat(KV4_V, nV4, 0.0)
            
            let nh4 = transformer_layer(nh3, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_k.weight, L4.attn_v.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, KV4, KV4_V)
            
            let nK5 = linear(nh4, L5.attn_k.weight)
            let nV5 = linear(nh4, L5.attn_v.weight)
            KV5 = concat(KV5, nK5, 0.0)
            KV5_V = concat(KV5_V, nV5, 0.0)
            
            let nh5 = transformer_layer(nh4, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_k.weight, L5.attn_v.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, KV5, KV5_V)
            
            let nK6 = linear(nh5, L6.attn_k.weight)
            let nV6 = linear(nh5, L6.attn_v.weight)
            KV6 = concat(KV6, nK6, 0.0)
            KV6_V = concat(KV6_V, nV6, 0.0)
            
            let nh6 = transformer_layer(nh5, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_k.weight, L6.attn_v.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, KV6, KV6_V)
            
            let nK7 = linear(nh6, L7.attn_k.weight)
            let nV7 = linear(nh6, L7.attn_v.weight)
            KV7 = concat(KV7, nK7, 0.0)
            KV7_V = concat(KV7_V, nV7, 0.0)
            
            let nh7 = transformer_layer(nh6, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_k.weight, L7.attn_v.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, KV7, KV7_V)
            
            let nK8 = linear(nh7, L8.attn_k.weight)
            let nV8 = linear(nh7, L8.attn_v.weight)
            KV8 = concat(KV8, nK8, 0.0)
            KV8_V = concat(KV8_V, nV8, 0.0)
            
            let nh8 = transformer_layer(nh7, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_k.weight, L8.attn_v.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, KV8, KV8_V)
            
            let nK9 = linear(nh8, L9.attn_k.weight)
            let nV9 = linear(nh8, L9.attn_v.weight)
            KV9 = concat(KV9, nK9, 0.0)
            KV9_V = concat(KV9_V, nV9, 0.0)
            
            let nh9 = transformer_layer(nh8, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_k.weight, L9.attn_v.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, KV9, KV9_V)
            
            let nK10 = linear(nh9, L10.attn_k.weight)
            let nV10 = linear(nh9, L10.attn_v.weight)
            KV10 = concat(KV10, nK10, 0.0)
            KV10_V = concat(KV10_V, nV10, 0.0)
            
            let nh10 = transformer_layer(nh9, L10.attn_norm.weight, L10.attn_q.weight, L10.attn_k.weight, L10.attn_v.weight, L10.attn_output.weight, L10.ffn_norm.weight, L10.ffn_gate.weight, L10.ffn_up.weight, L10.ffn_down.weight, KV10, KV10_V)
            
            let nK11 = linear(nh10, L11.attn_k.weight)
            let nV11 = linear(nh10, L11.attn_v.weight)
            KV11 = concat(KV11, nK11, 0.0)
            KV11_V = concat(KV11_V, nV11, 0.0)
            
            let nh11 = transformer_layer(nh10, L11.attn_norm.weight, L11.attn_q.weight, L11.attn_k.weight, L11.attn_v.weight, L11.attn_output.weight, L11.ffn_norm.weight, L11.ffn_gate.weight, L11.ffn_up.weight, L11.ffn_down.weight, KV11, KV11_V)
            
            let nK12 = linear(nh11, L12.attn_k.weight)
            let nV12 = linear(nh11, L12.attn_v.weight)
            KV12 = concat(KV12, nK12, 0.0)
            KV12_V = concat(KV12_V, nV12, 0.0)
            
            let nh12 = transformer_layer(nh11, L12.attn_norm.weight, L12.attn_q.weight, L12.attn_k.weight, L12.attn_v.weight, L12.attn_output.weight, L12.ffn_norm.weight, L12.ffn_gate.weight, L12.ffn_up.weight, L12.ffn_down.weight, KV12, KV12_V)
            
            let nK13 = linear(nh12, L13.attn_k.weight)
            let nV13 = linear(nh12, L13.attn_v.weight)
            KV13 = concat(KV13, nK13, 0.0)
            KV13_V = concat(KV13_V, nV13, 0.0)
            
            let nh13 = transformer_layer(nh12, L13.attn_norm.weight, L13.attn_q.weight, L13.attn_k.weight, L13.attn_v.weight, L13.attn_output.weight, L13.ffn_norm.weight, L13.ffn_gate.weight, L13.ffn_up.weight, L13.ffn_down.weight, KV13, KV13_V)
            
            let nK14 = linear(nh13, L14.attn_k.weight)
            let nV14 = linear(nh13, L14.attn_v.weight)
            KV14 = concat(KV14, nK14, 0.0)
            KV14_V = concat(KV14_V, nV14, 0.0)
            
            let nh14 = transformer_layer(nh13, L14.attn_norm.weight, L14.attn_q.weight, L14.attn_k.weight, L14.attn_v.weight, L14.attn_output.weight, L14.ffn_norm.weight, L14.ffn_gate.weight, L14.ffn_up.weight, L14.ffn_down.weight, KV14, KV14_V)
            
            let nK15 = linear(nh14, L15.attn_k.weight)
            let nV15 = linear(nh14, L15.attn_v.weight)
            KV15 = concat(KV15, nK15, 0.0)
            KV15_V = concat(KV15_V, nV15, 0.0)
            
            let nh15 = transformer_layer(nh14, L15.attn_norm.weight, L15.attn_q.weight, L15.attn_k.weight, L15.attn_v.weight, L15.attn_output.weight, L15.ffn_norm.weight, L15.ffn_gate.weight, L15.ffn_up.weight, L15.ffn_down.weight, KV15, KV15_V)
            
            let nK16 = linear(nh15, L16.attn_k.weight)
            let nV16 = linear(nh15, L16.attn_v.weight)
            KV16 = concat(KV16, nK16, 0.0)
            KV16_V = concat(KV16_V, nV16, 0.0)
            
            let nh16 = transformer_layer(nh15, L16.attn_norm.weight, L16.attn_q.weight, L16.attn_k.weight, L16.attn_v.weight, L16.attn_output.weight, L16.ffn_norm.weight, L16.ffn_gate.weight, L16.ffn_up.weight, L16.ffn_down.weight, KV16, KV16_V)
            
            let nK17 = linear(nh16, L17.attn_k.weight)
            let nV17 = linear(nh16, L17.attn_v.weight)
            KV17 = concat(KV17, nK17, 0.0)
            KV17_V = concat(KV17_V, nV17, 0.0)
            
            let nh17 = transformer_layer(nh16, L17.attn_norm.weight, L17.attn_q.weight, L17.attn_k.weight, L17.attn_v.weight, L17.attn_output.weight, L17.ffn_norm.weight, L17.ffn_gate.weight, L17.ffn_up.weight, L17.ffn_down.weight, KV17, KV17_V)
            
            let nK18 = linear(nh17, L18.attn_k.weight)
            let nV18 = linear(nh17, L18.attn_v.weight)
            KV18 = concat(KV18, nK18, 0.0)
            KV18_V = concat(KV18_V, nV18, 0.0)
            
            let nh18 = transformer_layer(nh17, L18.attn_norm.weight, L18.attn_q.weight, L18.attn_k.weight, L18.attn_v.weight, L18.attn_output.weight, L18.ffn_norm.weight, L18.ffn_gate.weight, L18.ffn_up.weight, L18.ffn_down.weight, KV18, KV18_V)
            
            let nK19 = linear(nh18, L19.attn_k.weight)
            let nV19 = linear(nh18, L19.attn_v.weight)
            KV19 = concat(KV19, nK19, 0.0)
            KV19_V = concat(KV19_V, nV19, 0.0)
            
            let nh19 = transformer_layer(nh18, L19.attn_norm.weight, L19.attn_q.weight, L19.attn_k.weight, L19.attn_v.weight, L19.attn_output.weight, L19.ffn_norm.weight, L19.ffn_gate.weight, L19.ffn_up.weight, L19.ffn_down.weight, KV19, KV19_V)
            
            let nK20 = linear(nh19, L20.attn_k.weight)
            let nV20 = linear(nh19, L20.attn_v.weight)
            KV20 = concat(KV20, nK20, 0.0)
            KV20_V = concat(KV20_V, nV20, 0.0)
            
            let nh20 = transformer_layer(nh19, L20.attn_norm.weight, L20.attn_q.weight, L20.attn_k.weight, L20.attn_v.weight, L20.attn_output.weight, L20.ffn_norm.weight, L20.ffn_gate.weight, L20.ffn_up.weight, L20.ffn_down.weight, KV20, KV20_V)
            
            let nK21 = linear(nh20, L21.attn_k.weight)
            let nV21 = linear(nh20, L21.attn_v.weight)
            KV21 = concat(KV21, nK21, 0.0)
            KV21_V = concat(KV21_V, nV21, 0.0)
            
            let nh21 = transformer_layer(nh20, L21.attn_norm.weight, L21.attn_q.weight, L21.attn_k.weight, L21.attn_v.weight, L21.attn_output.weight, L21.ffn_norm.weight, L21.ffn_gate.weight, L21.ffn_up.weight, L21.ffn_down.weight, KV21, KV21_V)

            let norm_new = rms_norm(nh21, output_norm)
            current_logits = linear(norm_new, output)
        }
    }

    print("")
    print("")
    print("=== Generation Complete ===")
    print("Tokens generated: {}", token_count)
}
