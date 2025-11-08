// Test transformer with increasing number of layers to find where it breaks
// Run with: ./target/release/tl run debug/test_layer_scaling.tl --test

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    result = x * sigmoid(x)
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
    result = linear(mul_result, W_down)
}

fn apply_rope_k(K: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result = reshape(K_r, [seq_len, 256.0])
}

fn apply_rope_q(Q: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let Q_h = reshape(Q, [seq_len, 32.0, 64.0])
    let Q_r = rope(Q_h, pos)
    result = reshape(Q_r, [seq_len, 2048.0])
}

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
    let normed = rms_norm(x, W_attn_norm)
    let Q_raw = linear(normed, W_q)
    let Q = apply_rope_q(Q_raw, seq_len, position)

    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out

    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result = after_attn + ffn_out
}

test layer_scaling {
    print("=== Layer Scaling Test (35 tokens) ===")
    print("")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight

    // Chat template (35 tokens)
    let chat_prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    print("Input: {} tokens", seq_len)
    print("")

    // Load all layers
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

    // ====================================================================
    // Test 1: 1 layer
    // ====================================================================
    print("TEST 1: 1 layer")
    let K0_raw = linear(x, L0.attn_k.weight)
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)

    let h1 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)

    let norm1 = rms_norm(h1, output_norm)
    let last1 = slice_last(norm1, 0)
    let last1_2d = reshape(last1, [1.0, 2048.0])
    let logits1 = linear(last1_2d, tok_embd)
    let tok1 = temperature_sample(logits1, 0.0)
    let text1 = detokenize_single(tokenizer, tok1, false)

    print("  Output: '{}' (id={})", text1, tok1)
    if tok1 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    // ====================================================================
    // Test 2: 5 layers
    // ====================================================================
    print("TEST 2: 5 layers")
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

    let h5 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h5 = transformer_layer(h5, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h5 = transformer_layer(h5, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)
    let h5 = transformer_layer(h5, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3, 0.0, seq_len)
    let h5 = transformer_layer(h5, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, K4, V4, 0.0, seq_len)

    let norm5 = rms_norm(h5, output_norm)
    let last5 = slice_last(norm5, 0)
    let last5_2d = reshape(last5, [1.0, 2048.0])
    let logits5 = linear(last5_2d, tok_embd)
    let tok5 = temperature_sample(logits5, 0.0)
    let text5 = detokenize_single(tokenizer, tok5, false)

    print("  Output: '{}' (id={})", text5, tok5)
    if tok5 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    // ====================================================================
    // Test 3: 10 layers
    // ====================================================================
    print("TEST 3: 10 layers")
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

    let h10 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h10 = transformer_layer(h10, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h10 = transformer_layer(h10, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)
    let h10 = transformer_layer(h10, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3, 0.0, seq_len)
    let h10 = transformer_layer(h10, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, K4, V4, 0.0, seq_len)
    let h10 = transformer_layer(h10, L5.attn_norm.weight, L5.attn_q.weight, L5.attn_output.weight, L5.ffn_norm.weight, L5.ffn_gate.weight, L5.ffn_up.weight, L5.ffn_down.weight, K5, V5, 0.0, seq_len)
    let h10 = transformer_layer(h10, L6.attn_norm.weight, L6.attn_q.weight, L6.attn_output.weight, L6.ffn_norm.weight, L6.ffn_gate.weight, L6.ffn_up.weight, L6.ffn_down.weight, K6, V6, 0.0, seq_len)
    let h10 = transformer_layer(h10, L7.attn_norm.weight, L7.attn_q.weight, L7.attn_output.weight, L7.ffn_norm.weight, L7.ffn_gate.weight, L7.ffn_up.weight, L7.ffn_down.weight, K7, V7, 0.0, seq_len)
    let h10 = transformer_layer(h10, L8.attn_norm.weight, L8.attn_q.weight, L8.attn_output.weight, L8.ffn_norm.weight, L8.ffn_gate.weight, L8.ffn_up.weight, L8.ffn_down.weight, K8, V8, 0.0, seq_len)
    let h10 = transformer_layer(h10, L9.attn_norm.weight, L9.attn_q.weight, L9.attn_output.weight, L9.ffn_norm.weight, L9.ffn_gate.weight, L9.ffn_up.weight, L9.ffn_down.weight, K9, V9, 0.0, seq_len)

    let norm10 = rms_norm(h10, output_norm)
    let last10 = slice_last(norm10, 0)
    let last10_2d = reshape(last10, [1.0, 2048.0])
    let logits10 = linear(last10_2d, tok_embd)
    let tok10 = temperature_sample(logits10, 0.0)
    let text10 = detokenize_single(tokenizer, tok10, false)

    print("  Output: '{}' (id={})", text10, tok10)
    if tok10 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    print("=== Summary ===")
    print("1 layer:  '{}' (id={})", text1, tok1)
    print("5 layers: '{}' (id={})", text5, tok5)
    print("10 layers: '{}' (id={})", text10, tok10)
}
