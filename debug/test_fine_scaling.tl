// Fine-grained layer scaling test to find exact breaking point (2, 3, 4 layers)

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

test fine_scaling {
    print("=== Fine-Grained Layer Scaling Test (35 tokens) ===")
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

    // Load layers
    let L0 = model.blk[0]
    let L1 = model.blk[1]
    let L2 = model.blk[2]
    let L3 = model.blk[3]

    // ====================================================================
    // Test 1: 2 layers
    // ====================================================================
    print("TEST 1: 2 layers")
    let K0_raw = linear(x, L0.attn_k.weight)
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)

    let K1_raw = linear(x, L1.attn_k.weight)
    let K1 = apply_rope_k(K1_raw, seq_len, 0.0)
    let V1 = linear(x, L1.attn_v.weight)

    let h2 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h2 = transformer_layer(h2, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)

    let norm2 = rms_norm(h2, output_norm)
    let last2 = slice_last(norm2, 0)
    let last2_2d = reshape(last2, [1.0, 2048.0])
    let logits2 = linear(last2_2d, tok_embd)
    let tok2 = temperature_sample(logits2, 0.0)
    let text2 = detokenize_single(tokenizer, tok2, false)

    print("  Output: '{}' (id={})", text2, tok2)
    if tok2 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    // ====================================================================
    // Test 2: 3 layers
    // ====================================================================
    print("TEST 2: 3 layers")
    let K2_raw = linear(x, L2.attn_k.weight)
    let K2 = apply_rope_k(K2_raw, seq_len, 0.0)
    let V2 = linear(x, L2.attn_v.weight)

    let h3 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h3 = transformer_layer(h3, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h3 = transformer_layer(h3, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)

    let norm3 = rms_norm(h3, output_norm)
    let last3 = slice_last(norm3, 0)
    let last3_2d = reshape(last3, [1.0, 2048.0])
    let logits3 = linear(last3_2d, tok_embd)
    let tok3 = temperature_sample(logits3, 0.0)
    let text3 = detokenize_single(tokenizer, tok3, false)

    print("  Output: '{}' (id={})", text3, tok3)
    if tok3 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    // ====================================================================
    // Test 3: 4 layers
    // ====================================================================
    print("TEST 3: 4 layers")
    let K3_raw = linear(x, L3.attn_k.weight)
    let K3 = apply_rope_k(K3_raw, seq_len, 0.0)
    let V3 = linear(x, L3.attn_v.weight)

    let h4 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h4 = transformer_layer(h4, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h4 = transformer_layer(h4, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)
    let h4 = transformer_layer(h4, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3, 0.0, seq_len)

    let norm4 = rms_norm(h4, output_norm)
    let last4 = slice_last(norm4, 0)
    let last4_2d = reshape(last4, [1.0, 2048.0])
    let logits4 = linear(last4_2d, tok_embd)
    let tok4 = temperature_sample(logits4, 0.0)
    let text4 = detokenize_single(tokenizer, tok4, false)

    print("  Output: '{}' (id={})", text4, tok4)
    if tok4 == 0 {
        print("  ❌ ZERO LOGITS")
    } else {
        print("  ✓ OK")
    }
    print("")

    print("=== Summary ===")
    print("2 layers: '{}' (id={})", text2, tok2)
    print("3 layers: '{}' (id={})", text3, tok3)
    print("4 layers: '{}' (id={})", text4, tok4)
}
