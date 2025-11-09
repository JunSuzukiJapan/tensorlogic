// Test 22 layers prefill WITHOUT KVCache initialization
// To isolate if KVCache::new_f32() causes the zero logits issue

fn silu(x: float32[?, ?]) -> float32[?, ?] {
    x * sigmoid(x)
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
    linear(mul_result, W_down)
}

fn apply_rope_k(K: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    reshape(K_r, [seq_len, 256.0])
}

fn apply_rope_q(Q: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let Q_h = reshape(Q, [seq_len, 32.0, 64.0])
    let Q_r = rope(Q_h, pos)
    reshape(Q_r, [seq_len, 2048.0])
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
    after_attn + ffn_out
}

test no_kvcache {
    print("=== 22-Layer Test WITHOUT KVCache initialization ===")
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

    // Build KV caches (same as before)
    print("Building KV caches...")
    let K0 = apply_rope_k(linear(x, L0.attn_k.weight), seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = apply_rope_k(linear(x, L1.attn_k.weight), seq_len, 0.0)
    let V1 = linear(x, L1.attn_v.weight)
    let K2 = apply_rope_k(linear(x, L2.attn_k.weight), seq_len, 0.0)
    let V2 = linear(x, L2.attn_v.weight)
    let K3 = apply_rope_k(linear(x, L3.attn_k.weight), seq_len, 0.0)
    let V3 = linear(x, L3.attn_v.weight)
    let K4 = apply_rope_k(linear(x, L4.attn_k.weight), seq_len, 0.0)
    let V4 = linear(x, L4.attn_v.weight)
    let K5 = apply_rope_k(linear(x, L5.attn_k.weight), seq_len, 0.0)
    let V5 = linear(x, L5.attn_v.weight)
    let K6 = apply_rope_k(linear(x, L6.attn_k.weight), seq_len, 0.0)
    let V6 = linear(x, L6.attn_v.weight)
    let K7 = apply_rope_k(linear(x, L7.attn_k.weight), seq_len, 0.0)
    let V7 = linear(x, L7.attn_v.weight)
    let K8 = apply_rope_k(linear(x, L8.attn_k.weight), seq_len, 0.0)
    let V8 = linear(x, L8.attn_v.weight)
    let K9 = apply_rope_k(linear(x, L9.attn_k.weight), seq_len, 0.0)
    let V9 = linear(x, L9.attn_v.weight)
    let K10 = apply_rope_k(linear(x, L10.attn_k.weight), seq_len, 0.0)
    let V10 = linear(x, L10.attn_v.weight)
    let K11 = apply_rope_k(linear(x, L11.attn_k.weight), seq_len, 0.0)
    let V11 = linear(x, L11.attn_v.weight)
    let K12 = apply_rope_k(linear(x, L12.attn_k.weight), seq_len, 0.0)
    let V12 = linear(x, L12.attn_v.weight)
    let K13 = apply_rope_k(linear(x, L13.attn_k.weight), seq_len, 0.0)
    let V13 = linear(x, L13.attn_v.weight)
    let K14 = apply_rope_k(linear(x, L14.attn_k.weight), seq_len, 0.0)
    let V14 = linear(x, L14.attn_v.weight)
    let K15 = apply_rope_k(linear(x, L15.attn_k.weight), seq_len, 0.0)
    let V15 = linear(x, L15.attn_v.weight)
    let K16 = apply_rope_k(linear(x, L16.attn_k.weight), seq_len, 0.0)
    let V16 = linear(x, L16.attn_v.weight)
    let K17 = apply_rope_k(linear(x, L17.attn_k.weight), seq_len, 0.0)
    let V17 = linear(x, L17.attn_v.weight)
    let K18 = apply_rope_k(linear(x, L18.attn_k.weight), seq_len, 0.0)
    let V18 = linear(x, L18.attn_v.weight)
    let K19 = apply_rope_k(linear(x, L19.attn_k.weight), seq_len, 0.0)
    let V19 = linear(x, L19.attn_v.weight)
    let K20 = apply_rope_k(linear(x, L20.attn_k.weight), seq_len, 0.0)
    let V20 = linear(x, L20.attn_v.weight)
    let K21 = apply_rope_k(linear(x, L21.attn_k.weight), seq_len, 0.0)
    let V21 = linear(x, L21.attn_v.weight)
    print("✓ KV caches built")
    print("")

    // ⚠️ DIFFERENCE: NO KVCache::new_f32() here!
    // Just pass through transformer layers and generate output
    print("Passing through 22 transformer layers...")
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
    print("✓ All layers complete")
    print("")

    // Final norm and projection
    let norm = rms_norm(h, output_norm)
    let last = slice_last(norm, 0)
    let last_2d = reshape(last, [1.0, 2048.0])
    let logits = linear(last_2d, tok_embd)

    let tok = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, tok, false)

    print("Output: '{}' (id={})", text, tok)
    if tok == 0 {
        print("❌ ZERO LOGITS")
    } else {
        print("✓ SUCCESS")
    }
}
