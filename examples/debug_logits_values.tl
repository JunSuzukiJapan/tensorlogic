// Debug actual logits values to compare with llama.cpp

// Helper: Apply RoPE to K for caching (optimized: no shape() call)
fn apply_rope_k(K: float16[?, ?], seq_len: float, pos: float) -> float16[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result = reshape(K_r, [seq_len, 256.0])
}

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    result = linear(mul_result, W_down)
}

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
    print("=== Debug Logits Values ===")
    print("")

    print("Loading model...")
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    // Load ALL 22 layers
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

    print("Tokenizing...")
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + "Hello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)

    print("Running full 22-layer inference...")
    let x = embedding(tok_embd, tokens)

    // Get sequence length for RoPE
    let x_shp = shape(x)
    let seq_len = x_shp[0]

    // Build KV caches (with RoPE applied to K)
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

    // Run through all 22 layers
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

    let final_norm = rms_norm(h21, output_norm)
    let logits = linear(final_norm, output)

    print("Logits shape:")
    let logits_shape = shape(logits)
    print("  [{}, {}]", logits_shape[0], logits_shape[1])

    print("")
    print("Sampling with temperature=0.8...")
    let token_id = temperature_sample(logits, 0.8)
    let text = detokenize_single(tokenizer, token_id, false)

    print("  Token ID: {}", token_id)
    print("  Text: '{}'", text)

    print("")
    print("=== Test Complete ===")
}
