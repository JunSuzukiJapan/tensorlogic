// 3-layer test with debug output to find where NaN occurs

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

test three_layer_debug {
    print("=== 3-Layer Debug Test (35 tokens) ===")
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

    print("1. Input embedding shape: {}", shape(x))
    print("")

    // Load layers
    let L0 = model.blk[0]
    let L1 = model.blk[1]
    let L2 = model.blk[2]

    // Prepare KV caches
    let K0_raw = linear(x, L0.attn_k.weight)
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)
    print("2. Layer 0 K/V cache shapes: {} / {}", shape(K0), shape(V0))

    let K1_raw = linear(x, L1.attn_k.weight)
    let K1 = apply_rope_k(K1_raw, seq_len, 0.0)
    let V1 = linear(x, L1.attn_v.weight)
    print("   Layer 1 K/V cache shapes: {} / {}", shape(K1), shape(V1))

    let K2_raw = linear(x, L2.attn_k.weight)
    let K2 = apply_rope_k(K2_raw, seq_len, 0.0)
    let V2 = linear(x, L2.attn_v.weight)
    print("   Layer 2 K/V cache shapes: {} / {}", shape(K2), shape(V2))
    print("")

    // Layer 0
    print("3. Passing through layer 0...")
    let h = transformer_layer(
        x,
        L0.attn_norm.weight,
        L0.attn_q.weight,
        L0.attn_output.weight,
        L0.ffn_norm.weight,
        L0.ffn_gate.weight,
        L0.ffn_up.weight,
        L0.ffn_down.weight,
        K0,
        V0,
        0.0,
        seq_len
    )
    print("   After layer 0 shape: {}", shape(h))
    print("")

    // Layer 1
    print("4. Passing through layer 1...")
    let h = transformer_layer(
        h,
        L1.attn_norm.weight,
        L1.attn_q.weight,
        L1.attn_output.weight,
        L1.ffn_norm.weight,
        L1.ffn_gate.weight,
        L1.ffn_up.weight,
        L1.ffn_down.weight,
        K1,
        V1,
        0.0,
        seq_len
    )
    print("   After layer 1 shape: {}", shape(h))
    print("")

    // Layer 2
    print("5. Passing through layer 2...")
    let h = transformer_layer(
        h,
        L2.attn_norm.weight,
        L2.attn_q.weight,
        L2.attn_output.weight,
        L2.ffn_norm.weight,
        L2.ffn_gate.weight,
        L2.ffn_up.weight,
        L2.ffn_down.weight,
        K2,
        V2,
        0.0,
        seq_len
    )
    print("   After layer 2 shape: {}", shape(h))
    print("")

    // Final norm and projection
    print("6. Final normalization and projection...")
    let norm = rms_norm(h, output_norm)
    print("   After final norm shape: {}", shape(norm))

    let last = slice_last(norm, 0)
    print("   After slice_last shape: {}", shape(last))

    let last_2d = reshape(last, [1.0, 2048.0])
    print("   After reshape shape: {}", shape(last_2d))

    let logits = linear(last_2d, tok_embd)
    print("   Logits shape: {}", shape(logits))
    print("")

    let tok = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, tok, false)

    print("Output: '{}' (id={})", text, tok)
    if tok == 0 {
        print("❌ ZERO LOGITS")
    } else {
        print("✓ SUCCESS")
    }
}
