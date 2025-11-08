// Test with just ONE transformer layer

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

main {
    print("=== Single Layer Test ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let L0 = model.blk[0]

    let tokens = tokenizer.tokenize("Hello!", true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    print("1. Embedding shape: {}", shape(x))

    // Build K, V cache for layer 0
    let K0_raw = linear(x, L0.attn_k.weight)
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)

    // Pass through ONE transformer layer
    let h = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)

    print("2. After transformer layer 0: {}", shape(h))

    // Output
    let final_norm = rms_norm(h, output_norm)
    let last_token = slice_last(final_norm, 0)
    let last_token_2d = reshape(last_token, [1.0, 2048.0])
    let logits = linear(last_token_2d, tok_embd)

    print("3. Logits shape: {}", shape(logits))

    let token_id = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, token_id, false)

    print("")
    print("Generated: '{}' (id={})", text, token_id)
    print("")

    if token_id == 0 {
        print("❌ FAILED")
    } else {
        print("✓ SUCCESS")
    }
}
