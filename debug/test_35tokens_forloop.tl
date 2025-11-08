// Test with 35 tokens (chat template) + 22 layers + for-loop
// Exact reproduction of test_with_empty_loop conditions

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

test test1_35tokens_3layers {
    print("=== Test 1: 35 tokens + 3 layers + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let chat_prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    print("Sequence length: {}", seq_len)

    let L0 = model.blk[0]
    let L1 = model.blk[1]
    let L2 = model.blk[2]

    let K0 = apply_rope_k(linear(x, L0.attn_k.weight), seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = apply_rope_k(linear(x, L1.attn_k.weight), seq_len, 0.0)
    let V1 = linear(x, L1.attn_v.weight)
    let K2 = apply_rope_k(linear(x, L2.attn_k.weight), seq_len, 0.0)
    let V2 = linear(x, L2.attn_v.weight)

    let h = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0, 0.0, seq_len)
    let h = transformer_layer(h, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1, 0.0, seq_len)
    let h = transformer_layer(h, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2, 0.0, seq_len)

    print("After 3 layers")

    let count = 0
    for i in range(3) {
        count = count + 1
    }

    print("Loop count: {}", count)
    if count == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test2_35tokens_10layers {
    print("=== Test 2: 35 tokens + 10 layers + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let chat_prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    print("Sequence length: {}", seq_len)

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

    print("After 10 layers")

    let count = 0
    for i in range(3) {
        count = count + 1
    }

    print("Loop count: {}", count)
    if count == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}
