// Test for-loop with heavy GPU memory usage (22 layers + KVCache)

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

test test1_few_layers_then_loop {
    print("=== Test 1: 3 layers + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

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

    print("After 3 layers: {}", shape(h))

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

test test2_many_kv_caches_then_loop {
    print("=== Test 2: 10 KV caches + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    // Create 10 KV caches
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

    print("Created 10 KV caches")

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

test test3_kvcache_object_then_loop {
    print("=== Test 3: KVCache object + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    let L0 = model.blk[0]
    let K0 = apply_rope_k(linear(x, L0.attn_k.weight), seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)

    print("Creating KVCache object...")
    let kv_cache = KVCache::new_f32(3)
    kv_cache.set(0, K0, V0)
    kv_cache.set(1, K0, V0)
    kv_cache.set(2, K0, V0)
    print("✓ KVCache initialized")

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

test test4_large_kvcache_then_loop {
    print("=== Test 4: KVCache(22) + for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    let L0 = model.blk[0]
    let K0 = apply_rope_k(linear(x, L0.attn_k.weight), seq_len, 0.0)
    let V0 = linear(x, L0.attn_v.weight)

    print("Creating KVCache(22)...")
    let kv_cache = KVCache::new_f32(22)
    print("✓ KVCache allocated")

    print("Setting KV caches...")
    kv_cache.set(0, K0, V0)
    kv_cache.set(1, K0, V0)
    kv_cache.set(2, K0, V0)
    kv_cache.set(3, K0, V0)
    kv_cache.set(4, K0, V0)
    kv_cache.set(5, K0, V0)
    kv_cache.set(6, K0, V0)
    kv_cache.set(7, K0, V0)
    kv_cache.set(8, K0, V0)
    kv_cache.set(9, K0, V0)
    kv_cache.set(10, K0, V0)
    kv_cache.set(11, K0, V0)
    kv_cache.set(12, K0, V0)
    kv_cache.set(13, K0, V0)
    kv_cache.set(14, K0, V0)
    kv_cache.set(15, K0, V0)
    kv_cache.set(16, K0, V0)
    kv_cache.set(17, K0, V0)
    kv_cache.set(18, K0, V0)
    kv_cache.set(19, K0, V0)
    kv_cache.set(20, K0, V0)
    kv_cache.set(21, K0, V0)
    print("✓ All 22 caches set")

    print("Starting for-loop...")
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
