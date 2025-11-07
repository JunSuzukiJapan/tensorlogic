// Test K cache calculation to identify buffer sharing issue

fn apply_rope_k(K: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result = reshape(K_r, [seq_len, 256.0])
}

main {
    print("=== Testing K cache calculation ===")

    // Load model
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]
    let L1 = model.blk[1]

    // Create input embedding (fixed prompt)
    let chat_prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)
    let x = embedding(tok_embd, tokens)
    let seq_len = 34.0

    print("Step 1: Computing K0_raw...")
    let K0_raw = linear(x, L0.attn_k.weight)
    print("K0_raw shape: {}", shape(K0_raw))

    print("Step 2: Applying RoPE to K0_raw...")
    let K0 = apply_rope_k(K0_raw, seq_len, 0.0)
    print("K0 shape: {}", shape(K0))

    print("Step 3: Computing K1_raw...")
    let K1_raw = linear(x, L1.attn_k.weight)
    print("K1_raw shape: {}", shape(K1_raw))

    print("Step 4: Applying RoPE to K1_raw...")
    let K1 = apply_rope_k(K1_raw, seq_len, 0.0)
    print("K1 shape: {}", shape(K1))

    print("\n=== Creating KV cache ===")
    let kv_cache = KVCache::new_f32(2)
    let V0 = linear(x, L0.attn_v.weight)
    let V1 = linear(x, L1.attn_v.weight)

    kv_cache.set(0, K0, V0)
    kv_cache.set(1, K1, V1)

    print("\n=== Retrieving from cache ===")
    let retrieved_K0 = kv_cache.get_k(0)
    let retrieved_K1 = kv_cache.get_k(1)

    print("Retrieved K0 shape: {}", shape(retrieved_K0))
    print("Retrieved K1 shape: {}", shape(retrieved_K1))

    print("\n✓ Test complete")
}
