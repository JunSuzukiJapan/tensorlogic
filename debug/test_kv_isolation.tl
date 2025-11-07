// Test if KV cache properly isolates layers

fn apply_rope_k(K: float32[?, ?], seq_len: float, pos: float) -> float32[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result = reshape(K_r, [seq_len, 256.0])
}

main {
    // Create simple tensors to test isolation
    let K0_raw = f32::ones([34, 256])
    let K0 = apply_rope_k(K0_raw, 34.0, 0.0)

    let K1_raw = f32::zeros([34, 256])
    let K1 = apply_rope_k(K1_raw, 34.0, 0.0)

    // Create KV cache
    let kv_cache = KVCache::new_f32(2)

    // Create dummy V tensors
    let V0 = f32::ones([34, 256])
    let V1 = f32::zeros([34, 256])

    // Store in cache
    kv_cache.set(0, K0, V0)
    kv_cache.set(1, K1, V1)

    // Retrieve from cache
    let retrieved_K0 = kv_cache.get_k(0)
    let retrieved_K1 = kv_cache.get_k(1)

    // Print to check if they're different
    print("K0 shape: {}", shape(retrieved_K0))
    print("K1 shape: {}", shape(retrieved_K1))
    print("K0 and K1 should be different!")
}
