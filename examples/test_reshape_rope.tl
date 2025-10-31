// Test reshape + rope fix

fn apply_rope_k_test(K: float16[?, ?], pos: float) -> float16[?, ?] {
    let shp = shape(K)
    let s = shp[0]
    print("Before reshape: shape = {}", shp)
    let K_h = reshape(K, [s, 4.0, 64.0])
    print("After reshape: calling rope...")
    let K_r = rope(K_h, pos)
    print("After rope: reshaping back...")
    result := reshape(K_r, [s, 256.0])
}

main {
    print("=== Testing reshape + rope fix ===")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")
    let L0 = model.blk[0]
    let tok_embd = model.token_embd.weight

    // Create small embedding
    let prompt = "Hello"
    let tokens = tokenizer.tokenize(prompt, false)
    let x = embedding(tok_embd, tokens)  // Shape: [?, 2048]
    print("Embeddings: shape = {}", shape(x))

    // Apply linear to get K
    let K_raw = linear(x, L0.attn_k.weight)  // Shape: [4, 256]
    print("K_raw: shape = {}", shape(K_raw))

    // Apply RoPE with reshape
    print("Applying RoPE with reshape...")
    let K = apply_rope_k_test(K_raw, 0.0)
    print("K: shape = {}", shape(K))

    print("✅ Test passed!")
}
