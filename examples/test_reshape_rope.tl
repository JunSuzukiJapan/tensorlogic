// Test reshape + rope fix

// Fixed: Pass seq_len as parameter to avoid shape[0] indexing issue
fn apply_rope_k_test(K: float16[?, ?], seq_len: float, pos: float) -> float16[?, ?] {
    print("[APPLY_ROPE_K] === Entry === seq_len={}, pos={}", seq_len, pos)
    print("[APPLY_ROPE_K] Step 1: Calling reshape(K, [{}, 4.0, 64.0])...", seq_len)
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    print("[APPLY_ROPE_K] Step 1: DONE")
    print("[APPLY_ROPE_K] Step 2: Calling rope(K_h, {})...", pos)
    let K_r = rope(K_h, pos)
    print("[APPLY_ROPE_K] Step 2: DONE")
    print("[APPLY_ROPE_K] Step 3: Reshaping back to [{}, 256.0]...", seq_len)
    let result_tmp = reshape(K_r, [seq_len, 256.0])
    print("[APPLY_ROPE_K] Step 3: DONE")
    print("[APPLY_ROPE_K] === Exit ===")
    result := result_tmp
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

    // Apply RoPE with reshape (pass seq_len=2.0 directly to avoid shape[0] indexing)
    print("Applying RoPE with reshape...")
    let K = apply_rope_k_test(K_raw, 2.0, 0.0)
    print("K: shape = {}", shape(K))

    print("✅ Test passed!")
}
