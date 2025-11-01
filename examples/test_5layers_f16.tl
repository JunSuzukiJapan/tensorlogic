// 5-Layer Test to isolate reshape hang issue

fn apply_rope_k(K: float16[?, ?], pos: float) -> float16[?, ?] {
    let shp = shape(K)
    let s = shp[0]
    print("    [ROPE] Starting reshape...")
    let K_h = reshape(K, [s, 4.0, 64.0])
    print("    [ROPE] Starting rope...")
    let K_r = rope(K_h, pos)
    print("    [ROPE] Reshaping back...")
    result = reshape(K_r, [s, 256.0])
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = gate * sigmoid(gate)
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
    print("=== 5-Layer Test: Isolate reshape hang ===")
    print("")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("[1] Loading 5 layers...")
    let L0 = model.blk[0]
    let L1 = model.blk[1]
    let L2 = model.blk[2]
    let L3 = model.blk[3]
    let L4 = model.blk[4]
    print("    ✓ 5 layers loaded")
    print("")

    print("[2] Tokenizing prompt...")
    let prompt = "Hello"
    let tokens = tokenizer.tokenize(prompt, false)
    print("    Tokens ready")
    print("")

    print("[3] Embedding...")
    let x = embedding(tok_embd, tokens)
    print("    Embeddings: shape = {}", shape(x))
    print("")

    print("[4] Building KV caches (with reshape)...")

    print("  Layer 0...")
    let K0_raw = linear(x, L0.attn_k.weight)
    print("    K0_raw: shape = {}", shape(K0_raw))
    let K0 = apply_rope_k(K0_raw, 0.0)
    print("    K0: shape = {}", shape(K0))
    let V0 = linear(x, L0.attn_v.weight)

    print("  Layer 1...")
    let K1_raw = linear(x, L1.attn_k.weight)
    let K1 = apply_rope_k(K1_raw, 0.0)
    let V1 = linear(x, L1.attn_v.weight)

    print("  Layer 2...")
    let K2_raw = linear(x, L2.attn_k.weight)
    let K2 = apply_rope_k(K2_raw, 0.0)
    let V2 = linear(x, L2.attn_v.weight)

    print("  Layer 3...")
    let K3_raw = linear(x, L3.attn_k.weight)
    let K3 = apply_rope_k(K3_raw, 0.0)
    let V3 = linear(x, L3.attn_v.weight)

    print("  Layer 4...")
    let K4_raw = linear(x, L4.attn_k.weight)
    let K4 = apply_rope_k(K4_raw, 0.0)
    let V4 = linear(x, L4.attn_v.weight)

    print("    ✓ All KV caches built")
    print("")

    print("[5] Running prefill through 5 layers...")
    let h0 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)
    print("    Layer 0 complete")
    let h1 = transformer_layer(h0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)
    print("    Layer 1 complete")
    let h2 = transformer_layer(h1, L2.attn_norm.weight, L2.attn_q.weight, L2.attn_k.weight, L2.attn_v.weight, L2.attn_output.weight, L2.ffn_norm.weight, L2.ffn_gate.weight, L2.ffn_up.weight, L2.ffn_down.weight, K2, V2)
    print("    Layer 2 complete")
    let h3 = transformer_layer(h2, L3.attn_norm.weight, L3.attn_q.weight, L3.attn_k.weight, L3.attn_v.weight, L3.attn_output.weight, L3.ffn_norm.weight, L3.ffn_gate.weight, L3.ffn_up.weight, L3.ffn_down.weight, K3, V3)
    print("    Layer 3 complete")
    let h4 = transformer_layer(h3, L4.attn_norm.weight, L4.attn_q.weight, L4.attn_k.weight, L4.attn_v.weight, L4.attn_output.weight, L4.ffn_norm.weight, L4.ffn_gate.weight, L4.ffn_up.weight, L4.ffn_down.weight, K4, V4)
    print("    Layer 4 complete")
    print("")

    print("[6] Final normalization...")
    let final_norm = rms_norm(h4, output_norm)
    let logits = linear(final_norm, output)
    print("    Logits: shape = {}", shape(logits))
    print("")

    print("✅ 5-layer test PASSED!")
    print("   All reshape operations completed successfully")
}
