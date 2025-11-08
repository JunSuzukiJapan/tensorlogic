// Test with single transformer layer (no KV cache, no autoregressive)
// Just: embedding → 1 layer → output

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

main {
    print("=== Single Layer Transformer Test ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let L0 = model.blk[0]

    print("Model loaded - testing layer 0")

    // Simple prompt
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)
    let seq_len_shape = shape(x)
    let seq_len = seq_len_shape[0]

    print("Input shape: {}, seq_len: {}", shape(x), seq_len)

    // Single transformer layer (no KV cache - just forward pass)
    // Attention
    let normed = rms_norm(x, L0.attn_norm.weight)
    let Q = linear(normed, L0.attn_q.weight)
    let K = linear(normed, L0.attn_k.weight)
    let V = linear(normed, L0.attn_v.weight)

    // Simple attention (no cache)
    let attn_out = attention(Q, K, V, L0.attn_output.weight)
    let after_attn = x + attn_out

    // FFN
    let normed2 = rms_norm(after_attn, L0.ffn_norm.weight)
    let ffn_out = swiglu_ffn(normed2, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight)
    let h = after_attn + ffn_out

    print("After layer 0: {}", shape(h))

    // Output
    let final_norm = rms_norm(h, output_norm)
    let logits = linear(final_norm, tok_embd)

    print("Logits shape: {}", shape(logits))

    // Sample
    let token_id = temperature_sample(logits, 0.0)
    let text = detokenize_single(tokenizer, token_id, false)

    print("")
    print("Generated token: '{}' (id={})", text, token_id)
    print("")
    print("=== Test complete ===")
}
