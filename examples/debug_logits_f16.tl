// Debug f16 logits values
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let silu_result = silu(gate)
    let mul_result = silu_result * up
    result := linear(mul_result, W_down)
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
    result := after_attn + ffn_out
}

main {
    print("=== Debug F16 Logits ===")
    print("")

    print("Loading model...")
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    let L0 = model.blk[0]
    let L1 = model.blk[1]

    print("Tokenizing...")
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + "Hello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)

    print("Running inference...")
    let x = embedding(tok_embd, tokens)

    print("  Shape of x after embedding:")
    let x_shape = shape(x)
    print("    x: [{}, {}]", x_shape[0], x_shape[1])

    let K0 = linear(x, L0.attn_k.weight)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = linear(x, L1.attn_k.weight)
    let V1 = linear(x, L1.attn_v.weight)

    let h0 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)

    print("  Shape of h0 after layer 0:")
    let h0_shape = shape(h0)
    print("    h0: [{}, {}]", h0_shape[0], h0_shape[1])

    let h1 = transformer_layer(h0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)

    print("  Shape of h1 after layer 1:")
    let h1_shape = shape(h1)
    print("    h1: [{}, {}]", h1_shape[0], h1_shape[1])

    let final_norm = rms_norm(h1, output_norm)

    print("  Shape after final norm:")
    let norm_shape = shape(final_norm)
    print("    final_norm: [{}, {}]", norm_shape[0], norm_shape[1])

    let logits = linear(final_norm, output)

    print("  Shape of logits:")
    let logits_shape = shape(logits)
    print("    logits: [{}, {}]", logits_shape[0], logits_shape[1])

    print("")
    print("Sampling token...")
    let temperature = 0.8
    let token_id = temperature_sample(logits, temperature)
    print("  Token ID: {}", token_id)

    let text = detokenize_single(tokenizer, token_id, false)
    print("  Text: '{}'", text)

    print("")
    print("=== Test Complete ===")
}
