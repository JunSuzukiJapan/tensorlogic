// Debug sampling and token generation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
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
    print("=== Debug Sampling ===")
    print("")

    print("Loading model...")
    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    // Use all 22 layers
    let L0 = model.blk[0]
    let L1 = model.blk[1]

    print("Tokenizing prompt...")
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + "Hello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)

    print("Running inference through 2 layers...")
    let x = embedding(tok_embd, tokens)

    // Build KV caches
    let K0 = linear(x, L0.attn_k.weight)
    let V0 = linear(x, L0.attn_v.weight)
    let K1 = linear(x, L1.attn_k.weight)
    let V1 = linear(x, L1.attn_v.weight)

    // Run through layers
    let h0 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)
    let h1 = transformer_layer(h0, L1.attn_norm.weight, L1.attn_q.weight, L1.attn_k.weight, L1.attn_v.weight, L1.attn_output.weight, L1.ffn_norm.weight, L1.ffn_gate.weight, L1.ffn_up.weight, L1.ffn_down.weight, K1, V1)

    let final_norm = rms_norm(h1, output_norm)
    let logits = linear(final_norm, output)

    print("")
    print("Testing different sampling methods...")
    print("")

    // Test 1: Temperature = 0.0 (greedy)
    print("1. Greedy sampling (temp=0.0):")
    let token_greedy = temperature_sample(logits, 0.0)
    let text_greedy = detokenize_single(tokenizer, token_greedy, false)
    print("   Token ID: {}", token_greedy)
    print("   Text: '{}'", text_greedy)
    print("")

    // Test 2: Temperature = 0.8
    print("2. Temperature sampling (temp=0.8):")
    let token_temp = temperature_sample(logits, 0.8)
    let text_temp = detokenize_single(tokenizer, token_temp, false)
    print("   Token ID: {}", token_temp)
    print("   Text: '{}'", text_temp)
    print("")

    // Test 3: Try with skip_special_tokens = true
    print("3. Same token with skip_special_tokens=true:")
    let text_skip = detokenize_single(tokenizer, token_temp, true)
    print("   Text: '{}'", text_skip)
    print("")

    // Test 4: Try some known token IDs
    print("4. Testing known token IDs:")
    let text_1 = detokenize_single(tokenizer, 1, false)
    let text_2 = detokenize_single(tokenizer, 2, false)
    let text_10 = detokenize_single(tokenizer, 10, false)
    let text_100 = detokenize_single(tokenizer, 100, false)
    let text_1000 = detokenize_single(tokenizer, 1000, false)
    print("   ID 1: '{}'", text_1)
    print("   ID 2: '{}'", text_2)
    print("   ID 10: '{}'", text_10)
    print("   ID 100: '{}'", text_100)
    print("   ID 1000: '{}'", text_1000)

    print("")
    print("=== Test Complete ===")
}
