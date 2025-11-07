// Compare f32 vs f16 outputs side by side
main {
    print("=== Comparing F32 vs F16 ===")
    print("")

    let home = env("HOME")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    // Test prompt
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + "Hello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.tokenize(chat_prompt, true)

    // F32 version
    print("[1/2] Testing F32 version...")
    let model_f32 = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    let x_f32 = embedding(model_f32.token_embd.weight, tokens)
    let K0_f32 = linear(x_f32, model_f32.blk[0].attn_k.weight)
    let token_f32 = temperature_sample(K0_f32, 0.8)
    print("  F32 first K0 sample token: {}", token_f32)

    // F16 version
    print("[2/2] Testing F16 version...")
    let model_f16 = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    let x_f16 = embedding(model_f16.token_embd.weight, tokens)
    let K0_f16 = linear(x_f16, model_f16.blk[0].attn_k.weight)
    let token_f16 = temperature_sample(K0_f16, 0.8)
    print("  F16 first K0 sample token: {}", token_f16)

    print("")
    print("=== Comparison Complete ===")
}
