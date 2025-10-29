// Debug version to find where it hangs

main {
    print("=== Debug: Finding the hang ===")
    print("")

    let EOS_TOKEN = 2

    print("[1/3] Loading model...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")
    print("  Model loaded")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight

    print("  Loading layer 0...")
    let L0 = model.blk[0]
    print("  ✓ Layer 0 loaded")

    print("")
    print("[2/3] Preparing prompt...")
    let user_msg = "Hello!"
    let chat_template = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
    let chat_prompt = chat_template + user_msg + "</s>\n<|assistant|>\n"
    print("  Tokenizing...")
    let tokens = tokenizer.tokenize(chat_prompt, true)
    print("  ✓ Tokenized")
    print("  User: {}", user_msg)

    print("")
    print("[3/3] Running inference...")
    print("  Step 1: Embedding")
    let x = embedding(tok_embd, tokens)
    print("  ✓ Embedding complete")

    print("  Step 2: Building KV cache for layer 0")
    print("    Computing K0...")
    let K0 = linear(x, L0.attn_k.weight)
    print("    ✓ K0 complete")
    print("    Computing V0...")
    let V0 = linear(x, L0.attn_v.weight)
    print("    ✓ V0 complete")

    print("  Step 3: Running attention")
    print("    RMS norm...")
    let normed = rms_norm(x, L0.attn_norm.weight)
    print("    ✓ RMS norm complete")
    print("    Computing Q...")
    let Q = linear(normed, L0.attn_q.weight)
    print("    ✓ Q complete")
    print("    Running attention_with_cache...")
    let attn_out = attention_with_cache(Q, K0, V0, L0.attn_output.weight)
    print("    ✓ Attention complete")

    print("")
    print("=== All steps completed successfully! ===")
}
