// Compare RoPE behavior between 38 and 46 tokens

main {
    print("=== RoPE Comparison: 38 vs 46 Tokens ===")
    print("")

    // Load model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let L0_attn_norm = model.blk[0].attn_norm.weight
    let L0_W_q = model.blk[0].attn_q.weight

    // Test 1: 38 tokens
    print("=== Test 1: 38 Tokens ===")
    let system_prompt_38 = "You are a friendly chatbot."
    let user_message_38 = "Hello! How are you?"
    let formatted_prompt_38 = "<|system|>\n" + system_prompt_38 + "</s>\n<|user|>\n" + user_message_38 + "</s>\n<|assistant|>\n"

    let tokens_38 = tokenize(tokenizer, formatted_prompt_38, false)
    print("Token count:", len(tokens_38))

    let emb_38 = embedding(emb_weight, tokens_38)
    let normed_38 = rms_norm(emb_38, L0_attn_norm)
    let Q_38 = linear(normed_38, L0_W_q)
    print("Q shape:", shape(Q_38))
    print("Q sum:", sum(Q_38), " min:", min(Q_38), " max:", max(Q_38))

    let Q_heads_38 = reshape(Q_38, [38, 32, 64])
    print("Q_heads sum:", sum(Q_heads_38), " min:", min(Q_heads_38), " max:", max(Q_heads_38))

    let Q_rope_38 = rope(Q_heads_38)
    print("Q_rope sum:", sum(Q_rope_38))
    print("  min:", min(Q_rope_38), " max:", max(Q_rope_38))
    print("")

    // Test 2: 46 tokens
    print("=== Test 2: 46 Tokens ===")
    let system_prompt_46 = "You are a friendly and helpful AI assistant."
    let user_message_46 = "Hello! Tell me a short fun fact about computers."
    let formatted_prompt_46 = "<|system|>\n" + system_prompt_46 + "</s>\n<|user|>\n" + user_message_46 + "</s>\n<|assistant|>\n"

    let tokens_46 = tokenize(tokenizer, formatted_prompt_46, false)
    print("Token count:", len(tokens_46))

    let emb_46 = embedding(emb_weight, tokens_46)
    let normed_46 = rms_norm(emb_46, L0_attn_norm)
    let Q_46 = linear(normed_46, L0_W_q)
    print("Q shape:", shape(Q_46))
    print("Q sum:", sum(Q_46), " min:", min(Q_46), " max:", max(Q_46))

    let Q_heads_46 = reshape(Q_46, [46, 32, 64])
    print("Q_heads sum:", sum(Q_heads_46), " min:", min(Q_heads_46), " max:", max(Q_heads_46))

    print("Applying RoPE to 46 tokens...")
    let Q_rope_46 = rope(Q_heads_46)
    print("Q_rope sum:", sum(Q_rope_46))
    print("  min:", min(Q_rope_46), " max:", max(Q_rope_46))
    print("")

    print("=== Comparison ===")
    print("38 tokens Q_rope: sum=", sum(Q_rope_38), " range=[", min(Q_rope_38), ",", max(Q_rope_38), "]")
    print("46 tokens Q_rope: sum=", sum(Q_rope_46), " range=[", min(Q_rope_46), ",", max(Q_rope_46), "]")
    print("")
    print("=== Test Complete ===")
}
