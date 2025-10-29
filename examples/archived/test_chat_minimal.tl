// Minimal chat demo test to debug hang

main {
    print("=== Minimal Chat Demo Debug ===")

    print("[1/3] Loading model...")
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer("/Users/suzukijun/.llm/tokenizers/tinyllama-tokenizer.json")
    print("      ✓ Model and tokenizer loaded")

    print("[2/3] Loading embedding table...")
    let embed_table = get_tensor(model, "token_embd.weight")
    print("      ✓ Embedding table loaded")

    print("[3/3] Testing embedding with TokenIds...")
    let test_tokens = tokenize(tokenizer, " Hello ", false)
    print("      Tokens:", test_tokens)

    print("      Calling embedding(embed_table, test_tokens)...")
    let e = embedding(embed_table, test_tokens)
    print("      ✓ Embedding done!")
    print("      Result:", e)

    print("=== Test Complete ===")
}
