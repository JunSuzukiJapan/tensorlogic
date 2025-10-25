// Test basic model operations: embedding + linear projection

main {
    print("=== Basic Model Operations Test ===")

    // Load model
    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("[1] Loading embedding table...")
    let embed_table = get_tensor(model, "token_embd.weight")
    print("Embedding table shape:", shape(embed_table))

    // Test embedding lookup for single token (BOS token = 1)
    print("[2] Testing embedding lookup...")
    let test_tokens = [1]  // BOS token
    let embeddings = embedding(embed_table, test_tokens)
    print("Embedding shape:", shape(embeddings))
    print("Embedding (first 5 values):", embeddings)

    // Load Q projection weight
    print("[3] Testing linear projection (Q)...")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    print("W_q shape:", shape(W_q))

    // Apply linear projection
    let Q_proj = linear(embeddings, W_q)
    print("Q projection shape:", shape(Q_proj))
    print("Q projection (first 5 values):", Q_proj)

    print("[4] Checking for NaN/Inf...")
    print("If you see NaN or Inf above, there's a problem")
}
