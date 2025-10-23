main {
    print("=== Testing Token Embedding Layer ===")
    print("")

    print("Test 1: Simple embedding lookup")
    let vocab_size = 10
    let embedding_dim = 4

    print("  Creating embedding table: [vocab_size=", vocab_size, ", embedding_dim=", embedding_dim, "]")
    let weight = positional_encoding(vocab_size, embedding_dim)

    print("  Token IDs: [0, 2, 5, 1] (4 tokens)")
    let token_ids = [0.0, 2.0, 5.0, 1.0]

    let embeddings = embedding(weight, token_ids)
    print("  ✓ Embedding lookup completed")
    print("  Output shape:", shape(embeddings))
    print("  Expected: [4.0, 4.0] (4 tokens × 4 dimensions)")
    print("")

    print("Test 2: Batch embedding (using multiple 1D lookups)")
    let batch_size = 2
    let seq_len = 3

    print("  Batch size:", batch_size, ", Sequence length:", seq_len)
    print("  Creating token ID sequences...")

    let token_batch_1 = [0.0, 1.0, 2.0]
    let token_batch_2 = [3.0, 4.0, 5.0]

    let emb_1 = embedding(weight, token_batch_1)
    let emb_2 = embedding(weight, token_batch_2)

    print("  ✓ Batch 1 embeddings:", shape(emb_1))
    print("  ✓ Batch 2 embeddings:", shape(emb_2))
    print("  Expected: [3.0, 4.0] each (3 tokens × 4 dimensions)")
    print("")

    print("Test 3: Simulating embedding layer")
    let small_vocab = 100
    let small_dim = 8

    print("  Model config:")
    print("    - embedding_dim:", small_dim)
    print("    - vocab_size:", small_vocab)
    print("    - Table shape: [", small_vocab, ",", small_dim, "]")

    let emb_table = ones([small_vocab, small_dim])

    print("  Input prompt: 'Hello world' → [15, 72, 0, 45] (4 tokens)")
    let prompt_ids = [15.0, 72.0, 0.0, 45.0]

    let prompt_embeddings = embedding(emb_table, prompt_ids)
    print("  ✓ Prompt embeddings generated")
    print("  Output shape:", shape(prompt_embeddings))
    print("  Expected: [4.0, 8.0] (4 tokens × 8 dimensions)")
    print("")

    print("=== Embedding Layer Explanation ===")
    print("")
    print("Purpose:")
    print("  • Convert token IDs (integers) to dense vector representations")
    print("  • First step in Transformer forward pass")
    print("  • Each token gets a learned embedding_dim-dimensional vector")
    print("")
    print("Standard Format (PyTorch):")
    print("  • Embedding table: [vocab_size, embedding_dim]")
    print("  • Token ID i → Extract row i from table")
    print("  • Example: [32000, 2048] for TinyLlama")
    print("")
    print("TinyLlama Format (GGUF/llama.cpp):")
    print("  • Embedding weight: token_embd.weight [2048, 32000]")
    print("  • Shape is transposed: [embedding_dim, vocab_size]")
    print("  • Token ID i → Extract column i from weight matrix")
    print("")
    print("Operation:")
    print("  • Input: token_ids [seq_len] or [batch, seq_len]")
    print("  • Table: emb_table [vocab_size, embedding_dim]")
    print("  • Operation: Look up row token_id from table")
    print("  • Output: embeddings [seq_len, embedding_dim]")
    print("")
    print("Example:")
    print("  • Token ID 42 → Extract row 42 from embedding table")
    print("  • Result: Vector of size embedding_dim (e.g., 2048 floats)")
    print("  • This vector represents the token's meaning in model space")
    print("")

    print("=== Next Steps for Transformer ===")
    print("")
    print("1. ✅ Token Embedding (completed)")
    print("2. ✅ Positional Encoding (already available)")
    print("3. ✅ RMSNorm (already implemented)")
    print("4. ⏳ Multi-Head Attention (Grouped-Query Attention for TinyLlama)")
    print("5. ⏳ Feed-Forward Network (SwiGLU activation)")
    print("6. ⏳ Layer Stacking (22 layers for TinyLlama)")
    print("7. ⏳ Output Projection (to vocabulary logits)")
    print("")

    print("✅ Embedding layer tests completed!")
}
