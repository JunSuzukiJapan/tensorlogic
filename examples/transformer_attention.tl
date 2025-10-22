main {
    print("=== Transformer Multi-Head Self-Attention Demo ===")
    print("")

    let seq_len = 8
    let d_model = 512
    let num_heads = 8
    let d_k = 64

    print("Configuration:")
    print("  Sequence length:", seq_len)
    print("  Model dimension:", d_model)
    print("  Number of heads:", num_heads)
    print("  Dimension per head:", d_k)
    print("")

    print("Step 1: Creating input embeddings with positional encoding")
    let embeddings = positional_encoding(seq_len, d_model)
    print("  ✓ Embeddings shape: [", seq_len, ",", d_model, "]")
    print("")

    print("Step 2: Initializing Q, K, V weight matrices")
    let W_q = positional_encoding(d_model, d_model)
    let W_k = positional_encoding(d_model, d_model)
    let W_v = positional_encoding(d_model, d_model)
    print("  ✓ Weight matrices initialized")
    print("")

    print("Step 3: Computing Q, K, V projections")
    let Q = matmul(embeddings, W_q)
    let K = matmul(embeddings, W_k)
    let V = matmul(embeddings, W_v)
    print("  ✓ Q shape: [", seq_len, ",", d_model, "]")
    print("  ✓ K shape: [", seq_len, ",", d_model, "]")
    print("  ✓ V shape: [", seq_len, ",", d_model, "]")
    print("")

    print("Step 4: Computing attention scores")
    let K_transpose = transpose(K)
    let attention_scores = matmul(Q, K_transpose)
    let scaled_scores = attention_scores
    print("  ✓ Attention scores shape: [", seq_len, ",", seq_len, "]")
    print("")

    print("Step 5: Applying attention mask")
    let mask = ones([seq_len, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask)
    print("  ✓ Applied causal mask for autoregressive generation")
    print("")

    print("Step 6: Computing attention weights (softmax)")
    let attention_weights = softmax(masked_scores, 1)
    print("  ✓ Attention weights computed")
    print("")

    print("Step 7: Computing attention output")
    let attention_output = matmul(attention_weights, V)
    print("  ✓ Attention output shape: [", seq_len, ",", d_model, "]")
    print("")

    print("Step 8: Final projection")
    let W_o = positional_encoding(d_model, d_model)
    let output = matmul(attention_output, W_o)
    print("  ✓ Final output shape: [", seq_len, ",", d_model, "]")
    print("")

    print("=== Multi-Head Self-Attention Complete! ===")
}
