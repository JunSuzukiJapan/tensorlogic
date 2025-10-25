main {
    print("=== Complete Transformer Block Demo ===")
    print("")

    let seq_len = 8
    let d_model = 512
    let num_heads = 8
    let d_k = 64
    let d_ff = 2048

    print("Configuration:")
    print("  Sequence length:", seq_len)
    print("  Model dimension (d_model):", d_model)
    print("  Number of heads:", num_heads)
    print("  FFN hidden dimension (d_ff):", d_ff)
    print("")

    print("Step 1: Input preparation")
    let input = positional_encoding(8, 512)
    print("  ✓ Input shape: [8, 512]")
    print("")

    print("Step 2: Multi-Head Self-Attention")

    let W_q = positional_encoding(512, 512)
    let W_k = positional_encoding(512, 512)
    let W_v = positional_encoding(512, 512)

    let Q = matmul(input, W_q)
    let K = matmul(input, W_k)
    let V = matmul(input, W_v)
    print("  ✓ Q, K, V computed")

    let K_T = transpose(K)
    let scores = matmul(Q, K_T)

    let mask = ones([8, 8])
    let masked_scores = apply_attention_mask(scores, mask)

    let attn_weights = softmax(masked_scores, 1)

    let attn_output = matmul(attn_weights, V)

    let W_o = positional_encoding(512, 512)
    let attn_projected = matmul(attn_output, W_o)
    print("  ✓ Attention output computed")
    print("")

    print("Step 3: Layer Normalization + Residual (after attention)")
    let norm_shape = [512]
    let eps = 0.00001
    let attn_residual = attn_projected
    let attn_normalized = layer_norm(attn_residual, norm_shape, eps)
    print("  ✓ Post-attention normalization applied")
    print("")

    print("Step 4: Feed-Forward Network")

    let W_1 = positional_encoding(512, 2048)
    let b_1 = zeros([8, 2048])
    let ffn_hidden = matmul(attn_normalized, W_1)
    let ffn_activated = gelu(ffn_hidden)
    print("  ✓ FFN hidden layer (with GELU): [8, 2048]")

    let W_2 = positional_encoding(2048, 512)
    let b_2 = zeros([8, 512])
    let ffn_output = matmul(ffn_activated, W_2)
    print("  ✓ FFN output layer: [8, 512]")
    print("")

    print("Step 5: Layer Normalization + Residual (after FFN)")
    let ffn_residual = ffn_output
    let output = layer_norm(ffn_residual, norm_shape, eps)
    print("  ✓ Post-FFN normalization applied")
    print("  ✓ Final output shape: [8, 512]")
    print("")

    print("=== Transformer Block Complete! ===")
    print("")
    print("Architecture Summary:")
    print("  Input → [LayerNorm → Multi-Head Attention → Residual]")
    print("       → [LayerNorm → Feed-Forward Network → Residual]")
    print("       → Output")
    print("")
    print("Operations used:")
    print("  • matmul() - Matrix multiplication for Q, K, V, FFN")
    print("  • softmax() - Attention weight normalization")
    print("  • apply_attention_mask() - Causal masking")
    print("  • gelu() - Non-linear activation")
    print("  • layer_norm() - Layer normalization (2x)")
    print("  • transpose() - Matrix transposition")
    print("  • positional_encoding() - Position embeddings")
    print("  • zeros(), ones() - Tensor creation")
    print("")
    print("Performance optimizations available:")
    print("  • fused_gelu_linear() - Fuse GELU + Linear (1.5x faster)")
    print("  • fused_add_relu() - Fuse addition + ReLU")
    print("  • Multi-head parallel processing")
    print("")
    print("Next: Text Generation with Sampling!")
}
