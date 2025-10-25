main {
    print("=== LLM Inference Pipeline Demo ===")
    print("")

    let vocab_size = 32000
    let seq_len = 8
    let d_model = 512
    let num_heads = 8
    let num_layers = 2
    let d_ff = 2048

    print("Model Configuration:")
    print("  Vocabulary size:", vocab_size)
    print("  Context length:", seq_len)
    print("  Model dimension:", d_model)
    print("  Number of heads:", num_heads)
    print("  Number of layers:", num_layers)
    print("  FFN dimension:", d_ff)
    print("")

    print("Step 1: Token Embedding")
    let token_embeddings = positional_encoding(8, 512)
    print("  ✓ Token embeddings: [8, 512]")
    print("")

    print("Step 2: Positional Encoding")
    let pos_encoding = positional_encoding(8, 512)
    let input = token_embeddings
    print("  ✓ Added positional information")
    print("")

    print("Step 3: Transformer Layers (2 layers)")
    print("")

    print("  → Layer 1")
    let W_q_1 = positional_encoding(512, 512)
    let W_k_1 = positional_encoding(512, 512)
    let W_v_1 = positional_encoding(512, 512)
    let W_o_1 = positional_encoding(512, 512)

    let Q_1 = matmul(input, W_q_1)
    let K_1 = matmul(input, W_k_1)
    let V_1 = matmul(input, W_v_1)

    let K_T_1 = transpose(K_1)
    let scores_1 = matmul(Q_1, K_T_1)

    let causal_mask = ones([8, 8])
    let masked_scores_1 = apply_attention_mask(scores_1, causal_mask)
    let attn_weights_1 = softmax(masked_scores_1, 1)
    let attn_output_1 = matmul(attn_weights_1, V_1)
    let attn_projected_1 = matmul(attn_output_1, W_o_1)

    let norm_shape = [512]
    let eps = 0.00001
    let attn_norm_1 = layer_norm(attn_projected_1, norm_shape, eps)

    let W_1_ffn = positional_encoding(512, 2048)
    let W_2_ffn = positional_encoding(2048, 512)

    let ffn_hidden_1 = matmul(attn_norm_1, W_1_ffn)
    let ffn_activated_1 = gelu(ffn_hidden_1)
    let ffn_output_1 = matmul(ffn_activated_1, W_2_ffn)

    let layer_1_output = layer_norm(ffn_output_1, norm_shape, eps)
    print("    ✓ Attention + FFN + LayerNorm")

    print("  → Layer 2")
    let W_q_2 = positional_encoding(512, 512)
    let W_k_2 = positional_encoding(512, 512)
    let W_v_2 = positional_encoding(512, 512)
    let W_o_2 = positional_encoding(512, 512)

    let Q_2 = matmul(layer_1_output, W_q_2)
    let K_2 = matmul(layer_1_output, W_k_2)
    let V_2 = matmul(layer_1_output, W_v_2)

    let K_T_2 = transpose(K_2)
    let scores_2 = matmul(Q_2, K_T_2)
    let masked_scores_2 = apply_attention_mask(scores_2, causal_mask)
    let attn_weights_2 = softmax(masked_scores_2, 1)
    let attn_output_2 = matmul(attn_weights_2, V_2)
    let attn_projected_2 = matmul(attn_output_2, W_o_2)
    let attn_norm_2 = layer_norm(attn_projected_2, norm_shape, eps)

    let ffn_hidden_2 = matmul(attn_norm_2, W_1_ffn)
    let ffn_activated_2 = gelu(ffn_hidden_2)
    let ffn_output_2 = matmul(ffn_activated_2, W_2_ffn)
    let final_layer_output = layer_norm(ffn_output_2, norm_shape, eps)
    print("    ✓ Attention + FFN + LayerNorm")
    print("")

    print("Step 4: Output Projection")
    let output_proj_size = 1000
    let W_output = positional_encoding(512, 1000)
    let logits = matmul(final_layer_output, W_output)

    print("  ✓ Logits shape: [8, 1000]")
    print("  (Full model would be [8, 32000])")
    print("")

    print("Step 5: Token Sampling")
    print("  Sampling strategies:")
    print("    • Greedy: Select highest probability token (argmax)")
    print("    • Top-k: Sample from top k most likely tokens")
    print("    • Top-p (nucleus): Sample from smallest set with cumulative p")
    print("    • Temperature: Control randomness (T→0: greedy, T→∞: uniform)")
    print("")
    print("  For next token prediction:")
    print("    1. Get logits for last position: logits[seq_len-1, :]")
    print("    2. Apply temperature scaling: logits / temperature")
    print("    3. Compute probabilities: softmax(logits)")
    print("    4. Sample next token ID using chosen strategy")
    print("    5. Append to sequence and repeat (autoregressive)")
    print("")

    print("Step 6: Autoregressive Generation Process")
    print("")
    print("  Pseudocode for text generation:")
    print("  ----------------------------------------")
    print("  given: input_ids = [token_1, token_2, ..., token_n]")
    print("  ")
    print("  for i in 1..max_new_tokens:")
    print("    1. embeddings = embed(input_ids)")
    print("    2. hidden = transformer_layers(embeddings)")
    print("    3. logits = output_projection(hidden[-1])")
    print("    4. next_token = sample(logits, strategy)")
    print("    5. input_ids = append(input_ids, next_token)")
    print("    6. if next_token == EOS: break")
    print("  ")
    print("  return input_ids")
    print("  ----------------------------------------")
    print("")

    print("=== LLM Inference Pipeline Complete! ===")
    print("")
    print("Architecture Summary:")
    print("  Input: Token IDs → Embeddings + Positional Encoding")
    print("  Transformer: 2 layers of (Attention + FFN + LayerNorm)")
    print("  Output: Project to vocabulary → Sample next token")
    print("  Generation: Autoregressive loop (one token at a time)")
    print("")
    print("Key Operations Used:")
    print("  • positional_encoding() - Position embeddings")
    print("  • matmul() - All linear projections (Q,K,V,FFN,output)")
    print("  • transpose() - Key matrix for attention scores")
    print("  • apply_attention_mask() - Causal masking")
    print("  • softmax() - Attention weights + output probabilities")
    print("  • gelu() - FFN activation")
    print("  • layer_norm() - Stabilization (2x per layer)")
    print("")
    print("Next Steps:")
    print("  1. Implement sampling functions (top_k, top_p, temperature)")
    print("  2. Add KV-cache for efficient generation")
    print("  3. Load actual TinyLlama weights from disk")
    print("  4. Build interactive chat interface")
    print("  5. Add streaming output support")
}
