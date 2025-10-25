// ============================================================================
// Attention Mask Demonstration
// ============================================================================
//
// アテンションマスクの使い方を実演します。
//
// 機能：
//   1. apply_mask(scores, mask) - マスクを適用
//   2. causal_mask(seq_len) - 因果マスク生成
//
// 用途：
//   - Transformer Decoderの自己回帰アテンション
//   - パディングトークンの無視
//   - 特定位置へのアテンション制限
// ============================================================================

main {
    print("=" * 70)
    print("Attention Mask Demonstration")
    print("=" * 70)

    // ========================================
    // Example 1: Basic Attention Mask
    // ========================================
    print("\nExample 1: Basic Attention Mask")
    print("-" * 70)

    // Attention scores [seq_len, seq_len]
    tensor scores: float16[3, 3] = [[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0],
                                     [7.0, 8.0, 9.0]]

    print("\nOriginal Attention Scores [3, 3]:")
    print("  Row 0:", [scores[0, 0], scores[0, 1], scores[0, 2]])
    print("  Row 1:", [scores[1, 0], scores[1, 1], scores[1, 2]])
    print("  Row 2:", [scores[2, 0], scores[2, 1], scores[2, 2]])

    // Mask: 1.0 = keep, 0.0 = mask out
    tensor mask: float16[3, 3] = [[1.0, 0.0, 1.0],
                                   [1.0, 1.0, 0.0],
                                   [0.0, 1.0, 1.0]]

    print("\nMask [3, 3]:")
    print("  Row 0:", [mask[0, 0], mask[0, 1], mask[0, 2]], "← Masks position 1")
    print("  Row 1:", [mask[1, 0], mask[1, 1], mask[1, 2]], "← Masks position 2")
    print("  Row 2:", [mask[2, 0], mask[2, 1], mask[2, 2]], "← Masks position 0")

    // Apply mask
    tensor masked_scores: float16[3, 3] = apply_mask(scores, mask)

    print("\nMasked Attention Scores [3, 3]:")
    print("  Row 0:", [masked_scores[0, 0], masked_scores[0, 1], masked_scores[0, 2]])
    print("  Row 1:", [masked_scores[1, 0], masked_scores[1, 1], masked_scores[1, 2]])
    print("  Row 2:", [masked_scores[2, 0], masked_scores[2, 1], masked_scores[2, 2]])
    print("\n  Note: Masked positions are set to -10000.0")
    print("        After softmax, these become ~0.0")

    // ========================================
    // Example 2: Causal Mask for Decoder
    // ========================================
    print("\n" + "=" * 70)
    print("Example 2: Causal Mask for Autoregressive Models")
    print("=" * 70)

    // Generate causal mask
    tensor causal: float16[4, 4] = causal_mask(4)

    print("\nCausal Mask [4, 4]:")
    print("  Row 0:", [causal[0, 0], causal[0, 1], causal[0, 2], causal[0, 3]], "← Position 0 can only see itself")
    print("  Row 1:", [causal[1, 0], causal[1, 1], causal[1, 2], causal[1, 3]], "← Position 1 sees 0,1")
    print("  Row 2:", [causal[2, 0], causal[2, 1], causal[2, 2], causal[2, 3]], "← Position 2 sees 0,1,2")
    print("  Row 3:", [causal[3, 0], causal[3, 1], causal[3, 2], causal[3, 3]], "← Position 3 sees all")

    print("\nStructure:")
    print("  [[1, 0, 0, 0],   ← Lower triangular matrix")
    print("   [1, 1, 0, 0],   ← Prevents attending to future positions")
    print("   [1, 1, 1, 0],   ← Used in Transformer Decoder self-attention")
    print("   [1, 1, 1, 1]]")

    // Apply causal mask to attention scores
    tensor decoder_scores: float16[4, 4] = [[1.0, 2.0, 3.0, 4.0],
                                             [5.0, 6.0, 7.0, 8.0],
                                             [9.0, 10.0, 11.0, 12.0],
                                             [13.0, 14.0, 15.0, 16.0]]

    print("\nDecoder Attention Scores [4, 4]:")
    print("  Row 0:", [decoder_scores[0, 0], decoder_scores[0, 1], decoder_scores[0, 2], decoder_scores[0, 3]])
    print("  Row 1:", [decoder_scores[1, 0], decoder_scores[1, 1], decoder_scores[1, 2], decoder_scores[1, 3]])
    print("  Row 2:", [decoder_scores[2, 0], decoder_scores[2, 1], decoder_scores[2, 2], decoder_scores[2, 3]])
    print("  Row 3:", [decoder_scores[3, 0], decoder_scores[3, 1], decoder_scores[3, 2], decoder_scores[3, 3]])

    tensor causal_masked: float16[4, 4] = apply_mask(decoder_scores, causal)

    print("\nAfter Causal Masking [4, 4]:")
    print("  Row 0:", [causal_masked[0, 0], causal_masked[0, 1], causal_masked[0, 2], causal_masked[0, 3]])
    print("  Row 1:", [causal_masked[1, 0], causal_masked[1, 1], causal_masked[1, 2], causal_masked[1, 3]])
    print("  Row 2:", [causal_masked[2, 0], causal_masked[2, 1], causal_masked[2, 2], causal_masked[2, 3]])
    print("  Row 3:", [causal_masked[3, 0], causal_masked[3, 1], causal_masked[3, 2], causal_masked[3, 3]])

    print("\n  Future positions are masked with -10000.0")

    // ========================================
    // Summary
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nAttention Mask Functions:")
    print("  1. apply_mask(scores, mask)")
    print("     - Applies mask to attention scores")
    print("     - mask[i,j] = 0.0 → scores[i,j] = -10000.0")
    print("     - mask[i,j] = 1.0 → scores[i,j] unchanged")
    print("")
    print("  2. causal_mask(seq_len)")
    print("     - Generates lower triangular mask")
    print("     - Prevents attending to future positions")
    print("     - Essential for autoregressive generation")

    print("\nUse Cases:")
    print("  - Transformer Decoder: causal mask for autoregression")
    print("  - Padding: mask out padding tokens")
    print("  - Selective Attention: focus on specific positions")
    print("  - Cross-Attention: encoder-decoder attention patterns")

    print("\nIntegration with Softmax:")
    print("  scores_masked = apply_mask(scores, mask)")
    print("  attention_weights = softmax(scores_masked)")
    print("  → Masked positions become ~0.0 after softmax")

    print("\n" + "=" * 70)
    print("End of Attention Mask Demonstration")
    print("=" * 70)
}
