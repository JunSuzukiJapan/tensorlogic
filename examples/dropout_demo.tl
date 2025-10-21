// ============================================================================
// Dropout Demonstration
// ============================================================================
//
// ドロップアウト正則化の使い方を実演します。
//
// 機能：
//   dropout(x, p, training) - ランダムに要素をゼロ化
//
// 数式（Training mode）：
//   y_i = {
//     0           with probability p
//     x_i/(1-p)   with probability (1-p)
//   }
//
// 数式（Inference mode）：
//   y_i = x_i  (unchanged)
//
// 用途：
//   - ニューラルネットワークの正則化
//   - 過学習の防止
//   - モデルのアンサンブル効果
// ============================================================================

main {
    print("=" * 70)
    print("Dropout Demonstration")
    print("=" * 70)

    // ========================================
    // Example 1: Dropout in Training Mode
    // ========================================
    print("\nExample 1: Dropout in Training Mode")
    print("-" * 70)

    // Input tensor
    tensor x: float16[2, 4] = [[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0]]

    print("\nInput X [2, 4]:")
    print("  Row 0:", [x[0, 0], x[0, 1], x[0, 2], x[0, 3]])
    print("  Row 1:", [x[1, 0], x[1, 1], x[1, 2], x[1, 3]])

    // Apply dropout with p=0.5 (50% dropout rate)
    print("\nApplying dropout with p=0.5 (training mode)")
    tensor dropped: float16[2, 4] = dropout(x, 0.5, true)

    print("\nOutput after Dropout [2, 4]:")
    print("  Row 0:", [dropped[0, 0], dropped[0, 1], dropped[0, 2], dropped[0, 3]])
    print("  Row 1:", [dropped[1, 0], dropped[1, 1], dropped[1, 2], dropped[1, 3]])

    print("\nObservations:")
    print("  - ~50% of elements are set to 0.0 (dropped)")
    print("  - Remaining elements are scaled by 1/(1-0.5) = 2.0")
    print("  - This maintains expected value: E[output] = input")
    print("  - Each run produces different random dropout pattern")

    // ========================================
    // Example 2: Dropout in Inference Mode
    // ========================================
    print("\n" + "=" * 70)
    print("Example 2: Dropout in Inference Mode")
    print("=" * 70)

    // Same input
    print("\nInput X [2, 4]:")
    print("  Row 0:", [x[0, 0], x[0, 1], x[0, 2], x[0, 3]])
    print("  Row 1:", [x[1, 0], x[1, 1], x[1, 2], x[1, 3]])

    // Apply dropout in inference mode (training=false)
    print("\nApplying dropout with p=0.5 (inference mode)")
    tensor inference: float16[2, 4] = dropout(x, 0.5, false)

    print("\nOutput in Inference Mode [2, 4]:")
    print("  Row 0:", [inference[0, 0], inference[0, 1], inference[0, 2], inference[0, 3]])
    print("  Row 1:", [inference[1, 0], inference[1, 1], inference[1, 2], inference[1, 3]])

    print("\nObservations:")
    print("  - All elements remain unchanged")
    print("  - No dropout applied during inference")
    print("  - Output identical to input")
    print("  - Deterministic behavior for production")

    // ========================================
    // Example 3: Different Dropout Rates
    // ========================================
    print("\n" + "=" * 70)
    print("Example 3: Different Dropout Rates")
    print("=" * 70)

    tensor data: float16[10] = [1.0, 1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0]

    print("\nInput (all 1.0s) [10]:", data)

    // Low dropout (p=0.2)
    print("\np=0.2 (20% dropout, light regularization):")
    tensor drop_20: float16[10] = dropout(data, 0.2, true)
    print("  Output:", drop_20)
    print("  → ~80% elements kept, scaled by 1.25")

    // Medium dropout (p=0.5)
    print("\np=0.5 (50% dropout, moderate regularization):")
    tensor drop_50: float16[10] = dropout(data, 0.5, true)
    print("  Output:", drop_50)
    print("  → ~50% elements kept, scaled by 2.0")

    // High dropout (p=0.7)
    print("\np=0.7 (70% dropout, strong regularization):")
    tensor drop_70: float16[10] = dropout(data, 0.7, true)
    print("  Output:", drop_70)
    print("  → ~30% elements kept, scaled by 3.33")

    // ========================================
    // Summary
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nDropout Function:")
    print("  dropout(x, p, training)")
    print("")
    print("  Arguments:")
    print("    x:        Input tensor")
    print("    p:        Dropout probability [0.0, 1.0]")
    print("    training: Boolean (true=training, false=inference)")

    print("\nBehavior:")
    print("  Training mode (training=true):")
    print("    - Randomly sets p% of elements to 0.0")
    print("    - Scales remaining by 1/(1-p)")
    print("    - Different mask each forward pass")
    print("")
    print("  Inference mode (training=false):")
    print("    - Returns input unchanged")
    print("    - Deterministic output")
    print("    - No dropout applied")

    print("\nScaling Factor:")
    print("  p=0.2 → scale = 1/(1-0.2) = 1.25")
    print("  p=0.5 → scale = 1/(1-0.5) = 2.0")
    print("  p=0.7 → scale = 1/(1-0.7) = 3.33")
    print("  → Maintains E[output] = input")

    print("\nKey Benefits:")
    print("  - Prevents overfitting")
    print("  - Reduces co-adaptation of neurons")
    print("  - Implicit ensemble of many sub-networks")
    print("  - Simple and effective regularization")

    print("\nTypical Usage:")
    print("  - Fully connected layers: p=0.5")
    print("  - Input layer: p=0.1 to 0.2")
    print("  - Convolutional layers: p=0.1 to 0.3")
    print("  - Recurrent layers: p=0.2 to 0.5")

    print("\nIntegration in Neural Network:")
    print("  1. Training:")
    print("     hidden = relu(linear(input))")
    print("     dropped = dropout(hidden, 0.5, true)")
    print("     output = linear(dropped)")
    print("")
    print("  2. Inference:")
    print("     hidden = relu(linear(input))")
    print("     dropped = dropout(hidden, 0.5, false)  ← No dropout")
    print("     output = linear(dropped)")

    print("\n" + "=" * 70)
    print("End of Dropout Demonstration")
    print("=" * 70)
}
