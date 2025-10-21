// ============================================================================
// Batch Normalization Demonstration
// ============================================================================
//
// バッチ正規化の使い方を実演します。
//
// 機能：
//   batch_norm(x, gamma, beta, eps) - バッチ次元で正規化
//
// 数式：
//   y = gamma * ((x - mean) / sqrt(var + eps)) + beta
//
//   where:
//     mean = (1/N) Σ x_i  (across batch)
//     var = (1/N) Σ (x_i - mean)^2
//
// 用途：
//   - ニューラルネットワークの学習安定化
//   - 内部共変量シフトの軽減
//   - 高い学習率の使用を可能に
// ============================================================================

main {
    print("=" * 70)
    print("Batch Normalization Demonstration")
    print("=" * 70)

    // ========================================
    // Example 1: Basic Batch Normalization
    // ========================================
    print("\nExample 1: Basic Batch Normalization")
    print("-" * 70)

    // Input: [batch=3, features=2]
    tensor x: float16[3, 2] = [[1.0, 4.0],    // Sample 0
                                [2.0, 5.0],    // Sample 1
                                [3.0, 6.0]]    // Sample 2

    print("\nInput X [batch=3, features=2]:")
    print("  Sample 0:", [x[0, 0], x[0, 1]])
    print("  Sample 1:", [x[1, 0], x[1, 1]])
    print("  Sample 2:", [x[2, 0], x[2, 1]])

    print("\nStatistics before normalization:")
    print("  Feature 0: values = [1.0, 2.0, 3.0]")
    print("             mean = 2.0, std ≈ 0.816")
    print("  Feature 1: values = [4.0, 5.0, 6.0]")
    print("             mean = 5.0, std ≈ 0.816")

    // Gamma (scale) and Beta (shift) parameters
    tensor gamma: float16[2] = [1.0, 1.0]  // No scaling
    tensor beta: float16[2] = [0.0, 0.0]   // No shift

    print("\nParameters:")
    print("  Gamma (scale):", gamma)
    print("  Beta (shift): ", beta)
    print("  Epsilon:       1e-5 (numerical stability)")

    // Apply batch normalization
    tensor normalized: float16[3, 2] = batch_norm(x, gamma, beta, 0.00001)

    print("\nNormalized Output [3, 2]:")
    print("  Sample 0:", [normalized[0, 0], normalized[0, 1]])
    print("  Sample 1:", [normalized[1, 0], normalized[1, 1]])
    print("  Sample 2:", [normalized[2, 0], normalized[2, 1]])

    print("\nInterpretation:")
    print("  Each feature now has mean ≈ 0.0 and std ≈ 1.0")
    print("  Feature 0: [-1.22, 0.0, 1.22] (normalized)")
    print("  Feature 1: [-1.22, 0.0, 1.22] (normalized)")

    // ========================================
    // Example 2: Batch Norm with Learnable Parameters
    // ========================================
    print("\n" + "=" * 70)
    print("Example 2: Batch Norm with Learnable Parameters")
    print("=" * 70)

    // Input data
    tensor data: float16[4, 3] = [[0.0, 1.0, 2.0],
                                   [1.0, 2.0, 3.0],
                                   [2.0, 3.0, 4.0],
                                   [3.0, 4.0, 5.0]]

    print("\nInput Data [batch=4, features=3]:")
    print("  Sample 0:", [data[0, 0], data[0, 1], data[0, 2]])
    print("  Sample 1:", [data[1, 0], data[1, 1], data[1, 2]])
    print("  Sample 2:", [data[2, 0], data[2, 1], data[2, 2]])
    print("  Sample 3:", [data[3, 0], data[3, 1], data[3, 2]])

    // Learnable parameters (scale and shift)
    tensor scale: float16[3] learnable = [2.0, 1.5, 1.0]
    tensor shift: float16[3] learnable = [1.0, 0.5, 0.0]

    print("\nLearnable Parameters:")
    print("  Scale (gamma):", scale)
    print("  Shift (beta): ", shift)

    // Apply batch norm with affine transformation
    tensor output: float16[4, 3] = batch_norm(data, scale, shift, 0.00001)

    print("\nOutput after Batch Norm [4, 3]:")
    print("  Sample 0:", [output[0, 0], output[0, 1], output[0, 2]])
    print("  Sample 1:", [output[1, 0], output[1, 1], output[1, 2]])
    print("  Sample 2:", [output[2, 0], output[2, 1], output[2, 2]])
    print("  Sample 3:", [output[3, 0], output[3, 1], output[3, 2]])

    print("\nAffine Transformation:")
    print("  Feature 0: normalized * 2.0 + 1.0")
    print("  Feature 1: normalized * 1.5 + 0.5")
    print("  Feature 2: normalized * 1.0 + 0.0")

    // ========================================
    // Summary
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nBatch Normalization Function:")
    print("  batch_norm(x, gamma, beta, eps)")
    print("")
    print("  Arguments:")
    print("    x:     Input tensor [batch, features]")
    print("    gamma: Scale parameter [features] (learnable)")
    print("    beta:  Shift parameter [features] (learnable)")
    print("    eps:   Small constant for numerical stability (1e-5)")

    print("\nFormula:")
    print("  1. Compute batch statistics:")
    print("     mean = (1/N) Σ x_i")
    print("     var  = (1/N) Σ (x_i - mean)^2")
    print("")
    print("  2. Normalize:")
    print("     x_norm = (x - mean) / sqrt(var + eps)")
    print("")
    print("  3. Affine transformation:")
    print("     y = gamma * x_norm + beta")

    print("\nKey Properties:")
    print("  - Normalizes each feature independently")
    print("  - Computed across batch dimension")
    print("  - gamma and beta are learnable parameters")
    print("  - Reduces internal covariate shift")
    print("  - Enables higher learning rates")

    print("\nUse Cases:")
    print("  - Deep neural networks (after linear/conv layers)")
    print("  - Training stabilization")
    print("  - Accelerated convergence")
    print("  - Regularization effect (similar to dropout)")

    print("\nDifference from Layer Normalization:")
    print("  Batch Norm:  Normalizes across batch dimension")
    print("  Layer Norm:  Normalizes across feature dimension")
    print("  → Batch Norm requires batch statistics")
    print("  → Layer Norm works with batch_size=1")

    print("\n" + "=" * 70)
    print("End of Batch Normalization Demonstration")
    print("=" * 70)
}
