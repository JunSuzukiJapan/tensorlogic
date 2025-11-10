//! Advanced Kernel Fusion - Multi-operation Chains
//!
//! Combines 3-5 operations in single kernels for maximum performance.
//! Expected improvement: 2-3x throughput for neural network inference/training.
//!
//! Common patterns in neural networks:
//! 1. Linear → BatchNorm → Activation (forward pass)
//! 2. Activation → Dropout → Linear (residual connections)
//! 3. Linear → Activation → Add (skip connections)

#include <metal_stdlib>
using namespace metal;

/// Fused: Linear + BatchNorm + ReLU
///
/// Common pattern in CNN/ResNet forward pass.
/// Combines matmul, batch normalization, and activation.
///
/// output = relu(batchnorm(matmul(x, w) + bias))
///        = relu((matmul(x, w) + bias - mean) / sqrt(var + eps) * gamma + beta)
kernel void fused_linear_batchnorm_relu_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device const half* bn_mean [[buffer(3)]],    // BatchNorm mean [N]
    device const half* bn_var [[buffer(4)]],     // BatchNorm variance [N]
    device const half* bn_gamma [[buffer(5)]],   // BatchNorm scale [N]
    device const half* bn_beta [[buffer(6)]],    // BatchNorm shift [N]
    device half* output [[buffer(7)]],           // Output [M, N]
    constant uint& M [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant half& eps [[buffer(11)]],           // Small constant for numerical stability
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    half normalized = (sum - bn_mean[col]) / sqrt(bn_var[col] + eps);
    half scaled = normalized * bn_gamma[col] + bn_beta[col];

    // 3. ReLU activation
    output[row * N + col] = max(scaled, half(0.0));
}

/// Fused: Linear + Residual + ReLU
///
/// Common in ResNet skip connections:
/// output = relu(matmul(x, w) + bias + residual)
kernel void fused_linear_residual_relu_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device const half* residual [[buffer(3)]],   // Residual connection [M, N]
    device half* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. Add residual
    sum += residual[row * N + col];

    // 3. ReLU activation
    output[row * N + col] = max(sum, half(0.0));
}

/// Fused: Dropout + Linear
///
/// Applies dropout then linear transformation.
/// Used in attention mechanisms and transformer layers.
///
/// output = matmul(dropout(x, mask, keep_prob), w) + bias
kernel void fused_dropout_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const uint* dropout_mask [[buffer(1)]], // Dropout mask [M, K] (0 or 1)
    device const half* w [[buffer(2)]],          // Weight [K, N]
    device const half* bias [[buffer(3)]],       // Bias [N]
    device half* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant half& scale [[buffer(8)]],          // 1.0 / keep_prob for scaling
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with dropout applied inline
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Apply dropout: multiply by mask and scale
        uint mask_val = dropout_mask[row * K + k];
        half x_val = x[row * K + k] * half(mask_val) * scale;
        sum += x_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: LayerNorm + Linear
///
/// Common in transformer architectures.
/// Applies layer normalization then linear transformation.
///
/// output = matmul(layernorm(x), w) + bias
kernel void fused_layernorm_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* ln_gamma [[buffer(1)]],   // LayerNorm scale [K]
    device const half* ln_beta [[buffer(2)]],    // LayerNorm shift [K]
    device const half* w [[buffer(3)]],          // Weight [K, N]
    device const half* bias [[buffer(4)]],       // Bias [N]
    device half* output [[buffer(5)]],           // Output [M, N]
    constant uint& M [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant half& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Compute mean and variance for this row (layer norm is per-sample)
    half mean = 0.0h;
    for (uint k = 0; k < K; k++) {
        mean += x[row * K + k];
    }
    mean /= half(K);

    half variance = 0.0h;
    for (uint k = 0; k < K; k++) {
        half diff = x[row * K + k] - mean;
        variance += diff * diff;
    }
    variance /= half(K);

    // 2. Compute matmul with normalized input
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        half normalized = (x[row * K + k] - mean) / sqrt(variance + eps);
        half scaled = normalized * ln_gamma[k] + ln_beta[k];

        // Multiply with weight
        sum += scaled * w[k * N + col];
    }

    // 3. Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: GELU + Linear
///
/// Common in transformer feed-forward networks.
/// Applies GELU activation then linear transformation.
///
/// output = matmul(gelu(x), w) + bias
kernel void fused_gelu_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device half* output [[buffer(3)]],           // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with GELU applied inline
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        half x_val = x[row * K + k];
        half x3 = x_val * x_val * x_val;
        half inner = half(0.7978845608) * (x_val + half(0.044715) * x3);
        half gelu_val = half(0.5) * x_val * (half(1.0) + tanh(inner));

        // Multiply with weight
        sum += gelu_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: Softmax + CrossEntropy Loss
///
/// Combines softmax and cross-entropy loss computation.
/// Used in classification tasks for efficiency.
///
/// Computes: -log(softmax(logits)[target_class])
kernel void fused_softmax_crossentropy_f16(
    device const half* logits [[buffer(0)]],     // Input logits [M, N]
    device const uint* targets [[buffer(1)]],    // Target classes [M]
    device half* loss [[buffer(2)]],             // Output loss [M]
    constant uint& M [[buffer(3)]],              // Batch size
    constant uint& N [[buffer(4)]],              // Number of classes
    uint id [[thread_position_in_grid]]
) {
    if (id >= M) return;

    uint row = id;
    uint target_class = targets[row];

    // 1. Find max for numerical stability
    half max_logit = logits[row * N];
    for (uint i = 1; i < N; i++) {
        max_logit = max(max_logit, logits[row * N + i]);
    }

    // 2. Compute exp(logits - max) and sum
    half sum_exp = 0.0h;
    for (uint i = 0; i < N; i++) {
        sum_exp += exp(logits[row * N + i] - max_logit);
    }

    // 3. Compute log_softmax for target class
    half log_softmax = (logits[row * N + target_class] - max_logit) - log(sum_exp);

    // 4. Cross-entropy loss: -log(softmax(target))
    loss[row] = -log_softmax;
}

/// Fused: Attention Score Computation
///
/// Computes attention scores: softmax(Q @ K^T / sqrt(d_k))
/// Core operation in transformer attention mechanism.
kernel void fused_attention_scores_f16(
    device const half* Q [[buffer(0)]],          // Query [M, d_k]
    device const half* K [[buffer(1)]],          // Key [N, d_k]
    device half* scores [[buffer(2)]],           // Output scores [M, N]
    constant uint& M [[buffer(3)]],              // Query sequence length
    constant uint& N [[buffer(4)]],              // Key sequence length
    constant uint& d_k [[buffer(5)]],            // Dimension
    constant half& scale [[buffer(6)]],          // 1 / sqrt(d_k)
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // Query position
    uint col = gid.x;  // Key position

    if (row >= M || col >= N) return;

    // 1. Compute Q @ K^T (scaled dot product)
    half dot_product = 0.0h;
    for (uint k = 0; k < d_k; k++) {
        dot_product += Q[row * d_k + k] * K[col * d_k + k];
    }
    dot_product *= scale;

    // Note: Softmax should be applied across the N dimension
    // This kernel computes the scaled dot products
    // A separate kernel or pass is needed for softmax normalization
    scores[row * N + col] = dot_product;
}

// F32 version
kernel void fused_linear_batchnorm_relu_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device const float* bn_mean [[buffer(3)]],    // BatchNorm mean [N]
    device const float* bn_var [[buffer(4)]],     // BatchNorm variance [N]
    device const float* bn_gamma [[buffer(5)]],   // BatchNorm scale [N]
    device const float* bn_beta [[buffer(6)]],    // BatchNorm shift [N]
    device float* output [[buffer(7)]],           // Output [M, N]
    constant uint& M [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant half& eps [[buffer(11)]],           // Small constant for numerical stability
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    float normalized = (sum - bn_mean[col]) / sqrt(bn_var[col] + eps);
    float scaled = normalized * bn_gamma[col] + bn_beta[col];

    // 3. ReLU activation
    output[row * N + col] = max(scaled, float(0.0));
}

// F32 version
kernel void fused_linear_residual_relu_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device const float* residual [[buffer(3)]],   // Residual connection [M, N]
    device float* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. Add residual
    sum += residual[row * N + col];

    // 3. ReLU activation
    output[row * N + col] = max(sum, float(0.0));
}

// F32 version
kernel void fused_dropout_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const uint* dropout_mask [[buffer(1)]], // Dropout mask [M, K] (0 or 1)
    device const float* w [[buffer(2)]],          // Weight [K, N]
    device const float* bias [[buffer(3)]],       // Bias [N]
    device float* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant half& scale [[buffer(8)]],          // 1.0 / keep_prob for scaling
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with dropout applied inline
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Apply dropout: multiply by mask and scale
        uint mask_val = dropout_mask[row * K + k];
        float x_val = x[row * K + k] * float(mask_val) * scale;
        sum += x_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_layernorm_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* ln_gamma [[buffer(1)]],   // LayerNorm scale [K]
    device const float* ln_beta [[buffer(2)]],    // LayerNorm shift [K]
    device const float* w [[buffer(3)]],          // Weight [K, N]
    device const float* bias [[buffer(4)]],       // Bias [N]
    device float* output [[buffer(5)]],           // Output [M, N]
    constant uint& M [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant half& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Compute mean and variance for this row (layer norm is per-sample)
    float mean = 0.0f;
    for (uint k = 0; k < K; k++) {
        mean += x[row * K + k];
    }
    mean /= float(K);

    float variance = 0.0f;
    for (uint k = 0; k < K; k++) {
        float diff = x[row * K + k] - mean;
        variance += diff * diff;
    }
    variance /= float(K);

    // 2. Compute matmul with normalized input
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        float normalized = (x[row * K + k] - mean) / sqrt(variance + eps);
        float scaled = normalized * ln_gamma[k] + ln_beta[k];

        // Multiply with weight
        sum += scaled * w[k * N + col];
    }

    // 3. Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_gelu_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device float* output [[buffer(3)]],           // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with GELU applied inline
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_val = x[row * K + k];
        float x3 = x_val * x_val * x_val;
        float inner = float(0.7978845608) * (x_val + float(0.044715) * x3);
        float gelu_val = float(0.5) * x_val * (float(1.0) + tanh(inner));

        // Multiply with weight
        sum += gelu_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_softmax_crossentropy_f32(
    device const float* logits [[buffer(0)]],     // Input logits [M, N]
    device const uint* targets [[buffer(1)]],    // Target classes [M]
    device float* loss [[buffer(2)]],             // Output loss [M]
    constant uint& M [[buffer(3)]],              // Batch size
    constant uint& N [[buffer(4)]],              // Number of classes
    uint id [[thread_position_in_grid]]
) {
    if (id >= M) return;

    uint row = id;
    uint target_class = targets[row];

    // 1. Find max for numerical stability
    float max_logit = logits[row * N];
    for (uint i = 1; i < N; i++) {
        max_logit = max(max_logit, logits[row * N + i]);
    }

    // 2. Compute exp(logits - max) and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < N; i++) {
        sum_exp += exp(logits[row * N + i] - max_logit);
    }

    // 3. Compute log_softmax for target class
    float log_softmax = (logits[row * N + target_class] - max_logit) - log(sum_exp);

    // 4. Cross-entropy loss: -log(softmax(target))
    loss[row] = -log_softmax;
}

// F32 version
kernel void fused_attention_scores_f32(
    device const float* Q [[buffer(0)]],          // Query [M, d_k]
    device const float* K [[buffer(1)]],          // Key [N, d_k]
    device float* scores [[buffer(2)]],           // Output scores [M, N]
    constant uint& M [[buffer(3)]],              // Query sequence length
    constant uint& N [[buffer(4)]],              // Key sequence length
    constant uint& d_k [[buffer(5)]],            // Dimension
    constant half& scale [[buffer(6)]],          // 1 / sqrt(d_k)
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // Query position
    uint col = gid.x;  // Key position

    if (row >= M || col >= N) return;

    // 1. Compute Q @ K^T (scaled dot product)
    float dot_product = 0.0f;
    for (uint k = 0; k < d_k; k++) {
        dot_product += Q[row * d_k + k] * K[col * d_k + k];
    }
    dot_product *= scale;

    // Note: Softmax should be applied across the N dimension
    // This kernel computes the scaled dot products
    // A separate kernel or pass is needed for softmax normalization
    scores[row * N + col] = dot_product;
}
