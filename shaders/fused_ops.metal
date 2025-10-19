//! Fused operations for improved performance
//! Combines multiple operations into single kernels to reduce memory access

#include <metal_stdlib>
using namespace metal;

/// Fused add + relu: output = max(a + b, 0)
/// Reduces memory access by avoiding intermediate buffer
kernel void fused_add_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half sum = a[id] + b[id];
    output[id] = max(sum, half(0.0));
}

/// Fused multiply + relu: output = max(a * b, 0)
kernel void fused_mul_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half product = a[id] * b[id];
    output[id] = max(product, half(0.0));
}

/// Fused affine transformation: output = x * scale + bias
/// Used in batch normalization and similar operations
kernel void fused_affine_f16(
    device const half* x [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = x[id] * scale[id] + bias[id];
}

/// Fused linear layer: matmul + bias + optional activation
/// This is the most common pattern in neural networks
///
/// C[i,j] = sum_k(A[i,k] * B[k,j]) + bias[j] [+ activation]
kernel void fused_linear_f16(
    device const half* A [[buffer(0)]],  // Input [M, K]
    device const half* B [[buffer(1)]],  // Weight [K, N]
    device const half* bias [[buffer(2)]], // Bias [N] (can be null)
    device half* C [[buffer(3)]],        // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& activation [[buffer(7)]], // 0=none, 1=relu, 2=gelu
    constant bool& has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    // Compute matmul: C[row, col] = sum(A[row, k] * B[k, col])
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // Add bias if present
    if (has_bias) {
        sum += bias[col];
    }

    // Apply activation
    half result;
    if (activation == 1) {
        // ReLU
        result = max(sum, half(0.0));
    } else if (activation == 2) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        half x = sum;
        half x3 = x * x * x;
        half inner = half(0.7978845608) * (x + half(0.044715) * x3); // sqrt(2/π) ≈ 0.7978845608
        result = half(0.5) * x * (half(1.0) + tanh(inner));
    } else {
        // No activation
        result = sum;
    }

    C[row * N + col] = result;
}

/// Fused matmul + bias (no activation)
/// Simpler version when no activation is needed
kernel void fused_matmul_bias_f16(
    device const half* A [[buffer(0)]],  // [M, K]
    device const half* B [[buffer(1)]],  // [K, N]
    device const half* bias [[buffer(2)]], // [N]
    device half* C [[buffer(3)]],        // [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}

/// Fused subtract + relu: output = max(a - b, 0)
kernel void fused_sub_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half diff = a[id] - b[id];
    output[id] = max(diff, half(0.0));
}

/// Fused division + relu: output = max(a / b, 0)
kernel void fused_div_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half quotient = a[id] / b[id];
    output[id] = max(quotient, half(0.0));
}
