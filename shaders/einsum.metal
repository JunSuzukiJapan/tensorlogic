//! Einsum Operations for Attention Mechanisms
//!
//! Specialized einsum implementations for common attention patterns.
//! These are optimized for the specific tensor contractions used in
//! multi-head attention and grouped query attention.

#include <metal_stdlib>
using namespace metal;

/// Einsum: "ihd,jhd->ihj" - Batched dot product for attention scores
///
/// Computes attention scores for multi-head attention:
/// For each head h, computes dot product between query position i and key position j.
///
/// Inputs:
///   A: [I, H, D] - Query tensor (position, heads, head_dim)
///   B: [J, H, D] - Key tensor (position, heads, head_dim)
/// Output:
///   C: [I, H, J] - Attention scores
///
/// Computation: C[i,h,j] = sum_d A[i,h,d] * B[j,h,d]
///
/// Memory access pattern:
/// - Each thread computes one element C[i,h,j]
/// - Reads D elements from A and B
/// - No threadgroup memory needed for this pattern (small D=64)
kernel void einsum_ihd_jhd_ihj_f16(
    device const half* A [[buffer(0)]],      // [I, H, D]
    device const half* B [[buffer(1)]],      // [J, H, D]
    device half* C [[buffer(2)]],            // [I, H, J]
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, j]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint j = gid.z;  // Key position

    // Bounds check
    if (i >= I || h >= H || j >= J) {
        return;
    }

    // Compute dot product: sum over dimension d
    half sum = 0.0h;

    // A[i, h, :] · B[j, h, :]
    for (uint d = 0; d < D; d++) {
        uint a_idx = (i * H + h) * D + d;  // A[i, h, d]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * J + j;  // C[i, h, j]
    C[c_idx] = sum;
}

/// Optimized version with threadgroup memory for larger D
///
/// For D > 128, uses reduction in threadgroup memory
/// Currently not used (D=64 for TinyLlama), but available for larger models
kernel void einsum_ihd_jhd_ihj_f16_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& I [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& J [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Threadgroup shared memory for partial sums
    threadgroup half partial_sums[256];

    uint i = tgid.x;
    uint h = tgid.y;
    uint j = tgid.z;

    if (i >= I || h >= H || j >= J) {
        return;
    }

    // Each thread computes partial sum over chunk of D
    half sum = 0.0h;
    for (uint d = tid; d < D; d += 256) {
        uint a_idx = (i * H + h) * D + d;
        uint b_idx = (j * H + h) * D + d;
        sum += A[a_idx] * B[b_idx];
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes result
    if (tid == 0) {
        uint c_idx = (i * H + h) * J + j;
        C[c_idx] = partial_sums[0];
    }
}

/// Einsum: "ihj,jhd->ihd" - Weighted sum for attention output
///
/// Computes attention output by weighted sum of values:
/// For each head h and query position i, computes weighted sum over key positions j.
///
/// Inputs:
///   A: [I, H, J] - Attention weights (after softmax)
///   B: [J, H, D] - Value tensor
/// Output:
///   C: [I, H, D] - Attention output
///
/// Computation: C[i,h,d] = sum_j A[i,h,j] * B[j,h,d]
///
/// Memory access pattern:
/// - Each thread computes one element C[i,h,d]
/// - Reads J elements from A and B
kernel void einsum_ihj_jhd_ihd_f16(
    device const half* A [[buffer(0)]],      // [I, H, J] - attention weights
    device const half* B [[buffer(1)]],      // [J, H, D] - values
    device half* C [[buffer(2)]],            // [I, H, D] - output
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key/value)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, d]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint d = gid.z;  // Output dimension

    // Bounds check
    if (i >= I || h >= H || d >= D) {
        return;
    }

    // Compute weighted sum: sum over positions j
    half sum = 0.0h;

    for (uint j = 0; j < J; j++) {
        uint a_idx = (i * H + h) * J + j;  // A[i, h, j]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * D + d;  // C[i, h, d]
    C[c_idx] = sum;
}

/// General einsum for 3D tensors with arbitrary contraction
///
/// This is a fallback for other einsum patterns not covered above.
/// Uses a general loop structure that can handle various index patterns.
///
/// Note: This is slower than specialized kernels but more flexible.
kernel void einsum_3d_general_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint3& shape_a [[buffer(3)]],   // Shape of A
    constant uint3& shape_b [[buffer(4)]],   // Shape of B
    constant uint3& shape_c [[buffer(5)]],   // Shape of C
    constant uint& contract_dim [[buffer(6)]], // Which dimension to contract
    uint3 gid [[thread_position_in_grid]]
) {
    // This would need index mapping configuration
    // For now, specialized kernels are sufficient
    // TODO: Implement if needed for other einsum patterns
}
