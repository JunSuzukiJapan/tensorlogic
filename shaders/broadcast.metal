#include <metal_stdlib>
using namespace metal;

/// Broadcast tensor to target shape (f16)
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - input_shape: Shape of input tensor (max 8 dimensions)
/// - target_shape: Shape of output tensor (max 8 dimensions)
/// - input_ndim: Number of dimensions in input
/// - target_ndim: Number of dimensions in target
kernel void broadcast_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* target_shape [[buffer(3)]],
    device const uint* input_ndim [[buffer(4)]],
    device const uint* target_ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint target_n_dims = target_ndim[0];
    uint input_n_dims = input_ndim[0];

    uint total_elements = 1;
    for (uint i = 0; i < target_n_dims; i++) {
        total_elements *= target_shape[i];
    }

    if (gid >= total_elements) return;

    // Compute target multi-index from linear index
    uint target_idx[8];
    uint remaining = gid;
    for (int i = target_n_dims - 1; i >= 0; i--) {
        target_idx[i] = remaining % target_shape[i];
        remaining /= target_shape[i];
    }

    // Compute input strides
    uint input_strides[8];
    uint stride = 1;
    for (int i = input_n_dims - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    // Map target index to input index (align from right)
    uint rank_diff = target_n_dims - input_n_dims;
    uint input_linear = 0;
    for (uint i = rank_diff; i < target_n_dims; i++) {
        uint input_i = i - rank_diff;
        uint coord = (input_shape[input_i] == 1) ? 0 : target_idx[i];
        input_linear += coord * input_strides[input_i];
    }

    output[gid] = input[input_linear];
}

/// Broadcast tensor to target shape (f32)
kernel void broadcast_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* target_shape [[buffer(3)]],
    device const uint* input_ndim [[buffer(4)]],
    device const uint* target_ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint target_n_dims = target_ndim[0];
    uint input_n_dims = input_ndim[0];

    uint total_elements = 1;
    for (uint i = 0; i < target_n_dims; i++) {
        total_elements *= target_shape[i];
    }

    if (gid >= total_elements) return;

    uint target_idx[8];
    uint remaining = gid;
    for (int i = target_n_dims - 1; i >= 0; i--) {
        target_idx[i] = remaining % target_shape[i];
        remaining /= target_shape[i];
    }

    uint input_strides[8];
    uint stride = 1;
    for (int i = input_n_dims - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    uint rank_diff = target_n_dims - input_n_dims;
    uint input_linear = 0;
    for (uint i = rank_diff; i < target_n_dims; i++) {
        uint input_i = i - rank_diff;
        uint coord = (input_shape[input_i] == 1) ? 0 : target_idx[i];
        input_linear += coord * input_strides[input_i];
    }

    output[gid] = input[input_linear];
}
