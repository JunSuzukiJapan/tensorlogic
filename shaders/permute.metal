#include <metal_stdlib>
using namespace metal;

/// Permute (transpose) tensor dimensions (f16)
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - input_shape: Shape of input tensor (max 8 dimensions)
/// - output_shape: Shape of output tensor (max 8 dimensions)
/// - perm: Permutation array (which input dim goes to which output dim)
/// - ndim: Number of dimensions
kernel void permute_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* output_shape [[buffer(3)]],
    device const uint* perm [[buffer(4)]],
    device const uint* ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n_dims = ndim[0];
    uint total_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        total_elements *= output_shape[i];
    }

    if (gid >= total_elements) return;

    // Calculate output multi-index from linear index
    uint output_idx[8];
    uint remaining = gid;
    for (int i = n_dims - 1; i >= 0; i--) {
        output_idx[i] = remaining % output_shape[i];
        remaining /= output_shape[i];
    }

    // Map output index to input index using permutation
    uint input_idx[8];
    for (uint i = 0; i < n_dims; i++) {
        input_idx[perm[i]] = output_idx[i];
    }

    // Convert input multi-index to linear index
    uint input_linear = 0;
    uint stride = 1;
    for (int i = n_dims - 1; i >= 0; i--) {
        input_linear += input_idx[i] * stride;
        stride *= input_shape[i];
    }

    output[gid] = input[input_linear];
}

/// Permute (transpose) tensor dimensions (f32)
kernel void permute_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* output_shape [[buffer(3)]],
    device const uint* perm [[buffer(4)]],
    device const uint* ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n_dims = ndim[0];
    uint total_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        total_elements *= output_shape[i];
    }

    if (gid >= total_elements) return;

    // Calculate output multi-index from linear index
    uint output_idx[8];
    uint remaining = gid;
    for (int i = n_dims - 1; i >= 0; i--) {
        output_idx[i] = remaining % output_shape[i];
        remaining /= output_shape[i];
    }

    // Map output index to input index using permutation
    uint input_idx[8];
    for (uint i = 0; i < n_dims; i++) {
        input_idx[perm[i]] = output_idx[i];
    }

    // Convert input multi-index to linear index
    uint input_linear = 0;
    uint stride = 1;
    for (int i = n_dims - 1; i >= 0; i--) {
        input_linear += input_idx[i] * stride;
        stride *= input_shape[i];
    }

    output[gid] = input[input_linear];
}
