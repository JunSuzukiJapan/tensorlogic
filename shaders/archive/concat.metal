#include <metal_stdlib>
using namespace metal;

/// Concatenate tensors along a specified dimension (f16)
/// Each invocation processes one element from input and writes to correct position in output
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - dim_offset: Offset along concat dimension (sum of sizes of previous tensors)
/// - input_dim_size: Size of this input along concat dimension
/// - output_dim_size: Size of output along concat dimension
/// - chunk_size: Product of dimensions after concat dim
/// - num_chunks: Product of dimensions before concat dim
kernel void concat_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* dim_offset [[buffer(2)]],
    device const uint* input_dim_size [[buffer(3)]],
    device const uint* output_dim_size [[buffer(4)]],
    device const uint* chunk_size [[buffer(5)]],
    device const uint* num_chunks [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint offset = dim_offset[0];
    uint in_dim = input_dim_size[0];
    uint out_dim = output_dim_size[0];
    uint chunk_sz = chunk_size[0];
    uint n_chunks = num_chunks[0];

    // Total elements in input
    uint total_elements = n_chunks * in_dim * chunk_sz;
    if (gid >= total_elements) return;

    // Decompose gid into (chunk_idx, dim_idx, elem_idx)
    uint chunk_idx = gid / (in_dim * chunk_sz);
    uint remainder = gid % (in_dim * chunk_sz);
    uint dim_idx = remainder / chunk_sz;
    uint elem_idx = remainder % chunk_sz;

    // Output index: chunk_idx * (output_dim_size * chunk_size) + (offset + dim_idx) * chunk_size + elem_idx
    uint output_idx = chunk_idx * out_dim * chunk_sz + (offset + dim_idx) * chunk_sz + elem_idx;

    output[output_idx] = input[gid];
}

/// Concatenate tensors along a specified dimension (f32)
kernel void concat_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dim_offset [[buffer(2)]],
    device const uint* input_dim_size [[buffer(3)]],
    device const uint* output_dim_size [[buffer(4)]],
    device const uint* chunk_size [[buffer(5)]],
    device const uint* num_chunks [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint offset = dim_offset[0];
    uint in_dim = input_dim_size[0];
    uint out_dim = output_dim_size[0];
    uint chunk_sz = chunk_size[0];
    uint n_chunks = num_chunks[0];

    uint total_elements = n_chunks * in_dim * chunk_sz;
    if (gid >= total_elements) return;

    uint chunk_idx = gid / (in_dim * chunk_sz);
    uint remainder = gid % (in_dim * chunk_sz);
    uint dim_idx = remainder / chunk_sz;
    uint elem_idx = remainder % chunk_sz;

    uint output_idx = chunk_idx * out_dim * chunk_sz + (offset + dim_idx) * chunk_sz + elem_idx;

    output[output_idx] = input[gid];
}
