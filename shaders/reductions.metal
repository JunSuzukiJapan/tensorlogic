//! Metal GPU kernels for reduction operations

#include <metal_stdlib>
using namespace metal;

// Global reduction: sum all elements (f16)
// Uses two-stage reduction: local (threadgroup) then global
kernel void sum_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Each thread loads one element
    half local_sum = (gid < count) ? input[gid] : half(0.0);

    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread in each threadgroup writes result
    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: sum all elements (f32)
// Uses two-stage reduction: local (threadgroup) then global
kernel void sum_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Each thread loads one element
    float local_sum = (gid < count) ? input[gid] : 0.0f;

    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread in each threadgroup writes result
    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: mean of all elements
kernel void mean_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Same as sum, but divide by count at the end
    half local_sum = (gid < count) ? input[gid] : half(0.0);

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: max of all elements
kernel void max_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    half local_max = (gid < count) ? input[gid] : -INFINITY;

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: min of all elements
kernel void min_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    half local_min = (gid < count) ? input[gid] : INFINITY;

    shared[tid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Dimension-specific reduction: sum along one dimension
// For example, reducing [M, N, K] along dim=1 produces [M, K]
kernel void sum_dim_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],  // [dim0, dim1, dim2, ...]
    constant uint* output_shape [[buffer(3)]], // Shape after reduction
    constant uint& rank [[buffer(4)]],         // Number of dimensions
    constant uint& reduce_dim [[buffer(5)]],   // Which dimension to reduce
    uint gid [[thread_position_in_grid]]
) {
    // Calculate output index from global thread ID
    if (gid >= output_shape[0]) return; // Bounds check

    // Compute total output size
    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    // Convert linear output index to multi-dimensional indices
    uint temp = gid;
    uint output_indices[8]; // Max 8 dimensions
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    // Sum over the reduction dimension
    half sum = half(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        // Construct input index by inserting i at reduce_dim
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        // Convert multi-dimensional index to linear index
        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum;
}

// Dimension-specific reduction: mean along one dimension
kernel void mean_dim_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& reduce_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_shape[0]) return;

    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    uint temp = gid;
    uint output_indices[8];
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    half sum = half(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum / half(reduce_size);
}

// F32 version
kernel void mean_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Same as sum, but divide by count at the end
    float local_sum = (gid < count) ? input[gid] : float(0.0);

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void max_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    float local_max = (gid < count) ? input[gid] : -INFINITY;

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void min_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    float local_min = (gid < count) ? input[gid] : INFINITY;

    shared[tid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void sum_dim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],  // [dim0, dim1, dim2, ...]
    constant uint* output_shape [[buffer(3)]], // Shape after reduction
    constant uint& rank [[buffer(4)]],         // Number of dimensions
    constant uint& reduce_dim [[buffer(5)]],   // Which dimension to reduce
    uint gid [[thread_position_in_grid]]
) {
    // Calculate output index from global thread ID
    if (gid >= output_shape[0]) return; // Bounds check

    // Compute total output size
    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    // Convert linear output index to multi-dimensional indices
    uint temp = gid;
    uint output_indices[8]; // Max 8 dimensions
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    // Sum over the reduction dimension
    float sum = float(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        // Construct input index by inserting i at reduce_dim
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        // Convert multi-dimensional index to linear index
        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum;
}

// F32 version
kernel void mean_dim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& reduce_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_shape[0]) return;

    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    uint temp = gid;
    uint output_indices[8];
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    float sum = float(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum / float(reduce_size);
}
