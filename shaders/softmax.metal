//! Softmax Operation
//!
//! GPU-accelerated softmax implementation with numerical stability

#include <metal_stdlib>
using namespace metal;

/// Softmax kernel with reduction for finding max and sum
///
/// Computes softmax over the last dimension of the input tensor.
/// Uses threadgroup memory for reduction operations.
///
/// Numerically stable: subtracts max before exp to prevent overflow
/// Handles NaN and Inf: replaces invalid values with 0
///
/// Input:  [batch, last_dim]
/// Output: [batch, last_dim]
///
/// Each threadgroup processes one batch (row)
kernel void softmax_f16(
    device const half* input [[buffer(0)]],      // Input tensor
    device half* output [[buffer(1)]],           // Output tensor
    constant uint& last_dim [[buffer(2)]],       // Size of last dimension
    threadgroup float* shared_max [[threadgroup(0)]],   // Shared memory for max reduction
    threadgroup float* shared_sum [[threadgroup(1)]],   // Shared memory for sum reduction
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint offset = batch_id * last_dim;

    // Phase 1: Find maximum value for numerical stability
    // Each thread processes multiple elements
    float local_max = -INFINITY;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        // Only consider finite values
        if (isfinite(val)) {
            local_max = max(local_max, val);
        }
    }

    // Store local max to shared memory
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle case where all values are NaN/Inf
    if (isinf(max_val) || isnan(max_val)) {
        max_val = 0.0f;
    }

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;  // Replace NaN/Inf with 0
        }

        output[offset + i] = half(exp_val);
        local_sum += exp_val;
    }

    // Store local sum to shared memory
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize by sum
    if (sum > 0.0f && isfinite(sum)) {
        // Normal case: divide by sum
        for (uint i = tid; i < last_dim; i += tg_size) {
            float val = float(output[offset + i]);
            output[offset + i] = half(val / sum);
        }
    } else {
        // Degenerate case: uniform distribution
        float uniform = 1.0f / float(last_dim);
        for (uint i = tid; i < last_dim; i += tg_size) {
            output[offset + i] = half(uniform);
        }
    }
}

/// Simple softmax kernel for small dimensions (no reduction needed)
///
/// For last_dim <= 256, single thread can handle the entire row
kernel void softmax_simple_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint offset = batch_id * last_dim;

    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        if (isfinite(val)) {
            max_val = max(max_val, val);
        }
    }

    if (!isfinite(max_val)) {
        max_val = 0.0f;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;
        }

        output[offset + i] = half(exp_val);
        sum += exp_val;
    }

    // Normalize
    if (sum > 0.0f && isfinite(sum)) {
        for (uint i = 0; i < last_dim; i++) {
            float val = float(output[offset + i]);
            output[offset + i] = half(val / sum);
        }
    } else {
        float uniform = 1.0f / float(last_dim);
        for (uint i = 0; i < last_dim; i++) {
            output[offset + i] = half(uniform);
        }
    }
}

// F32 version
kernel void softmax_f32(
    device const float* input [[buffer(0)]],      // Input tensor
    device float* output [[buffer(1)]],           // Output tensor
    constant uint& last_dim [[buffer(2)]],       // Size of last dimension
    threadgroup float* shared_max [[threadgroup(0)]],   // Shared memory for max reduction
    threadgroup float* shared_sum [[threadgroup(1)]],   // Shared memory for sum reduction
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint offset = batch_id * last_dim;

    // Phase 1: Find maximum value for numerical stability
    // Each thread processes multiple elements
    float local_max = -INFINITY;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        // Only consider finite values
        if (isfinite(val)) {
            local_max = max(local_max, val);
        }
    }

    // Store local max to shared memory
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle case where all values are NaN/Inf
    if (isinf(max_val) || isnan(max_val)) {
        max_val = 0.0f;
    }

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;  // Replace NaN/Inf with 0
        }

        output[offset + i] = float(exp_val);
        local_sum += exp_val;
    }

    // Store local sum to shared memory
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize by sum
    if (sum > 0.0f && isfinite(sum)) {
        // Normal case: divide by sum
        for (uint i = tid; i < last_dim; i += tg_size) {
            float val = float(output[offset + i]);
            output[offset + i] = float(val / sum);
        }
    } else {
        // Degenerate case: uniform distribution
        float uniform = 1.0f / float(last_dim);
        for (uint i = tid; i < last_dim; i += tg_size) {
            output[offset + i] = float(uniform);
        }
    }
}

// F32 version
kernel void softmax_simple_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint offset = batch_id * last_dim;

    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        if (isfinite(val)) {
            max_val = max(max_val, val);
        }
    }

    if (!isfinite(max_val)) {
        max_val = 0.0f;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;
        }

        output[offset + i] = float(exp_val);
        sum += exp_val;
    }

    // Normalize
    if (sum > 0.0f && isfinite(sum)) {
        for (uint i = 0; i < last_dim; i++) {
            float val = float(output[offset + i]);
            output[offset + i] = float(val / sum);
        }
    } else {
        float uniform = 1.0f / float(last_dim);
        for (uint i = 0; i < last_dim; i++) {
            output[offset + i] = float(uniform);
        }
    }
}
