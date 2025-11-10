#include <metal_stdlib>
using namespace metal;

/// Layer Normalization kernel
/// Normalizes input across the last dimension: (x - mean) / sqrt(variance + eps)
/// Then applies affine transformation if weight and bias are provided
kernel void layer_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],     // nullable
    device const half* bias [[buffer(2)]],       // nullable
    device half* output [[buffer(3)]],
    device const half* normalized_size_ptr [[buffer(4)]],
    device const half* eps_ptr [[buffer(5)]],
    device const half* has_weight_ptr [[buffer(6)]],
    device const half* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > half(0.5));
    bool has_bias = (has_bias_ptr[0] > half(0.5));
    // Shared memory for reduction
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];

    uint offset = batch_idx * normalized_size;

    // Phase 1: Compute mean (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        local_sum += float(input[offset + i]);
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(normalized_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance (parallel reduction)
    float local_sq_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        float diff = float(input[offset + i]) - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for squared sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = shared_sq_sum[0] / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and apply affine transformation
    for (uint i = tid; i < normalized_size; i += tsize) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = half(normalized);
    }
}

/// Simplified layer normalization for small tensors (single thread per batch)
kernel void layer_norm_simple_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const half* normalized_size_ptr [[buffer(4)]],
    device const half* eps_ptr [[buffer(5)]],
    device const half* has_weight_ptr [[buffer(6)]],
    device const half* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > half(0.5));
    bool has_bias = (has_bias_ptr[0] > half(0.5));
    uint offset = batch_idx * normalized_size;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum += float(input[offset + i]);
    }
    float mean = sum / float(normalized_size);

    // Compute variance
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = float(input[offset + i]) - mean;
        sq_sum += diff * diff;
    }
    float variance = sq_sum / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transformation
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = half(normalized);
    }
}
// ============================================================================
// RMS Normalization (Root Mean Square Normalization)
// Used in LLaMA, TinyLlama models
// Formula: output = (x / rms(x)) * weight
// where rms(x) = sqrt(mean(x^2) + eps)
// ============================================================================

/// RMS Norm (simple version for small tensors)
kernel void rms_norm_simple_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* normalized_size_ptr [[buffer(3)]],
    device const half* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Compute RMS: sqrt(mean(x^2) + eps)
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float val = float(input[offset + i]);
        sq_sum += val * val;
    }
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = half(scaled);
    }
}

/// RMS Norm (optimized version with threadgroup reduction for large tensors)
kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* normalized_size_ptr [[buffer(3)]],
    device const half* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Thread-local sum for reduction
    threadgroup float local_sums[256];
    float thread_sq_sum = 0.0f;

    // Parallel computation of squared sum
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float val = float(input[offset + i]);
        thread_sq_sum += val * val;
    }
    local_sums[tid] = thread_sq_sum;

    // Reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_sum = local_sums[0];
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight (parallel)
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = half(scaled);
    }
}

// F32 version
kernel void layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],     // nullable
    device const float* bias [[buffer(2)]],       // nullable
    device float* output [[buffer(3)]],
    device const float* normalized_size_ptr [[buffer(4)]],
    device const float* eps_ptr [[buffer(5)]],
    device const float* has_weight_ptr [[buffer(6)]],
    device const float* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > float(0.5));
    bool has_bias = (has_bias_ptr[0] > float(0.5));
    // Shared memory for reduction
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];

    uint offset = batch_idx * normalized_size;

    // Phase 1: Compute mean (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        local_sum += float(input[offset + i]);
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(normalized_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance (parallel reduction)
    float local_sq_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        float diff = float(input[offset + i]) - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for squared sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = shared_sq_sum[0] / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and apply affine transformation
    for (uint i = tid; i < normalized_size; i += tsize) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = float(normalized);
    }
}

// F32 version
kernel void layer_norm_simple_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* normalized_size_ptr [[buffer(4)]],
    device const float* eps_ptr [[buffer(5)]],
    device const float* has_weight_ptr [[buffer(6)]],
    device const float* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > float(0.5));
    bool has_bias = (has_bias_ptr[0] > float(0.5));
    uint offset = batch_idx * normalized_size;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum += float(input[offset + i]);
    }
    float mean = sum / float(normalized_size);

    // Compute variance
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = float(input[offset + i]) - mean;
        sq_sum += diff * diff;
    }
    float variance = sq_sum / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transformation
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = float(normalized);
    }
}

// F32 version
kernel void rms_norm_simple_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* normalized_size_ptr [[buffer(3)]],
    device const float* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Compute RMS: sqrt(mean(x^2) + eps)
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float val = float(input[offset + i]);
        sq_sum += val * val;
    }
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = float(scaled);
    }
}

// F32 version
kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* normalized_size_ptr [[buffer(3)]],
    device const float* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Thread-local sum for reduction
    threadgroup float local_sums[256];
    float thread_sq_sum = 0.0f;

    // Parallel computation of squared sum
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float val = float(input[offset + i]);
        thread_sq_sum += val * val;
    }
    local_sums[tid] = thread_sq_sum;

    // Reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_sum = local_sums[0];
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight (parallel)
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = float(scaled);
    }
}
