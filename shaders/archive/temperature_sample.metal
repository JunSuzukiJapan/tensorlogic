#include <metal_stdlib>
using namespace metal;

/// Temperature sampling kernel (f16)
///
/// Step 1: Apply temperature scaling and compute softmax probabilities
/// Parameters:
/// - logits: Input logits [vocab_size]
/// - probs: Output probabilities [vocab_size]
/// - temperature: Temperature scaling factor
/// - vocab_size: Size of vocabulary
kernel void temperature_softmax_f16(
    device const half* logits [[buffer(0)]],
    device half* probs [[buffer(1)]],
    device const float* temperature [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vocab = vocab_size[0];
    float temp = temperature[0];

    if (gid >= vocab) return;

    // Apply temperature scaling
    float scaled_logit = float(logits[gid]) / temp;
    probs[gid] = half(scaled_logit);
}

/// Find maximum logit value (reduction kernel) - f16
kernel void find_max_f16(
    device const half* logits [[buffer(0)]],
    device half* max_val [[buffer(1)]],
    device const uint* vocab_size [[buffer(2)]],
    threadgroup half* shared_max [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];

    // Load value into shared memory
    half val = (gid < vocab) ? logits[gid] : half(-65504.0); // -inf for f16
    shared_max[lid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        max_val[0] = shared_max[0];
    }
}

/// Compute softmax probabilities (exp and normalize) - f16
kernel void softmax_normalize_f16(
    device half* probs [[buffer(0)]],
    device const half* max_val [[buffer(1)]],
    device half* sum_exp [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    threadgroup float* shared_sum [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];
    float max_logit = float(max_val[0]);

    // Compute exp(logit - max) and store in probs
    float exp_val = 0.0;
    if (gid < vocab) {
        exp_val = exp(float(probs[gid]) - max_logit);
        probs[gid] = half(exp_val);
    }
    shared_sum[lid] = exp_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction sum in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write sum
    if (lid == 0) {
        sum_exp[0] = half(shared_sum[0]);
    }
}

/// Normalize by sum to get final probabilities - f16
kernel void divide_by_sum_f16(
    device half* probs [[buffer(0)]],
    device const half* sum_exp [[buffer(1)]],
    device const uint* vocab_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vocab = vocab_size[0];
    if (gid >= vocab) return;

    float sum = float(sum_exp[0]);
    probs[gid] = half(float(probs[gid]) / sum);
}

/// Sample from probability distribution using cumulative sum - f16
/// Uses binary search for efficiency with large vocabularies
kernel void cumulative_sample_f16(
    device const half* probs [[buffer(0)]],
    device uint* sampled_token [[buffer(1)]],
    device const float* random_value [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return; // Only one thread does the sampling

    uint vocab = vocab_size[0];
    float target = random_value[0];
    float cumulative = 0.0;

    // Linear search for cumulative probability
    for (uint i = 0; i < vocab; i++) {
        cumulative += float(probs[i]);
        if (target < cumulative) {
            sampled_token[0] = i;
            return;
        }
    }

    // Fallback to last token
    sampled_token[0] = vocab - 1;
}

// ==================== f32 versions ====================

kernel void temperature_softmax_f32(
    device const float* logits [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device const float* temperature [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vocab = vocab_size[0];
    float temp = temperature[0];

    if (gid >= vocab) return;

    probs[gid] = logits[gid] / temp;
}

kernel void find_max_f32(
    device const float* logits [[buffer(0)]],
    device float* max_val [[buffer(1)]],
    device const uint* vocab_size [[buffer(2)]],
    threadgroup float* shared_max [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];

    float val = (gid < vocab) ? logits[gid] : -INFINITY;
    shared_max[lid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        max_val[0] = shared_max[0];
    }
}

kernel void softmax_normalize_f32(
    device float* probs [[buffer(0)]],
    device const float* max_val [[buffer(1)]],
    device float* sum_exp [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    threadgroup float* shared_sum [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];
    float max_logit = max_val[0];

    float exp_val = 0.0;
    if (gid < vocab) {
        exp_val = exp(probs[gid] - max_logit);
        probs[gid] = exp_val;
    }
    shared_sum[lid] = exp_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        sum_exp[0] = shared_sum[0];
    }
}

kernel void divide_by_sum_f32(
    device float* probs [[buffer(0)]],
    device const float* sum_exp [[buffer(1)]],
    device const uint* vocab_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vocab = vocab_size[0];
    if (gid >= vocab) return;

    probs[gid] /= sum_exp[0];
}

kernel void cumulative_sample_f32(
    device const float* probs [[buffer(0)]],
    device uint* sampled_token [[buffer(1)]],
    device const float* random_value [[buffer(2)]],
    device const uint* vocab_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return;

    uint vocab = vocab_size[0];
    float target = random_value[0];
    float cumulative = 0.0;

    for (uint i = 0; i < vocab; i++) {
        cumulative += probs[i];
        if (target < cumulative) {
            sampled_token[0] = i;
            return;
        }
    }

    sampled_token[0] = vocab - 1;
}
