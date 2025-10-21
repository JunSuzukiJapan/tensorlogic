#include <metal_stdlib>
using namespace metal;

/// Dropout forward pass
/// Randomly zeros out elements with probability p
/// Scales remaining elements by 1/(1-p) to maintain expected value
kernel void dropout_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const float* random [[buffer(2)]],  // Random values in [0, 1]
    constant float& drop_prob [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    float scale = 1.0f / (1.0f - drop_prob);

    if (random[gid] < drop_prob) {
        // Drop this element
        output[gid] = half(0.0f);
    } else {
        // Keep and scale
        output[gid] = half(float(input[gid]) * scale);
    }
}

/// Dropout backward pass
kernel void dropout_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const float* random [[buffer(1)]],  // Same random values from forward
    device half* grad_input [[buffer(2)]],
    constant float& drop_prob [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    float scale = 1.0f / (1.0f - drop_prob);

    if (random[gid] < drop_prob) {
        // This element was dropped
        grad_input[gid] = half(0.0f);
    } else {
        // Gradient passes through with scaling
        grad_input[gid] = half(float(grad_output[gid]) * scale);
    }
}
