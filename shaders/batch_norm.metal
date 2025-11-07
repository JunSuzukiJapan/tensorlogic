#include <metal_stdlib>
using namespace metal;

/// Batch Normalization forward pass
/// Input: [batch, features] or [batch, channels, height, width]
/// Normalizes across the batch dimension
kernel void batch_norm_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* gamma [[buffer(2)]],  // Scale parameter [features]
    device const half* beta [[buffer(3)]],   // Shift parameter [features]
    constant uint& batch_size [[buffer(4)]],
    constant uint& feature_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= feature_size) return;

    // Compute mean across batch dimension for this feature
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += float(input[b * feature_size + gid]);
    }
    float mean = sum / float(batch_size);

    // Compute variance across batch dimension
    float var_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float diff = float(input[b * feature_size + gid]) - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(batch_size);

    // Normalize and apply affine transformation
    float std = sqrt(variance + eps);
    float scale = float(gamma[gid]);
    float shift = float(beta[gid]);

    for (uint b = 0; b < batch_size; b++) {
        uint idx = b * feature_size + gid;
        float normalized = (float(input[idx]) - mean) / std;
        output[idx] = half(scale * normalized + shift);
    }
}

/// Batch Normalization backward pass
kernel void batch_norm_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device half* grad_input [[buffer(3)]],
    device half* grad_gamma [[buffer(4)]],
    device half* grad_beta [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& feature_size [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= feature_size) return;

    // Recompute mean and variance
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += float(input[b * feature_size + gid]);
    }
    float mean = sum / float(batch_size);

    float var_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float diff = float(input[b * feature_size + gid]) - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(batch_size);
    float std = sqrt(variance + eps);

    // Gradient w.r.t. beta (sum of grad_output)
    float grad_beta_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        grad_beta_sum += float(grad_output[b * feature_size + gid]);
    }
    grad_beta[gid] = half(grad_beta_sum);

    // Gradient w.r.t. gamma (sum of grad_output * normalized_input)
    float grad_gamma_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float normalized = (float(input[b * feature_size + gid]) - mean) / std;
        grad_gamma_sum += float(grad_output[b * feature_size + gid]) * normalized;
    }
    grad_gamma[gid] = half(grad_gamma_sum);

    // Gradient w.r.t. input
    float scale = float(gamma[gid]);
    float m = float(batch_size);

    for (uint b = 0; b < batch_size; b++) {
        uint idx = b * feature_size + gid;
        float x_hat = (float(input[idx]) - mean) / std;
        float dL_dx_hat = float(grad_output[idx]) * scale;

        // Gradient computation following batch norm backprop formula
        float term1 = dL_dx_hat;
        float term2 = grad_gamma_sum / m;
        float term3 = x_hat * grad_beta_sum / m;

        grad_input[idx] = half((term1 - term2 - term3) / std);
    }
}

// F32 version
kernel void batch_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],  // Scale parameter [features]
    device const float* beta [[buffer(3)]],   // Shift parameter [features]
    constant uint& batch_size [[buffer(4)]],
    constant uint& feature_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= feature_size) return;

    // Compute mean across batch dimension for this feature
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += float(input[b * feature_size + gid]);
    }
    float mean = sum / float(batch_size);

    // Compute variance across batch dimension
    float var_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float diff = float(input[b * feature_size + gid]) - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(batch_size);

    // Normalize and apply affine transformation
    float std = sqrt(variance + eps);
    float scale = float(gamma[gid]);
    float shift = float(beta[gid]);

    for (uint b = 0; b < batch_size; b++) {
        uint idx = b * feature_size + gid;
        float normalized = (float(input[idx]) - mean) / std;
        output[idx] = float(scale * normalized + shift);
    }
}

// F32 version
kernel void batch_norm_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& feature_size [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= feature_size) return;

    // Recompute mean and variance
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += float(input[b * feature_size + gid]);
    }
    float mean = sum / float(batch_size);

    float var_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float diff = float(input[b * feature_size + gid]) - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(batch_size);
    float std = sqrt(variance + eps);

    // Gradient w.r.t. beta (sum of grad_output)
    float grad_beta_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        grad_beta_sum += float(grad_output[b * feature_size + gid]);
    }
    grad_beta[gid] = float(grad_beta_sum);

    // Gradient w.r.t. gamma (sum of grad_output * normalized_input)
    float grad_gamma_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float normalized = (float(input[b * feature_size + gid]) - mean) / std;
        grad_gamma_sum += float(grad_output[b * feature_size + gid]) * normalized;
    }
    grad_gamma[gid] = float(grad_gamma_sum);

    // Gradient w.r.t. input
    float scale = float(gamma[gid]);
    float m = float(batch_size);

    for (uint b = 0; b < batch_size; b++) {
        uint idx = b * feature_size + gid;
        float x_hat = (float(input[idx]) - mean) / std;
        float dL_dx_hat = float(grad_output[idx]) * scale;

        // Gradient computation following batch norm backprop formula
        float term1 = dL_dx_hat;
        float term2 = grad_gamma_sum / m;
        float term3 = x_hat * grad_beta_sum / m;

        grad_input[idx] = float((term1 - term2 - term3) / std);
    }
}
