#include <metal_stdlib>
using namespace metal;

// Rotary Position Embedding (RoPE) - NeoX style for LLaMA
//
// For each position `pos` and dimension pair `(2i, 2i+1)`:
//   freq = 1.0 / (rope_base^(2i/head_dim))
//   theta = pos * freq
//
//   out[2i]   = in[2i]   * cos(theta) - in[2i+1] * sin(theta)
//   out[2i+1] = in[2i]   * sin(theta) + in[2i+1] * cos(theta)

kernel void rope_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint4 &params [[buffer(2)]],  // [seq_len, n_heads, head_dim, rope_base]
    uint tid [[thread_position_in_grid]]
) {
    const uint seq_len = params.x;
    const uint n_heads = params.y;
    const uint head_dim = params.z;
    const float rope_base = float(params.w);

    const uint total_per_seq = n_heads * head_dim;
    const uint total_elements = seq_len * total_per_seq;

    if (tid >= total_elements) {
        return;
    }

    // Decode position: which sequence position, head, and dimension
    const uint pos = tid / total_per_seq;
    const uint remainder = tid % total_per_seq;
    const uint head = remainder / head_dim;
    const uint dim = remainder % head_dim;

    // RoPE only operates on dimension pairs (even indices)
    // If odd dimension, copy from corresponding even dimension's rotation
    if (dim % 2 == 1) {
        // This will be computed by the even thread, just copy for now
        // (We'll compute both in the even thread)
        return;
    }

    // We're at an even dimension (dim = 2i)
    const uint dim_pair = dim / 2;
    const uint idx0 = tid;              // Index for dimension 2i
    const uint idx1 = tid + 1;          // Index for dimension 2i+1

    // Calculate frequency for this dimension pair
    // freq = 1.0 / (rope_base^(2i/head_dim))
    const float exponent = float(dim) / float(head_dim);  // 2i/head_dim
    const float freq = 1.0 / pow(rope_base, exponent);

    // Calculate angle
    const float theta = float(pos) * freq;
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);

    // Read input pair
    const float x0 = float(input[idx0]);
    const float x1 = float(input[idx1]);

    // Apply rotation
    const float rotated_x0 = x0 * cos_theta - x1 * sin_theta;
    const float rotated_x1 = x0 * sin_theta + x1 * cos_theta;

    // Write output
    output[idx0] = half(rotated_x0);
    output[idx1] = half(rotated_x1);
}
