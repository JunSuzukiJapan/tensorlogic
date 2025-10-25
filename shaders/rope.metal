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
    constant uint *params [[buffer(2)]],  // [seq_len, n_heads, head_dim, rope_base, position_offset]
    uint tid [[thread_position_in_grid]]
) {
    const uint seq_len = params[0];
    const uint n_heads = params[1];
    const uint head_dim = params[2];
    const float rope_base = float(params[3]);
    const uint position_offset = params[4];

    const uint total_per_seq = n_heads * head_dim;
    const uint total_elements = seq_len * total_per_seq;

    if (tid >= total_elements) {
        return;
    }

    // Calculate linear index components
    const uint relative_pos = tid / total_per_seq;
    const uint pos = relative_pos + position_offset;  // Add position offset for KV cache
    const uint remainder = tid % total_per_seq;
    const uint head = remainder / head_dim;
    const uint dim = remainder % head_dim;

    // RoPE operates on dimension pairs (2i, 2i+1)
    // Each thread handles one element, but we calculate based on pairs
    const uint dim_pair_idx = dim / 2;  // Which pair (0, 1, 2, ...)
    const bool is_even = (dim % 2 == 0);

    // Calculate the base index for this dimension pair in the actual tensor
    // Structure: [seq_len, n_heads, head_dim]
    const uint pos_head_base = relative_pos * total_per_seq + head * head_dim;
    const uint dim_pair_start = (dim / 2) * 2;  // Round dim down to even (0,1→0, 2,3→2)
    const uint idx0 = pos_head_base + dim_pair_start;
    const uint idx1 = pos_head_base + dim_pair_start + 1;

    // Calculate frequency for this dimension pair
    // freq = 1.0 / (rope_base^(2*dim_pair_idx/head_dim))
    const float exponent = float(2 * dim_pair_idx) / float(head_dim);
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

    // Each thread writes its own output
    if (is_even) {
        output[tid] = half(rotated_x0);
    } else {
        output[tid] = half(rotated_x1);
    }
}
