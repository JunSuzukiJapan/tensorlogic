//! Metal GPU kernels for gradient computation (backward pass)

#include <metal_stdlib>
using namespace metal;

// ReLU backward: grad_input = grad_output * (input > 0)
kernel void relu_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * (input[id] > half(0.0) ? half(1.0) : half(0.0));
}

// GELU backward (tanh approximation)
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// d/dx gelu(x) = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d/dx(...)
kernel void gelu_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half x = input[id];

    // Constants for GELU approximation
    const half sqrt_2_over_pi = half(0.7978845608);  // sqrt(2/π)
    const half coeff = half(0.044715);

    // Compute inner expression: sqrt(2/π) * (x + 0.044715 * x³)
    half x_cubed = x * x * x;
    half inner = sqrt_2_over_pi * (x + coeff * x_cubed);

    // tanh and sech²
    half tanh_inner = tanh(inner);
    half sech_sq = half(1.0) - tanh_inner * tanh_inner;

    // Derivative of inner expression
    half d_inner_dx = sqrt_2_over_pi * (half(1.0) + half(3.0) * coeff * x * x);

    // GELU derivative
    half gelu_grad = half(0.5) * (half(1.0) + tanh_inner) +
                     half(0.5) * x * sech_sq * d_inner_dx;

    grad_input[id] = grad_output[id] * gelu_grad;
}

// Softmax backward
// For output y = softmax(x), gradient is: grad_x = y * (grad_y - sum(y * grad_y))
// This kernel computes element-wise multiplication, reduction is separate
kernel void softmax_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* output [[buffer(1)]],
    device const half* sum_term [[buffer(2)]],  // Pre-computed sum(y * grad_y)
    device half* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    half y = output[id];
    half grad_y = grad_output[id];
    half sum_val = sum_term[0];  // Scalar sum

    grad_input[id] = y * (grad_y - sum_val);
}

// Element-wise multiplication (for gradient computation)
kernel void mul_grad_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* other_input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * other_input[id];
}

// Element-wise division gradient (for numerator)
// d/dx (x / y) = 1 / y
kernel void div_grad_numerator_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* denominator [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / denominator[id];
}

// Element-wise division gradient (for denominator)
// d/dy (x / y) = -x / y²
kernel void div_grad_denominator_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* numerator [[buffer(1)]],
    device const half* denominator [[buffer(2)]],
    device half* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    half denom = denominator[id];
    grad_input[id] = -grad_output[id] * numerator[id] / (denom * denom);
}

// Negate gradient (for subtraction)
kernel void negate_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = -input[id];
}

// Sum reduction for broadcast gradient reduction
// Simple parallel reduction - each thread computes partial sum
kernel void sum_reduction_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup half* shared_data [[threadgroup(0)]]
) {
    // Load input into shared memory
    uint global_id = bid * 256 + tid;
    shared_data[tid] = (global_id < count) ? input[global_id] : half(0.0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && (global_id + stride) < count) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result for this threadgroup
    if (tid == 0) {
        output[bid] = shared_data[0];
    }
}

// Exp backward: d/dx exp(x) = exp(x)
kernel void exp_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* output [[buffer(1)]],  // exp(x)
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * output[id];
}

// Log backward: d/dx log(x) = 1/x
kernel void log_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / input[id];
}

// Sqrt backward: d/dx sqrt(x) = 1/(2*sqrt(x))
kernel void sqrt_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* output [[buffer(1)]],  // sqrt(x)
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / (half(2.0) * output[id]);
}

// Pow backward: d/dx x^n = n*x^(n-1)
kernel void pow_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device const half* exponent_ptr [[buffer(2)]],
    device half* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    half n = exponent_ptr[0];
    half x = input[id];
    grad_input[id] = grad_output[id] * n * pow(x, n - half(1.0));
}

// Sin backward: d/dx sin(x) = cos(x)
kernel void sin_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * cos(input[id]);
}

// Cos backward: d/dx cos(x) = -sin(x)
kernel void cos_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * (-sin(input[id]));
}

// Sigmoid backward: d/dx σ(x) = σ(x)*(1-σ(x))
kernel void sigmoid_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* output [[buffer(1)]],  // σ(x)
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half sigmoid = output[id];
    grad_input[id] = grad_output[id] * sigmoid * (half(1.0) - sigmoid);
}

// Tanh backward: d/dx tanh(x) = 1 - tanh²(x)
kernel void tanh_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* output [[buffer(1)]],  // tanh(x)
    device half* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half tanh_val = output[id];
    grad_input[id] = grad_output[id] * (half(1.0) - tanh_val * tanh_val);
}

// F32 version
kernel void relu_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * (input[id] > float(0.0) ? float(1.0) : float(0.0));
}

// F32 version
kernel void gelu_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];

    // Constants for GELU approximation
    const float sqrt_2_over_pi = float(0.7978845608);  // sqrt(2/π)
    const float coeff = float(0.044715);

    // Compute inner expression: sqrt(2/π) * (x + 0.044715 * x³)
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);

    // tanh and sech²
    float tanh_inner = tanh(inner);
    float sech_sq = float(1.0) - tanh_inner * tanh_inner;

    // Derivative of inner expression
    float d_inner_dx = sqrt_2_over_pi * (float(1.0) + float(3.0) * coeff * x * x);

    // GELU derivative
    float gelu_grad = float(0.5) * (float(1.0) + tanh_inner) +
                     float(0.5) * x * sech_sq * d_inner_dx;

    grad_input[id] = grad_output[id] * gelu_grad;
}

// F32 version
kernel void softmax_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device const float* sum_term [[buffer(2)]],  // Pre-computed sum(y * grad_y)
    device float* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float y = output[id];
    float grad_y = grad_output[id];
    float sum_val = sum_term[0];  // Scalar sum

    grad_input[id] = y * (grad_y - sum_val);
}

// F32 version
kernel void mul_grad_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* other_input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * other_input[id];
}

// F32 version
kernel void div_grad_numerator_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* denominator [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / denominator[id];
}

// F32 version
kernel void div_grad_denominator_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* numerator [[buffer(1)]],
    device const float* denominator [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float denom = denominator[id];
    grad_input[id] = -grad_output[id] * numerator[id] / (denom * denom);
}

// F32 version
kernel void negate_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = -input[id];
}

// F32 version
kernel void sum_reduction_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup half* shared_data [[threadgroup(0)]]
) {
    // Load input into shared memory
    uint global_id = bid * 256 + tid;
    shared_data[tid] = (global_id < count) ? input[global_id] : float(0.0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && (global_id + stride) < count) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result for this threadgroup
    if (tid == 0) {
        output[bid] = shared_data[0];
    }
}

// F32 version
kernel void exp_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],  // exp(x)
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * output[id];
}

// F32 version
kernel void log_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / input[id];
}

// F32 version
kernel void sqrt_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],  // sqrt(x)
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] / (float(2.0) * output[id]);
}

// F32 version
kernel void pow_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* exponent_ptr [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float n = exponent_ptr[0];
    float x = input[id];
    grad_input[id] = grad_output[id] * n * pow(x, n - float(1.0));
}

// F32 version
kernel void sin_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * cos(input[id]);
}

// F32 version
kernel void cos_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    grad_input[id] = grad_output[id] * (-sin(input[id]));
}

// F32 version
kernel void sigmoid_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],  // σ(x)
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float sigmoid = output[id];
    grad_input[id] = grad_output[id] * sigmoid * (float(1.0) - sigmoid);
}

// F32 version
kernel void tanh_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],  // tanh(x)
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float tanh_val = output[id];
    grad_input[id] = grad_output[id] * (float(1.0) - tanh_val * tanh_val);
}
