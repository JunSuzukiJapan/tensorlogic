//! Element-wise operations for f16 tensors
//! All operations use half precision (f16) for Apple Silicon optimization

#include <metal_stdlib>
using namespace metal;

/// Element-wise addition: C[i] = A[i] + B[i]
kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}

/// Element-wise subtraction: C[i] = A[i] - B[i]
kernel void sub_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - b[index];
}

/// Element-wise multiplication: C[i] = A[i] * B[i]
kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}

/// Element-wise division: C[i] = A[i] / B[i]
kernel void div_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / b[index];
}

/// Scalar addition: C[i] = A[i] + scalar
kernel void add_scalar_f16(
    device const half* a [[buffer(0)]],
    device const half* scalar [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + scalar[0];
}

/// Scalar multiplication: C[i] = A[i] * scalar
kernel void mul_scalar_f16(
    device const half* a [[buffer(0)]],
    device const half* scalar [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * scalar[0];
}

/// Negation: C[i] = -A[i]
kernel void neg_f16(
    device const half* a [[buffer(0)]],
    device half* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = -a[index];
}

/// Absolute value: C[i] = |A[i]|
kernel void abs_f16(
    device const half* a [[buffer(0)]],
    device half* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = abs(a[index]);
}

/// Sign function: C[i] = sign(A[i])
kernel void sign_f16(
    device const half* a [[buffer(0)]],
    device half* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    half x = a[index];
    result[index] = (x > half(0.0)) ? half(1.0) : ((x < half(0.0)) ? half(-1.0) : half(0.0));
}

/// Element-wise maximum: C[i] = max(A[i], B[i])
kernel void max_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = max(a[index], b[index]);
}

/// Element-wise minimum: C[i] = min(A[i], B[i])
kernel void min_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = min(a[index], b[index]);
}

/// Clamp: C[i] = clamp(A[i], min_val, max_val)
kernel void clamp_f16(
    device const half* a [[buffer(0)]],
    device const half* min_val [[buffer(1)]],
    device const half* max_val [[buffer(2)]],
    device half* result [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = clamp(a[index], min_val[0], max_val[0]);
}

/// Fill with constant value
kernel void fill_f16(
    device half* result [[buffer(0)]],
    device const half* value [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = value[0];
}

/// ReLU activation: f(x) = max(0, x)
kernel void relu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = max(input[index], half(0.0));
}

/// GELU activation (approximation): f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    half x = input[index];
    half sqrt_2_over_pi = half(0.7978845608);  // sqrt(2/π)
    half coeff = half(0.044715);
    half x3 = x * x * x;
    half inner = sqrt_2_over_pi * (x + coeff * x3);
    output[index] = half(0.5) * x * (half(1.0) + tanh(inner));
}

/// Exponential: f(x) = e^x
kernel void exp_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = exp(input[index]);
}

/// Natural logarithm: f(x) = log(x)
kernel void log_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = log(input[index]);
}

/// Square root: f(x) = sqrt(x)
kernel void sqrt_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = sqrt(input[index]);
}

/// Power: f(x) = x^exponent
kernel void pow_f16(
    device const half* input [[buffer(0)]],
    device const half* exponent [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = pow(input[index], exponent[0]);
}

/// Sine: f(x) = sin(x)
kernel void sin_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = sin(input[index]);
}

/// Cosine: f(x) = cos(x)
kernel void cos_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = cos(input[index]);
}

/// Tangent: f(x) = tan(x)
kernel void tan_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = tan(input[index]);
}

/// Sigmoid: f(x) = 1 / (1 + exp(-x))
kernel void sigmoid_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = half(1.0) / (half(1.0) + exp(-input[index]));
}

/// Hyperbolic tangent: f(x) = tanh(x)
kernel void tanh_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = tanh(input[index]);
}

// F32 version
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}

// F32 version
kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - b[index];
}

// F32 version
kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}

// F32 version
kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / b[index];
}

// F32 version
kernel void add_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + scalar[0];
}

// F32 version
kernel void mul_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * scalar[0];
}

// F32 version
kernel void neg_f32(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = -a[index];
}

// F32 version
kernel void abs_f32(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = abs(a[index]);
}

// F32 version
kernel void sign_f32(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    float x = a[index];
    result[index] = (x > float(0.0)) ? float(1.0) : ((x < float(0.0)) ? float(-1.0) : float(0.0));
}

// F32 version
kernel void max_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = max(a[index], b[index]);
}

// F32 version
kernel void min_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = min(a[index], b[index]);
}

// F32 version
kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = max(input[index], float(0.0));
}

// F32 version
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    float x = input[index];
    float sqrt_2_over_pi = float(0.7978845608);  // sqrt(2/π)
    float coeff = float(0.044715);
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    output[index] = float(0.5) * x * (float(1.0) + tanh(inner));
}

// F32 version
kernel void exp_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = exp(input[index]);
}

// F32 version
kernel void log_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = log(input[index]);
}

// F32 version
kernel void sqrt_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = sqrt(input[index]);
}

// F32 version
kernel void pow_f32(
    device const float* input [[buffer(0)]],
    device const float* exponent [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = pow(input[index], exponent[0]);
}

// F32 version
kernel void sin_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = sin(input[index]);
}

// F32 version
kernel void cos_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = cos(input[index]);
}

// F32 version
kernel void tan_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = tan(input[index]);
}

// F32 version
kernel void sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = float(1.0) / (float(1.0) + exp(-input[index]));
}

// F32 version
kernel void tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = tanh(input[index]);
}
