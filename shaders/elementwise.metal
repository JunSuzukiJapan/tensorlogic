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
