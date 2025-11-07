#include <metal_stdlib>
using namespace metal;

/// Gather operation kernel
/// Gathers values from input tensor along a specified dimension using indices
kernel void gather_f16(
    device const half* input [[buffer(0)]],
    device const half* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* input_strides [[buffer(3)]],
    device const half* output_strides [[buffer(4)]],
    device const half* input_dims [[buffer(5)]],
    device const half* output_dims [[buffer(6)]],
    device const half* dim_ptr [[buffer(7)]],
    device const half* ndim_ptr [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dim = uint(dim_ptr[0]);
    uint ndim = uint(ndim_ptr[0]);

    // Convert flat output index to multi-dimensional coordinates
    uint coords[8]; // Support up to 8 dimensions
    uint remainder = gid;

    for (uint i = 0; i < ndim; ++i) {
        uint stride = uint(output_strides[i]);
        coords[i] = remainder / stride;
        remainder = remainder % stride;
    }

    // Get index value at this position
    uint index_value = uint(indices[gid]);

    // Replace coordinate at dim with index value
    coords[dim] = index_value;

    // Calculate flat index in input
    uint input_idx = 0;
    for (uint i = 0; i < ndim; ++i) {
        input_idx += coords[i] * uint(input_strides[i]);
    }

    output[gid] = input[input_idx];
}

/// Scatter operation kernel
/// Scatters values from src tensor into output along a specified dimension using indices
kernel void scatter_f16(
    device const half* input [[buffer(0)]],
    device const half* indices [[buffer(1)]],
    device const half* src [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const half* input_strides [[buffer(4)]],
    device const half* src_strides [[buffer(5)]],
    device const half* input_dims [[buffer(6)]],
    device const half* src_dims [[buffer(7)]],
    device const half* dim_ptr [[buffer(8)]],
    device const half* ndim_ptr [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    // First, copy input to output
    if (gid == 0) {
        uint input_size = 1;
        uint ndim = uint(ndim_ptr[0]);
        for (uint i = 0; i < ndim; ++i) {
            input_size *= uint(input_dims[i]);
        }
        for (uint i = 0; i < input_size; ++i) {
            output[i] = input[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    uint dim = uint(dim_ptr[0]);
    uint ndim = uint(ndim_ptr[0]);

    // Convert flat src index to multi-dimensional coordinates
    uint coords[8]; // Support up to 8 dimensions
    uint remainder = gid;

    for (uint i = 0; i < ndim; ++i) {
        uint stride = uint(src_strides[i]);
        coords[i] = remainder / stride;
        remainder = remainder % stride;
    }

    // Get index value at this position
    uint index_value = uint(indices[gid]);

    // Replace coordinate at dim with index value
    coords[dim] = index_value;

    // Calculate flat index in output
    uint output_idx = 0;
    for (uint i = 0; i < ndim; ++i) {
        output_idx += coords[i] * uint(input_strides[i]);
    }

    output[output_idx] = src[gid];
}

// F32 version
kernel void gather_f32(
    device const float* input [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* input_strides [[buffer(3)]],
    device const float* output_strides [[buffer(4)]],
    device const float* input_dims [[buffer(5)]],
    device const float* output_dims [[buffer(6)]],
    device const float* dim_ptr [[buffer(7)]],
    device const float* ndim_ptr [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dim = uint(dim_ptr[0]);
    uint ndim = uint(ndim_ptr[0]);

    // Convert flat output index to multi-dimensional coordinates
    uint coords[8]; // Support up to 8 dimensions
    uint remainder = gid;

    for (uint i = 0; i < ndim; ++i) {
        uint stride = uint(output_strides[i]);
        coords[i] = remainder / stride;
        remainder = remainder % stride;
    }

    // Get index value at this position
    uint index_value = uint(indices[gid]);

    // Replace coordinate at dim with index value
    coords[dim] = index_value;

    // Calculate flat index in input
    uint input_idx = 0;
    for (uint i = 0; i < ndim; ++i) {
        input_idx += coords[i] * uint(input_strides[i]);
    }

    output[gid] = input[input_idx];
}

// F32 version
kernel void scatter_f32(
    device const float* input [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device const float* src [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* input_strides [[buffer(4)]],
    device const float* src_strides [[buffer(5)]],
    device const float* input_dims [[buffer(6)]],
    device const float* src_dims [[buffer(7)]],
    device const float* dim_ptr [[buffer(8)]],
    device const float* ndim_ptr [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    // First, copy input to output
    if (gid == 0) {
        uint input_size = 1;
        uint ndim = uint(ndim_ptr[0]);
        for (uint i = 0; i < ndim; ++i) {
            input_size *= uint(input_dims[i]);
        }
        for (uint i = 0; i < input_size; ++i) {
            output[i] = input[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    uint dim = uint(dim_ptr[0]);
    uint ndim = uint(ndim_ptr[0]);

    // Convert flat src index to multi-dimensional coordinates
    uint coords[8]; // Support up to 8 dimensions
    uint remainder = gid;

    for (uint i = 0; i < ndim; ++i) {
        uint stride = uint(src_strides[i]);
        coords[i] = remainder / stride;
        remainder = remainder % stride;
    }

    // Get index value at this position
    uint index_value = uint(indices[gid]);

    // Replace coordinate at dim with index value
    coords[dim] = index_value;

    // Calculate flat index in output
    uint output_idx = 0;
    for (uint i = 0; i < ndim; ++i) {
        output_idx += coords[i] * uint(input_strides[i]);
    }

    output[output_idx] = src[gid];
}
