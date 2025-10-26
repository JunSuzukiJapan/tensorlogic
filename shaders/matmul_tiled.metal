//! Tiled Matrix Multiplication with Threadgroup Memory
//!
//! Uses shared memory (threadgroup memory) to cache tiles of the input matrices,
//! dramatically reducing global memory access and improving performance.
//!
//! Expected performance improvement: 1.5-2x GFLOPS over naive implementation
//!
//! Algorithm:
//! 1. Each threadgroup loads a tile of A and B into shared memory
//! 2. All threads in the threadgroup compute their partial results using shared memory
//! 3. Repeat for all tiles along the K dimension
//! 4. Write final result to global memory

#include <metal_stdlib>
using namespace metal;

/// Tiled matrix multiplication with threadgroup memory optimization
///
/// Uses 16x16 tiles in shared memory for both A and B matrices.
/// Each threadgroup computes a 16x16 block of the output matrix C.
///
/// Memory hierarchy:
/// - Global memory (slow): Input A, B and output C
/// - Threadgroup memory (fast): Cached tiles of A and B
/// - Thread registers (fastest): Accumulator for partial sum
///
/// Performance characteristics:
/// - Threadgroup memory bandwidth: ~400 GB/s (on Apple Silicon)
/// - Global memory bandwidth: ~90 GB/s (on Apple Silicon)
/// - Speedup: 4-5x reduction in global memory traffic
///
/// C[M,N] = A[M,K] @ B[K,N]
kernel void matmul_tiled_f16(
    device const half* A [[buffer(0)]],      // Input matrix A [M, K]
    device const half* B [[buffer(1)]],      // Input matrix B [K, N]
    device half* C [[buffer(2)]],            // Output matrix C [M, N]
    constant uint& M [[buffer(3)]],          // Number of rows in A and C
    constant uint& N [[buffer(4)]],          // Number of columns in B and C
    constant uint& K [[buffer(5)]],          // Shared dimension
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    // Tile size (16x16 for optimal occupancy on Apple Silicon)
    constexpr uint TILE_SIZE = 16;

    // Threadgroup shared memory for caching tiles
    // These are allocated in fast on-chip memory
    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    // Thread's position within threadgroup (0-15, 0-15)
    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    // Thread's global position in output matrix C
    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    // Accumulator for this thread's output element
    // Use float (f32) for accumulation to prevent precision loss in large matrices
    float sum = 0.0f;

    // Number of tiles along K dimension
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles along K dimension
    for (uint tile = 0; tile < num_tiles; tile++) {
        // Global K position for this tile
        uint k_offset = tile * TILE_SIZE;

        // Load tile of A into shared memory
        // Each thread loads one element
        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0h;  // Padding for out-of-bounds
        }

        // Load tile of B into shared memory
        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;  // Padding for out-of-bounds
        }

        // Synchronize to ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product using shared memory
        // This is the hot loop - all memory accesses are to fast threadgroup memory
        // Convert to f32 for accumulation to prevent precision loss
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[ty][k]) * float(B_tile[k][tx]);
        }

        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final result to global memory
    // Convert accumulated f32 result back to f16 for storage
    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

/// Tiled matrix multiplication with 32x32 tiles for larger matrices
///
/// Uses larger tiles for matrices where M, N, K > 512.
/// Better cache utilization for large matrices.
///
/// C[M,N] = A[M,K] @ B[K,N]
kernel void matmul_tiled_32x32_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 32;

    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    // Use float (f32) for accumulation to prevent precision loss
    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0h;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to f32 for accumulation
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[ty][k]) * float(B_tile[k][tx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert back to f16 for storage
    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

/// Tiled matrix multiplication with bias addition
///
/// Combines tiled matmul with bias addition in a single kernel.
/// Saves one kernel launch overhead.
///
/// C[M,N] = A[M,K] @ B[K,N] + bias[N]
kernel void matmul_tiled_bias_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    // Use float (f32) for accumulation to prevent precision loss
    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0h;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to f32 for accumulation
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[ty][k]) * float(B_tile[k][tx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Add bias and convert to f16
    if (row < M && col < N) {
        C[row * N + col] = half(sum + float(bias[col]));
    }
}

/// Tiled matrix multiplication with activation function
///
/// Combines tiled matmul with activation (ReLU or GELU).
/// Maximum performance for neural network forward pass.
///
/// C[M,N] = activation(A[M,K] @ B[K,N] + bias[N])
kernel void matmul_tiled_activation_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& activation [[buffer(7)]],  // 0=none, 1=relu, 2=gelu
    constant bool& has_bias [[buffer(8)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    half sum = 0.0h;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0h;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        // Add bias if present
        if (has_bias) {
            sum += bias[col];
        }

        // Apply activation
        half result;
        if (activation == 1) {
            // ReLU
            result = max(sum, half(0.0));
        } else if (activation == 2) {
            // GELU approximation
            half x = sum;
            half x3 = x * x * x;
            half inner = half(0.7978845608) * (x + half(0.044715) * x3);
            result = half(0.5) * x * (half(1.0) + tanh(inner));
        } else {
            // No activation
            result = sum;
        }

        C[row * N + col] = result;
    }
}
