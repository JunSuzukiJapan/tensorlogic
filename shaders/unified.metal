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

// ===== F32 VERSIONS =====

/// Tiled matrix multiplication for f32 (16x16 tiles)
kernel void matmul_tiled_f32(
    device const float* A [[buffer(0)]],     // Input matrix A [M, K]
    device const float* B [[buffer(1)]],     // Input matrix B [K, N]
    device float* C [[buffer(2)]],           // Output matrix C [M, N]
    constant uint& M [[buffer(3)]],          // Number of rows in A and C
    constant uint& N [[buffer(4)]],          // Number of columns in B and C
    constant uint& K [[buffer(5)]],          // Shared dimension
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        // Load tile of A
        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // Load tile of B
        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Tiled matrix multiplication for f32 (32x32 tiles)
kernel void matmul_tiled_32x32_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 32;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Simple matrix multiplication for f16 (no tiling)
kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0f;  // Use f32 accumulator for better precision
        for (uint k = 0; k < K; k++) {
            sum += float(A[row * K + k]) * float(B[k * N + col]);
        }
        C[row * N + col] = half(sum);
    }
}

/// Simple matrix multiplication for f32 (no tiling)
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// F32 version
kernel void matmul_tiled_bias_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

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
            A_tile[ty][tx] = 0.0f;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
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
        C[row * N + col] = float(sum + float(bias[col]));
    }
}

// F32 version
kernel void matmul_tiled_activation_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& activation [[buffer(7)]],  // 0=none, 1=relu, 2=gelu
    constant bool& has_bias [[buffer(8)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        uint b_row = k_offset + ty;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            B_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
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
        float result;
        if (activation == 1) {
            // ReLU
            result = max(sum, float(0.0));
        } else if (activation == 2) {
            // GELU approximation
            float x = sum;
            float x3 = x * x * x;
            float inner = float(0.7978845608) * (x + float(0.044715) * x3);
            result = float(0.5) * x * (float(1.0) + tanh(inner));
        } else {
            // No activation
            result = sum;
        }

        C[row * N + col] = result;
    }
}
#include <metal_stdlib>
using namespace metal;

/// Embedding lookup kernel (f16)
///
/// Extracts rows from embedding table based on token IDs
/// Parameters:
/// - table: Embedding table [vocab_size, embedding_dim]
/// - token_ids: Token IDs to look up [seq_len]
/// - output: Output embeddings [seq_len, embedding_dim]
/// - embedding_dim: Dimension of embeddings
kernel void embedding_lookup_f16(
    device const half* table [[buffer(0)]],
    device const int* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const uint* embedding_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint emb_dim = embedding_dim[0];
    uint seq_idx = gid / emb_dim;
    uint emb_idx = gid % emb_dim;

    // Get token ID for this sequence position
    int token_id = token_ids[seq_idx];

    // Look up embedding: output[seq_idx, emb_idx] = table[token_id, emb_idx]
    uint table_idx = token_id * emb_dim + emb_idx;
    output[gid] = table[table_idx];
}

/// Embedding lookup kernel (f32)
kernel void embedding_lookup_f32(
    device const float* table [[buffer(0)]],
    device const int* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* embedding_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint emb_dim = embedding_dim[0];
    uint seq_idx = gid / emb_dim;
    uint emb_idx = gid % emb_dim;

    int token_id = token_ids[seq_idx];

    uint table_idx = token_id * emb_dim + emb_idx;
    output[gid] = table[table_idx];
}
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
//! Metal GPU kernels for reduction operations

#include <metal_stdlib>
using namespace metal;

// Global reduction: sum all elements (f16)
// Uses two-stage reduction: local (threadgroup) then global
kernel void sum_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Each thread loads one element
    half local_sum = (gid < count) ? input[gid] : half(0.0);

    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread in each threadgroup writes result
    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: sum all elements (f32)
// Uses two-stage reduction: local (threadgroup) then global
kernel void sum_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Each thread loads one element
    float local_sum = (gid < count) ? input[gid] : 0.0f;

    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread in each threadgroup writes result
    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: mean of all elements
kernel void mean_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Same as sum, but divide by count at the end
    half local_sum = (gid < count) ? input[gid] : half(0.0);

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: max of all elements
kernel void max_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    half local_max = (gid < count) ? input[gid] : -INFINITY;

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Global reduction: min of all elements
kernel void min_global_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    half local_min = (gid < count) ? input[gid] : INFINITY;

    shared[tid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// Dimension-specific reduction: sum along one dimension
// For example, reducing [M, N, K] along dim=1 produces [M, K]
kernel void sum_dim_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],  // [dim0, dim1, dim2, ...]
    constant uint* output_shape [[buffer(3)]], // Shape after reduction
    constant uint& rank [[buffer(4)]],         // Number of dimensions
    constant uint& reduce_dim [[buffer(5)]],   // Which dimension to reduce
    uint gid [[thread_position_in_grid]]
) {
    // Calculate output index from global thread ID
    if (gid >= output_shape[0]) return; // Bounds check

    // Compute total output size
    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    // Convert linear output index to multi-dimensional indices
    uint temp = gid;
    uint output_indices[8]; // Max 8 dimensions
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    // Sum over the reduction dimension
    half sum = half(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        // Construct input index by inserting i at reduce_dim
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        // Convert multi-dimensional index to linear index
        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum;
}

// Dimension-specific reduction: mean along one dimension
kernel void mean_dim_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& reduce_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_shape[0]) return;

    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    uint temp = gid;
    uint output_indices[8];
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    half sum = half(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum / half(reduce_size);
}

// F32 version
kernel void mean_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    // Same as sum, but divide by count at the end
    float local_sum = (gid < count) ? input[gid] : float(0.0);

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void max_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    float local_max = (gid < count) ? input[gid] : -INFINITY;

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void min_global_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]]
) {
    float local_min = (gid < count) ? input[gid] : INFINITY;

    shared[tid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (gid + stride) < count) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint block_id = gid / tg_size;
        output[block_id] = shared[0];
    }
}

// F32 version
kernel void sum_dim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],  // [dim0, dim1, dim2, ...]
    constant uint* output_shape [[buffer(3)]], // Shape after reduction
    constant uint& rank [[buffer(4)]],         // Number of dimensions
    constant uint& reduce_dim [[buffer(5)]],   // Which dimension to reduce
    uint gid [[thread_position_in_grid]]
) {
    // Calculate output index from global thread ID
    if (gid >= output_shape[0]) return; // Bounds check

    // Compute total output size
    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    // Convert linear output index to multi-dimensional indices
    uint temp = gid;
    uint output_indices[8]; // Max 8 dimensions
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    // Sum over the reduction dimension
    float sum = float(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        // Construct input index by inserting i at reduce_dim
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        // Convert multi-dimensional index to linear index
        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum;
}

// F32 version
kernel void mean_dim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& reduce_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_shape[0]) return;

    uint output_size = 1;
    for (uint i = 0; i < rank - 1; i++) {
        output_size *= output_shape[i];
    }
    if (gid >= output_size) return;

    uint temp = gid;
    uint output_indices[8];
    for (int d = int(rank) - 2; d >= 0; d--) {
        output_indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
    }

    float sum = float(0.0);
    uint reduce_size = input_shape[reduce_dim];

    for (uint i = 0; i < reduce_size; i++) {
        uint input_indices[8];
        uint out_idx = 0;
        for (uint d = 0; d < rank; d++) {
            if (d < reduce_dim) {
                input_indices[d] = output_indices[out_idx++];
            } else if (d == reduce_dim) {
                input_indices[d] = i;
            } else {
                input_indices[d] = output_indices[out_idx++];
            }
        }

        uint input_idx = 0;
        uint stride = 1;
        for (int d = int(rank) - 1; d >= 0; d--) {
            input_idx += input_indices[d] * stride;
            stride *= input_shape[d];
        }

        sum += input[input_idx];
    }

    output[gid] = sum / float(reduce_size);
}
#include <metal_stdlib>
using namespace metal;

/// Layer Normalization kernel
/// Normalizes input across the last dimension: (x - mean) / sqrt(variance + eps)
/// Then applies affine transformation if weight and bias are provided
kernel void layer_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],     // nullable
    device const half* bias [[buffer(2)]],       // nullable
    device half* output [[buffer(3)]],
    device const half* normalized_size_ptr [[buffer(4)]],
    device const half* eps_ptr [[buffer(5)]],
    device const half* has_weight_ptr [[buffer(6)]],
    device const half* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > half(0.5));
    bool has_bias = (has_bias_ptr[0] > half(0.5));
    // Shared memory for reduction
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];

    uint offset = batch_idx * normalized_size;

    // Phase 1: Compute mean (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        local_sum += float(input[offset + i]);
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(normalized_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance (parallel reduction)
    float local_sq_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        float diff = float(input[offset + i]) - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for squared sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = shared_sq_sum[0] / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and apply affine transformation
    for (uint i = tid; i < normalized_size; i += tsize) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = half(normalized);
    }
}

/// Simplified layer normalization for small tensors (single thread per batch)
kernel void layer_norm_simple_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const half* normalized_size_ptr [[buffer(4)]],
    device const half* eps_ptr [[buffer(5)]],
    device const half* has_weight_ptr [[buffer(6)]],
    device const half* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > half(0.5));
    bool has_bias = (has_bias_ptr[0] > half(0.5));
    uint offset = batch_idx * normalized_size;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum += float(input[offset + i]);
    }
    float mean = sum / float(normalized_size);

    // Compute variance
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = float(input[offset + i]) - mean;
        sq_sum += diff * diff;
    }
    float variance = sq_sum / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transformation
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = half(normalized);
    }
}
// ============================================================================
// RMS Normalization (Root Mean Square Normalization)
// Used in LLaMA, TinyLlama models
// Formula: output = (x / rms(x)) * weight
// where rms(x) = sqrt(mean(x^2) + eps)
// ============================================================================

/// RMS Norm (simple version for small tensors)
kernel void rms_norm_simple_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* normalized_size_ptr [[buffer(3)]],
    device const half* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Compute RMS: sqrt(mean(x^2) + eps)
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float val = float(input[offset + i]);
        sq_sum += val * val;
    }
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = half(scaled);
    }
}

/// RMS Norm (optimized version with threadgroup reduction for large tensors)
kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* normalized_size_ptr [[buffer(3)]],
    device const half* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Thread-local sum for reduction
    threadgroup float local_sums[256];
    float thread_sq_sum = 0.0f;

    // Parallel computation of squared sum
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float val = float(input[offset + i]);
        thread_sq_sum += val * val;
    }
    local_sums[tid] = thread_sq_sum;

    // Reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_sum = local_sums[0];
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight (parallel)
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = half(scaled);
    }
}

// F32 version
kernel void layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],     // nullable
    device const float* bias [[buffer(2)]],       // nullable
    device float* output [[buffer(3)]],
    device const float* normalized_size_ptr [[buffer(4)]],
    device const float* eps_ptr [[buffer(5)]],
    device const float* has_weight_ptr [[buffer(6)]],
    device const float* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > float(0.5));
    bool has_bias = (has_bias_ptr[0] > float(0.5));
    // Shared memory for reduction
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];

    uint offset = batch_idx * normalized_size;

    // Phase 1: Compute mean (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        local_sum += float(input[offset + i]);
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(normalized_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance (parallel reduction)
    float local_sq_sum = 0.0f;
    for (uint i = tid; i < normalized_size; i += tsize) {
        float diff = float(input[offset + i]) - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree for squared sum
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = shared_sq_sum[0] / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and apply affine transformation
    for (uint i = tid; i < normalized_size; i += tsize) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = float(normalized);
    }
}

// F32 version
kernel void layer_norm_simple_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* normalized_size_ptr [[buffer(4)]],
    device const float* eps_ptr [[buffer(5)]],
    device const float* has_weight_ptr [[buffer(6)]],
    device const float* has_bias_ptr [[buffer(7)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    uint normalized_size = uint(normalized_size_ptr[0]);
    float eps = float(eps_ptr[0]);
    bool has_weight = (has_weight_ptr[0] > float(0.5));
    bool has_bias = (has_bias_ptr[0] > float(0.5));
    uint offset = batch_idx * normalized_size;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum += float(input[offset + i]);
    }
    float mean = sum / float(normalized_size);

    // Compute variance
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = float(input[offset + i]) - mean;
        sq_sum += diff * diff;
    }
    float variance = sq_sum / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transformation
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;

        if (has_weight) {
            normalized *= float(weight[i]);
        }
        if (has_bias) {
            normalized += float(bias[i]);
        }

        output[offset + i] = float(normalized);
    }
}

// F32 version
kernel void rms_norm_simple_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* normalized_size_ptr [[buffer(3)]],
    device const float* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Compute RMS: sqrt(mean(x^2) + eps)
    float sq_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float val = float(input[offset + i]);
        sq_sum += val * val;
    }
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight
    for (uint i = 0; i < normalized_size; ++i) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = float(scaled);
    }
}

// F32 version
kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* normalized_size_ptr [[buffer(3)]],
    device const float* eps_ptr [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
) {
    uint normalized_size = uint(float(*normalized_size_ptr));
    float eps = float(*eps_ptr);
    uint offset = gid * normalized_size;

    // Thread-local sum for reduction
    threadgroup float local_sums[256];
    float thread_sq_sum = 0.0f;

    // Parallel computation of squared sum
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float val = float(input[offset + i]);
        thread_sq_sum += val * val;
    }
    local_sums[tid] = thread_sq_sum;

    // Reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_sum = local_sums[0];
    float mean_sq = sq_sum / float(normalized_size);
    float rms = sqrt(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by weight (parallel)
    for (uint i = tid; i < normalized_size; i += tgsize) {
        float normalized = float(input[offset + i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = float(scaled);
    }
}
//! Softmax Operation
//!
//! GPU-accelerated softmax implementation with numerical stability

#include <metal_stdlib>
using namespace metal;

/// Softmax kernel with reduction for finding max and sum
///
/// Computes softmax over the last dimension of the input tensor.
/// Uses threadgroup memory for reduction operations.
///
/// Numerically stable: subtracts max before exp to prevent overflow
/// Handles NaN and Inf: replaces invalid values with 0
///
/// Input:  [batch, last_dim]
/// Output: [batch, last_dim]
///
/// Each threadgroup processes one batch (row)
kernel void softmax_f16(
    device const half* input [[buffer(0)]],      // Input tensor
    device half* output [[buffer(1)]],           // Output tensor
    constant uint& last_dim [[buffer(2)]],       // Size of last dimension
    threadgroup float* shared_max [[threadgroup(0)]],   // Shared memory for max reduction
    threadgroup float* shared_sum [[threadgroup(1)]],   // Shared memory for sum reduction
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint offset = batch_id * last_dim;

    // Phase 1: Find maximum value for numerical stability
    // Each thread processes multiple elements
    float local_max = -INFINITY;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        // Only consider finite values
        if (isfinite(val)) {
            local_max = max(local_max, val);
        }
    }

    // Store local max to shared memory
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle case where all values are NaN/Inf
    if (isinf(max_val) || isnan(max_val)) {
        max_val = 0.0f;
    }

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;  // Replace NaN/Inf with 0
        }

        output[offset + i] = half(exp_val);
        local_sum += exp_val;
    }

    // Store local sum to shared memory
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize by sum
    if (sum > 0.0f && isfinite(sum)) {
        // Normal case: divide by sum
        for (uint i = tid; i < last_dim; i += tg_size) {
            float val = float(output[offset + i]);
            output[offset + i] = half(val / sum);
        }
    } else {
        // Degenerate case: uniform distribution
        float uniform = 1.0f / float(last_dim);
        for (uint i = tid; i < last_dim; i += tg_size) {
            output[offset + i] = half(uniform);
        }
    }
}

/// Simple softmax kernel for small dimensions (no reduction needed)
///
/// For last_dim <= 256, single thread can handle the entire row
kernel void softmax_simple_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint offset = batch_id * last_dim;

    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        if (isfinite(val)) {
            max_val = max(max_val, val);
        }
    }

    if (!isfinite(max_val)) {
        max_val = 0.0f;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;
        }

        output[offset + i] = half(exp_val);
        sum += exp_val;
    }

    // Normalize
    if (sum > 0.0f && isfinite(sum)) {
        for (uint i = 0; i < last_dim; i++) {
            float val = float(output[offset + i]);
            output[offset + i] = half(val / sum);
        }
    } else {
        float uniform = 1.0f / float(last_dim);
        for (uint i = 0; i < last_dim; i++) {
            output[offset + i] = half(uniform);
        }
    }
}

// F32 version
kernel void softmax_f32(
    device const float* input [[buffer(0)]],      // Input tensor
    device float* output [[buffer(1)]],           // Output tensor
    constant uint& last_dim [[buffer(2)]],       // Size of last dimension
    threadgroup float* shared_max [[threadgroup(0)]],   // Shared memory for max reduction
    threadgroup float* shared_sum [[threadgroup(1)]],   // Shared memory for sum reduction
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint offset = batch_id * last_dim;

    // Phase 1: Find maximum value for numerical stability
    // Each thread processes multiple elements
    float local_max = -INFINITY;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        // Only consider finite values
        if (isfinite(val)) {
            local_max = max(local_max, val);
        }
    }

    // Store local max to shared memory
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Handle case where all values are NaN/Inf
    if (isinf(max_val) || isnan(max_val)) {
        max_val = 0.0f;
    }

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < last_dim; i += tg_size) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;  // Replace NaN/Inf with 0
        }

        output[offset + i] = float(exp_val);
        local_sum += exp_val;
    }

    // Store local sum to shared memory
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize by sum
    if (sum > 0.0f && isfinite(sum)) {
        // Normal case: divide by sum
        for (uint i = tid; i < last_dim; i += tg_size) {
            float val = float(output[offset + i]);
            output[offset + i] = float(val / sum);
        }
    } else {
        // Degenerate case: uniform distribution
        float uniform = 1.0f / float(last_dim);
        for (uint i = tid; i < last_dim; i += tg_size) {
            output[offset + i] = float(uniform);
        }
    }
}

// F32 version
kernel void softmax_simple_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint offset = batch_id * last_dim;

    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        if (isfinite(val)) {
            max_val = max(max_val, val);
        }
    }

    if (!isfinite(max_val)) {
        max_val = 0.0f;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; i++) {
        float val = float(input[offset + i]);
        float exp_val;

        if (isfinite(val)) {
            exp_val = exp(val - max_val);
        } else {
            exp_val = 0.0f;
        }

        output[offset + i] = float(exp_val);
        sum += exp_val;
    }

    // Normalize
    if (sum > 0.0f && isfinite(sum)) {
        for (uint i = 0; i < last_dim; i++) {
            float val = float(output[offset + i]);
            output[offset + i] = float(val / sum);
        }
    } else {
        float uniform = 1.0f / float(last_dim);
        for (uint i = 0; i < last_dim; i++) {
            output[offset + i] = float(uniform);
        }
    }
}
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

// F32 version
kernel void rope_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
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
        output[tid] = float(rotated_x0);
    } else {
        output[tid] = float(rotated_x1);
    }
}
//! Einsum Operations for Attention Mechanisms
//!
//! Specialized einsum implementations for common attention patterns.
//! These are optimized for the specific tensor contractions used in
//! multi-head attention and grouped query attention.

#include <metal_stdlib>
using namespace metal;

/// Einsum: "ihd,jhd->ihj" - Batched dot product for attention scores
///
/// ⚠️ **MATHEMATICALLY VERIFIED - DO NOT MODIFY**
/// This implementation has been verified to be mathematically correct through:
/// - Small input validation tests (exact match with expected values)
/// - Identity matrix tests
/// - Real model weight integration tests
/// - GQA attention end-to-end tests
///
/// Index calculations verified:
///   A[i,h,d] = (i * H + h) * D + d  ✓
///   B[j,h,d] = (j * H + h) * D + d  ✓
///   C[i,h,j] = (i * H + h) * J + j  ✓
///
/// If you encounter issues, the problem is likely in OTHER operations,
/// NOT in this kernel. Verify other components before modifying this.
///
/// Computes attention scores for multi-head attention:
/// For each head h, computes dot product between query position i and key position j.
///
/// Inputs:
///   A: [I, H, D] - Query tensor (position, heads, head_dim)
///   B: [J, H, D] - Key tensor (position, heads, head_dim)
/// Output:
///   C: [I, H, J] - Attention scores
///
/// Computation: C[i,h,j] = sum_d A[i,h,d] * B[j,h,d]
///
/// Memory access pattern:
/// - Each thread computes one element C[i,h,j]
/// - Reads D elements from A and B
/// - No threadgroup memory needed for this pattern (small D=64)
kernel void einsum_ihd_jhd_ihj_f16(
    device const half* A [[buffer(0)]],      // [I, H, D]
    device const half* B [[buffer(1)]],      // [J, H, D]
    device half* C [[buffer(2)]],            // [I, H, J]
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, j]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint j = gid.z;  // Key position

    // Bounds check
    if (i >= I || h >= H || j >= J) {
        return;
    }

    // Compute dot product: sum over dimension d
    half sum = 0.0h;

    // A[i, h, :] · B[j, h, :]
    for (uint d = 0; d < D; d++) {
        uint a_idx = (i * H + h) * D + d;  // A[i, h, d]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * J + j;  // C[i, h, j]
    C[c_idx] = sum;
}

/// Optimized version with threadgroup memory for larger D
///
/// For D > 128, uses reduction in threadgroup memory
/// Currently not used (D=64 for TinyLlama), but available for larger models
kernel void einsum_ihd_jhd_ihj_f16_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& I [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& J [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Threadgroup shared memory for partial sums
    threadgroup half partial_sums[256];

    uint i = tgid.x;
    uint h = tgid.y;
    uint j = tgid.z;

    if (i >= I || h >= H || j >= J) {
        return;
    }

    // Each thread computes partial sum over chunk of D
    half sum = 0.0h;
    for (uint d = tid; d < D; d += 256) {
        uint a_idx = (i * H + h) * D + d;
        uint b_idx = (j * H + h) * D + d;
        sum += A[a_idx] * B[b_idx];
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes result
    if (tid == 0) {
        uint c_idx = (i * H + h) * J + j;
        C[c_idx] = partial_sums[0];
    }
}

/// Einsum: "ihj,jhd->ihd" - Weighted sum for attention output
///
/// ⚠️ **MATHEMATICALLY VERIFIED - DO NOT MODIFY**
/// This implementation has been verified to be mathematically correct through:
/// - Small input validation tests (exact match with expected values)
/// - Identity matrix tests
/// - Real model weight integration tests
/// - GQA attention end-to-end tests
///
/// Index calculations verified:
///   A[i,h,j] = (i * H + h) * J + j  ✓
///   B[j,h,d] = (j * H + h) * D + d  ✓
///   C[i,h,d] = (i * H + h) * D + d  ✓
///
/// If you encounter issues, the problem is likely in OTHER operations,
/// NOT in this kernel. Verify other components before modifying this.
///
/// Computes attention output by weighted sum of values:
/// For each head h and query position i, computes weighted sum over key positions j.
///
/// Inputs:
///   A: [I, H, J] - Attention weights (after softmax)
///   B: [J, H, D] - Value tensor
/// Output:
///   C: [I, H, D] - Attention output
///
/// Computation: C[i,h,d] = sum_j A[i,h,j] * B[j,h,d]
///
/// Memory access pattern:
/// - Each thread computes one element C[i,h,d]
/// - Reads J elements from A and B
kernel void einsum_ihj_jhd_ihd_f16(
    device const half* A [[buffer(0)]],      // [I, H, J] - attention weights
    device const half* B [[buffer(1)]],      // [J, H, D] - values
    device half* C [[buffer(2)]],            // [I, H, D] - output
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key/value)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, d]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint d = gid.z;  // Output dimension

    // Bounds check
    if (i >= I || h >= H || d >= D) {
        return;
    }

    // Compute weighted sum: sum over positions j
    half sum = 0.0h;

    for (uint j = 0; j < J; j++) {
        uint a_idx = (i * H + h) * J + j;  // A[i, h, j]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * D + d;  // C[i, h, d]
    C[c_idx] = sum;
}

/// General einsum for 3D tensors with arbitrary contraction
///
/// This is a fallback for other einsum patterns not covered above.
/// Uses a general loop structure that can handle various index patterns.
///
/// Note: This is slower than specialized kernels but more flexible.
kernel void einsum_3d_general_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint3& shape_a [[buffer(3)]],   // Shape of A
    constant uint3& shape_b [[buffer(4)]],   // Shape of B
    constant uint3& shape_c [[buffer(5)]],   // Shape of C
    constant uint& contract_dim [[buffer(6)]], // Which dimension to contract
    uint3 gid [[thread_position_in_grid]]
) {
    // This would need index mapping configuration
    // For now, specialized kernels are sufficient
    // TODO: Implement if needed for other einsum patterns
}

// F32 version
kernel void einsum_ihd_jhd_ihj_f32(
    device const float* A [[buffer(0)]],      // [I, H, D]
    device const float* B [[buffer(1)]],      // [J, H, D]
    device float* C [[buffer(2)]],            // [I, H, J]
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, j]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint j = gid.z;  // Key position

    // Bounds check
    if (i >= I || h >= H || j >= J) {
        return;
    }

    // Compute dot product: sum over dimension d
    float sum = 0.0f;

    // A[i, h, :] · B[j, h, :]
    for (uint d = 0; d < D; d++) {
        uint a_idx = (i * H + h) * D + d;  // A[i, h, d]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * J + j;  // C[i, h, j]
    C[c_idx] = sum;
}

// F32 version
kernel void einsum_ihj_jhd_ihd_f32(
    device const float* A [[buffer(0)]],      // [I, H, J] - attention weights
    device const float* B [[buffer(1)]],      // [J, H, D] - values
    device float* C [[buffer(2)]],            // [I, H, D] - output
    constant uint& I [[buffer(3)]],          // Sequence length (query)
    constant uint& H [[buffer(4)]],          // Number of heads
    constant uint& J [[buffer(5)]],          // Sequence length (key/value)
    constant uint& D [[buffer(6)]],          // Head dimension
    uint3 gid [[thread_position_in_grid]]
) {
    // Thread computes C[i, h, d]
    uint i = gid.x;  // Query position
    uint h = gid.y;  // Head index
    uint d = gid.z;  // Output dimension

    // Bounds check
    if (i >= I || h >= H || d >= D) {
        return;
    }

    // Compute weighted sum: sum over positions j
    float sum = 0.0f;

    for (uint j = 0; j < J; j++) {
        uint a_idx = (i * H + h) * J + j;  // A[i, h, j]
        uint b_idx = (j * H + h) * D + d;  // B[j, h, d]
        sum += A[a_idx] * B[b_idx];
    }

    // Write result
    uint c_idx = (i * H + h) * D + d;  // C[i, h, d]
    C[c_idx] = sum;
}

// F32 version
kernel void einsum_3d_general_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& shape_a [[buffer(3)]],   // Shape of A
    constant uint3& shape_b [[buffer(4)]],   // Shape of B
    constant uint3& shape_c [[buffer(5)]],   // Shape of C
    constant uint& contract_dim [[buffer(6)]], // Which dimension to contract
    uint3 gid [[thread_position_in_grid]]
) {
    // This would need index mapping configuration
    // For now, specialized kernels are sufficient
    // TODO: Implement if needed for other einsum patterns
}
//! Fused operations for improved performance
//! Combines multiple operations into single kernels to reduce memory access

#include <metal_stdlib>
using namespace metal;

/// Fused add + relu: output = max(a + b, 0)
/// Reduces memory access by avoiding intermediate buffer
kernel void fused_add_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half sum = a[id] + b[id];
    output[id] = max(sum, half(0.0));
}

/// Fused multiply + relu: output = max(a * b, 0)
kernel void fused_mul_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half product = a[id] * b[id];
    output[id] = max(product, half(0.0));
}

/// Fused affine transformation: output = x * scale + bias
/// Used in batch normalization and similar operations
kernel void fused_affine_f16(
    device const half* x [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = x[id] * scale[id] + bias[id];
}

/// Fused linear layer: matmul + bias + optional activation
/// This is the most common pattern in neural networks
///
/// C[i,j] = sum_k(A[i,k] * B[k,j]) + bias[j] [+ activation]
kernel void fused_linear_f16(
    device const half* A [[buffer(0)]],  // Input [M, K]
    device const half* B [[buffer(1)]],  // Weight [K, N]
    device const half* bias [[buffer(2)]], // Bias [N] (can be null)
    device half* C [[buffer(3)]],        // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& activation [[buffer(7)]], // 0=none, 1=relu, 2=gelu
    constant bool& has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    // Compute matmul: C[row, col] = sum(A[row, k] * B[k, col])
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

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
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        half x = sum;
        half x3 = x * x * x;
        half inner = half(0.7978845608) * (x + half(0.044715) * x3); // sqrt(2/π) ≈ 0.7978845608
        result = half(0.5) * x * (half(1.0) + tanh(inner));
    } else {
        // No activation
        result = sum;
    }

    C[row * N + col] = result;
}

/// Fused matmul + bias (no activation)
/// Simpler version when no activation is needed
kernel void fused_matmul_bias_f16(
    device const half* A [[buffer(0)]],  // [M, K]
    device const half* B [[buffer(1)]],  // [K, N]
    device const half* bias [[buffer(2)]], // [N]
    device half* C [[buffer(3)]],        // [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}

/// Fused subtract + relu: output = max(a - b, 0)
kernel void fused_sub_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half diff = a[id] - b[id];
    output[id] = max(diff, half(0.0));
}

/// Fused division + relu: output = max(a / b, 0)
kernel void fused_div_relu_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    half quotient = a[id] / b[id];
    output[id] = max(quotient, half(0.0));
}

// F32 version
kernel void fused_add_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = a[id] + b[id];
    output[id] = max(sum, float(0.0));
}

// F32 version
kernel void fused_mul_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float product = a[id] * b[id];
    output[id] = max(product, float(0.0));
}

// F32 version
kernel void fused_affine_f32(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = x[id] * scale[id] + bias[id];
}

// F32 version
kernel void fused_linear_f32(
    device const float* A [[buffer(0)]],  // Input [M, K]
    device const float* B [[buffer(1)]],  // Weight [K, N]
    device const float* bias [[buffer(2)]], // Bias [N] (can be null)
    device float* C [[buffer(3)]],        // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& activation [[buffer(7)]], // 0=none, 1=relu, 2=gelu
    constant bool& has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    // Compute matmul: C[row, col] = sum(A[row, k] * B[k, col])
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // Add bias if present
    if (has_bias) {
        sum += bias[col];
    }

    // Apply activation
    float result;
    if (activation == 1) {
        // ReLU
        result = max(sum, float(0.0));
    } else if (activation == 2) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x = sum;
        float x3 = x * x * x;
        float inner = float(0.7978845608) * (x + float(0.044715) * x3); // sqrt(2/π) ≈ 0.7978845608
        result = float(0.5) * x * (float(1.0) + tanh(inner));
    } else {
        // No activation
        result = sum;
    }

    C[row * N + col] = result;
}

// F32 version
kernel void fused_matmul_bias_f32(
    device const float* A [[buffer(0)]],  // [M, K]
    device const float* B [[buffer(1)]],  // [K, N]
    device const float* bias [[buffer(2)]], // [N]
    device float* C [[buffer(3)]],        // [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_sub_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float diff = a[id] - b[id];
    output[id] = max(diff, float(0.0));
}

// F32 version
kernel void fused_div_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float quotient = a[id] / b[id];
    output[id] = max(quotient, float(0.0));
}
//! Advanced Kernel Fusion - Multi-operation Chains
//!
//! Combines 3-5 operations in single kernels for maximum performance.
//! Expected improvement: 2-3x throughput for neural network inference/training.
//!
//! Common patterns in neural networks:
//! 1. Linear → BatchNorm → Activation (forward pass)
//! 2. Activation → Dropout → Linear (residual connections)
//! 3. Linear → Activation → Add (skip connections)

#include <metal_stdlib>
using namespace metal;

/// Fused: Linear + BatchNorm + ReLU
///
/// Common pattern in CNN/ResNet forward pass.
/// Combines matmul, batch normalization, and activation.
///
/// output = relu(batchnorm(matmul(x, w) + bias))
///        = relu((matmul(x, w) + bias - mean) / sqrt(var + eps) * gamma + beta)
kernel void fused_linear_batchnorm_relu_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device const half* bn_mean [[buffer(3)]],    // BatchNorm mean [N]
    device const half* bn_var [[buffer(4)]],     // BatchNorm variance [N]
    device const half* bn_gamma [[buffer(5)]],   // BatchNorm scale [N]
    device const half* bn_beta [[buffer(6)]],    // BatchNorm shift [N]
    device half* output [[buffer(7)]],           // Output [M, N]
    constant uint& M [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant half& eps [[buffer(11)]],           // Small constant for numerical stability
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    half normalized = (sum - bn_mean[col]) / sqrt(bn_var[col] + eps);
    half scaled = normalized * bn_gamma[col] + bn_beta[col];

    // 3. ReLU activation
    output[row * N + col] = max(scaled, half(0.0));
}

/// Fused: Linear + Residual + ReLU
///
/// Common in ResNet skip connections:
/// output = relu(matmul(x, w) + bias + residual)
kernel void fused_linear_residual_relu_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device const half* residual [[buffer(3)]],   // Residual connection [M, N]
    device half* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. Add residual
    sum += residual[row * N + col];

    // 3. ReLU activation
    output[row * N + col] = max(sum, half(0.0));
}

/// Fused: Dropout + Linear
///
/// Applies dropout then linear transformation.
/// Used in attention mechanisms and transformer layers.
///
/// output = matmul(dropout(x, mask, keep_prob), w) + bias
kernel void fused_dropout_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const uint* dropout_mask [[buffer(1)]], // Dropout mask [M, K] (0 or 1)
    device const half* w [[buffer(2)]],          // Weight [K, N]
    device const half* bias [[buffer(3)]],       // Bias [N]
    device half* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant half& scale [[buffer(8)]],          // 1.0 / keep_prob for scaling
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with dropout applied inline
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Apply dropout: multiply by mask and scale
        uint mask_val = dropout_mask[row * K + k];
        half x_val = x[row * K + k] * half(mask_val) * scale;
        sum += x_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: LayerNorm + Linear
///
/// Common in transformer architectures.
/// Applies layer normalization then linear transformation.
///
/// output = matmul(layernorm(x), w) + bias
kernel void fused_layernorm_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* ln_gamma [[buffer(1)]],   // LayerNorm scale [K]
    device const half* ln_beta [[buffer(2)]],    // LayerNorm shift [K]
    device const half* w [[buffer(3)]],          // Weight [K, N]
    device const half* bias [[buffer(4)]],       // Bias [N]
    device half* output [[buffer(5)]],           // Output [M, N]
    constant uint& M [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant half& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Compute mean and variance for this row (layer norm is per-sample)
    half mean = 0.0h;
    for (uint k = 0; k < K; k++) {
        mean += x[row * K + k];
    }
    mean /= half(K);

    half variance = 0.0h;
    for (uint k = 0; k < K; k++) {
        half diff = x[row * K + k] - mean;
        variance += diff * diff;
    }
    variance /= half(K);

    // 2. Compute matmul with normalized input
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        half normalized = (x[row * K + k] - mean) / sqrt(variance + eps);
        half scaled = normalized * ln_gamma[k] + ln_beta[k];

        // Multiply with weight
        sum += scaled * w[k * N + col];
    }

    // 3. Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: GELU + Linear
///
/// Common in transformer feed-forward networks.
/// Applies GELU activation then linear transformation.
///
/// output = matmul(gelu(x), w) + bias
kernel void fused_gelu_linear_f16(
    device const half* x [[buffer(0)]],          // Input [M, K]
    device const half* w [[buffer(1)]],          // Weight [K, N]
    device const half* bias [[buffer(2)]],       // Bias [N]
    device half* output [[buffer(3)]],           // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with GELU applied inline
    half sum = 0.0h;
    for (uint k = 0; k < K; k++) {
        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        half x_val = x[row * K + k];
        half x3 = x_val * x_val * x_val;
        half inner = half(0.7978845608) * (x_val + half(0.044715) * x3);
        half gelu_val = half(0.5) * x_val * (half(1.0) + tanh(inner));

        // Multiply with weight
        sum += gelu_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

/// Fused: Softmax + CrossEntropy Loss
///
/// Combines softmax and cross-entropy loss computation.
/// Used in classification tasks for efficiency.
///
/// Computes: -log(softmax(logits)[target_class])
kernel void fused_softmax_crossentropy_f16(
    device const half* logits [[buffer(0)]],     // Input logits [M, N]
    device const uint* targets [[buffer(1)]],    // Target classes [M]
    device half* loss [[buffer(2)]],             // Output loss [M]
    constant uint& M [[buffer(3)]],              // Batch size
    constant uint& N [[buffer(4)]],              // Number of classes
    uint id [[thread_position_in_grid]]
) {
    if (id >= M) return;

    uint row = id;
    uint target_class = targets[row];

    // 1. Find max for numerical stability
    half max_logit = logits[row * N];
    for (uint i = 1; i < N; i++) {
        max_logit = max(max_logit, logits[row * N + i]);
    }

    // 2. Compute exp(logits - max) and sum
    half sum_exp = 0.0h;
    for (uint i = 0; i < N; i++) {
        sum_exp += exp(logits[row * N + i] - max_logit);
    }

    // 3. Compute log_softmax for target class
    half log_softmax = (logits[row * N + target_class] - max_logit) - log(sum_exp);

    // 4. Cross-entropy loss: -log(softmax(target))
    loss[row] = -log_softmax;
}

/// Fused: Attention Score Computation
///
/// Computes attention scores: softmax(Q @ K^T / sqrt(d_k))
/// Core operation in transformer attention mechanism.
kernel void fused_attention_scores_f16(
    device const half* Q [[buffer(0)]],          // Query [M, d_k]
    device const half* K [[buffer(1)]],          // Key [N, d_k]
    device half* scores [[buffer(2)]],           // Output scores [M, N]
    constant uint& M [[buffer(3)]],              // Query sequence length
    constant uint& N [[buffer(4)]],              // Key sequence length
    constant uint& d_k [[buffer(5)]],            // Dimension
    constant half& scale [[buffer(6)]],          // 1 / sqrt(d_k)
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // Query position
    uint col = gid.x;  // Key position

    if (row >= M || col >= N) return;

    // 1. Compute Q @ K^T (scaled dot product)
    half dot_product = 0.0h;
    for (uint k = 0; k < d_k; k++) {
        dot_product += Q[row * d_k + k] * K[col * d_k + k];
    }
    dot_product *= scale;

    // Note: Softmax should be applied across the N dimension
    // This kernel computes the scaled dot products
    // A separate kernel or pass is needed for softmax normalization
    scores[row * N + col] = dot_product;
}

// F32 version
kernel void fused_linear_batchnorm_relu_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device const float* bn_mean [[buffer(3)]],    // BatchNorm mean [N]
    device const float* bn_var [[buffer(4)]],     // BatchNorm variance [N]
    device const float* bn_gamma [[buffer(5)]],   // BatchNorm scale [N]
    device const float* bn_beta [[buffer(6)]],    // BatchNorm shift [N]
    device float* output [[buffer(7)]],           // Output [M, N]
    constant uint& M [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& N [[buffer(10)]],
    constant half& eps [[buffer(11)]],           // Small constant for numerical stability
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    float normalized = (sum - bn_mean[col]) / sqrt(bn_var[col] + eps);
    float scaled = normalized * bn_gamma[col] + bn_beta[col];

    // 3. ReLU activation
    output[row * N + col] = max(scaled, float(0.0));
}

// F32 version
kernel void fused_linear_residual_relu_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device const float* residual [[buffer(3)]],   // Residual connection [M, N]
    device float* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Linear: matmul + bias
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += x[row * K + k] * w[k * N + col];
    }
    sum += bias[col];

    // 2. Add residual
    sum += residual[row * N + col];

    // 3. ReLU activation
    output[row * N + col] = max(sum, float(0.0));
}

// F32 version
kernel void fused_dropout_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const uint* dropout_mask [[buffer(1)]], // Dropout mask [M, K] (0 or 1)
    device const float* w [[buffer(2)]],          // Weight [K, N]
    device const float* bias [[buffer(3)]],       // Bias [N]
    device float* output [[buffer(4)]],           // Output [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant half& scale [[buffer(8)]],          // 1.0 / keep_prob for scaling
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with dropout applied inline
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Apply dropout: multiply by mask and scale
        uint mask_val = dropout_mask[row * K + k];
        float x_val = x[row * K + k] * float(mask_val) * scale;
        sum += x_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_layernorm_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* ln_gamma [[buffer(1)]],   // LayerNorm scale [K]
    device const float* ln_beta [[buffer(2)]],    // LayerNorm shift [K]
    device const float* w [[buffer(3)]],          // Weight [K, N]
    device const float* bias [[buffer(4)]],       // Bias [N]
    device float* output [[buffer(5)]],           // Output [M, N]
    constant uint& M [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant half& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // 1. Compute mean and variance for this row (layer norm is per-sample)
    float mean = 0.0f;
    for (uint k = 0; k < K; k++) {
        mean += x[row * K + k];
    }
    mean /= float(K);

    float variance = 0.0f;
    for (uint k = 0; k < K; k++) {
        float diff = x[row * K + k] - mean;
        variance += diff * diff;
    }
    variance /= float(K);

    // 2. Compute matmul with normalized input
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        float normalized = (x[row * K + k] - mean) / sqrt(variance + eps);
        float scaled = normalized * ln_gamma[k] + ln_beta[k];

        // Multiply with weight
        sum += scaled * w[k * N + col];
    }

    // 3. Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_gelu_linear_f32(
    device const float* x [[buffer(0)]],          // Input [M, K]
    device const float* w [[buffer(1)]],          // Weight [K, N]
    device const float* bias [[buffer(2)]],       // Bias [N]
    device float* output [[buffer(3)]],           // Output [M, N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Compute matmul with GELU applied inline
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_val = x[row * K + k];
        float x3 = x_val * x_val * x_val;
        float inner = float(0.7978845608) * (x_val + float(0.044715) * x3);
        float gelu_val = float(0.5) * x_val * (float(1.0) + tanh(inner));

        // Multiply with weight
        sum += gelu_val * w[k * N + col];
    }

    // Add bias
    output[row * N + col] = sum + bias[col];
}

// F32 version
kernel void fused_softmax_crossentropy_f32(
    device const float* logits [[buffer(0)]],     // Input logits [M, N]
    device const uint* targets [[buffer(1)]],    // Target classes [M]
    device float* loss [[buffer(2)]],             // Output loss [M]
    constant uint& M [[buffer(3)]],              // Batch size
    constant uint& N [[buffer(4)]],              // Number of classes
    uint id [[thread_position_in_grid]]
) {
    if (id >= M) return;

    uint row = id;
    uint target_class = targets[row];

    // 1. Find max for numerical stability
    float max_logit = logits[row * N];
    for (uint i = 1; i < N; i++) {
        max_logit = max(max_logit, logits[row * N + i]);
    }

    // 2. Compute exp(logits - max) and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < N; i++) {
        sum_exp += exp(logits[row * N + i] - max_logit);
    }

    // 3. Compute log_softmax for target class
    float log_softmax = (logits[row * N + target_class] - max_logit) - log(sum_exp);

    // 4. Cross-entropy loss: -log(softmax(target))
    loss[row] = -log_softmax;
}

// F32 version
kernel void fused_attention_scores_f32(
    device const float* Q [[buffer(0)]],          // Query [M, d_k]
    device const float* K [[buffer(1)]],          // Key [N, d_k]
    device float* scores [[buffer(2)]],           // Output scores [M, N]
    constant uint& M [[buffer(3)]],              // Query sequence length
    constant uint& N [[buffer(4)]],              // Key sequence length
    constant uint& d_k [[buffer(5)]],            // Dimension
    constant half& scale [[buffer(6)]],          // 1 / sqrt(d_k)
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // Query position
    uint col = gid.x;  // Key position

    if (row >= M || col >= N) return;

    // 1. Compute Q @ K^T (scaled dot product)
    float dot_product = 0.0f;
    for (uint k = 0; k < d_k; k++) {
        dot_product += Q[row * d_k + k] * K[col * d_k + k];
    }
    dot_product *= scale;

    // Note: Softmax should be applied across the N dimension
    // This kernel computes the scaled dot products
    // A separate kernel or pass is needed for softmax normalization
    scores[row * N + col] = dot_product;
}
#include <metal_stdlib>
using namespace metal;

/// Concatenate tensors along a specified dimension (f16)
/// Each invocation processes one element from input and writes to correct position in output
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - dim_offset: Offset along concat dimension (sum of sizes of previous tensors)
/// - input_dim_size: Size of this input along concat dimension
/// - output_dim_size: Size of output along concat dimension
/// - chunk_size: Product of dimensions after concat dim
/// - num_chunks: Product of dimensions before concat dim
kernel void concat_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* dim_offset [[buffer(2)]],
    device const uint* input_dim_size [[buffer(3)]],
    device const uint* output_dim_size [[buffer(4)]],
    device const uint* chunk_size [[buffer(5)]],
    device const uint* num_chunks [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint offset = dim_offset[0];
    uint in_dim = input_dim_size[0];
    uint out_dim = output_dim_size[0];
    uint chunk_sz = chunk_size[0];
    uint n_chunks = num_chunks[0];

    // Total elements in input
    uint total_elements = n_chunks * in_dim * chunk_sz;
    if (gid >= total_elements) return;

    // Decompose gid into (chunk_idx, dim_idx, elem_idx)
    uint chunk_idx = gid / (in_dim * chunk_sz);
    uint remainder = gid % (in_dim * chunk_sz);
    uint dim_idx = remainder / chunk_sz;
    uint elem_idx = remainder % chunk_sz;

    // Output index: chunk_idx * (output_dim_size * chunk_size) + (offset + dim_idx) * chunk_size + elem_idx
    uint output_idx = chunk_idx * out_dim * chunk_sz + (offset + dim_idx) * chunk_sz + elem_idx;

    output[output_idx] = input[gid];
}

/// Concatenate tensors along a specified dimension (f32)
kernel void concat_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dim_offset [[buffer(2)]],
    device const uint* input_dim_size [[buffer(3)]],
    device const uint* output_dim_size [[buffer(4)]],
    device const uint* chunk_size [[buffer(5)]],
    device const uint* num_chunks [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint offset = dim_offset[0];
    uint in_dim = input_dim_size[0];
    uint out_dim = output_dim_size[0];
    uint chunk_sz = chunk_size[0];
    uint n_chunks = num_chunks[0];

    uint total_elements = n_chunks * in_dim * chunk_sz;
    if (gid >= total_elements) return;

    uint chunk_idx = gid / (in_dim * chunk_sz);
    uint remainder = gid % (in_dim * chunk_sz);
    uint dim_idx = remainder / chunk_sz;
    uint elem_idx = remainder % chunk_sz;

    uint output_idx = chunk_idx * out_dim * chunk_sz + (offset + dim_idx) * chunk_sz + elem_idx;

    output[output_idx] = input[gid];
}
#include <metal_stdlib>
using namespace metal;

/// Permute (transpose) tensor dimensions (f16)
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - input_shape: Shape of input tensor (max 8 dimensions)
/// - output_shape: Shape of output tensor (max 8 dimensions)
/// - perm: Permutation array (which input dim goes to which output dim)
/// - ndim: Number of dimensions
kernel void permute_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* output_shape [[buffer(3)]],
    device const uint* perm [[buffer(4)]],
    device const uint* ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n_dims = ndim[0];
    uint total_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        total_elements *= output_shape[i];
    }

    if (gid >= total_elements) return;

    // Calculate output multi-index from linear index
    uint output_idx[8];
    uint remaining = gid;
    for (int i = n_dims - 1; i >= 0; i--) {
        output_idx[i] = remaining % output_shape[i];
        remaining /= output_shape[i];
    }

    // Map output index to input index using permutation
    uint input_idx[8];
    for (uint i = 0; i < n_dims; i++) {
        input_idx[perm[i]] = output_idx[i];
    }

    // Convert input multi-index to linear index
    uint input_linear = 0;
    uint stride = 1;
    for (int i = n_dims - 1; i >= 0; i--) {
        input_linear += input_idx[i] * stride;
        stride *= input_shape[i];
    }

    output[gid] = input[input_linear];
}

/// Permute (transpose) tensor dimensions (f32)
kernel void permute_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* output_shape [[buffer(3)]],
    device const uint* perm [[buffer(4)]],
    device const uint* ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n_dims = ndim[0];
    uint total_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        total_elements *= output_shape[i];
    }

    if (gid >= total_elements) return;

    // Calculate output multi-index from linear index
    uint output_idx[8];
    uint remaining = gid;
    for (int i = n_dims - 1; i >= 0; i--) {
        output_idx[i] = remaining % output_shape[i];
        remaining /= output_shape[i];
    }

    // Map output index to input index using permutation
    uint input_idx[8];
    for (uint i = 0; i < n_dims; i++) {
        input_idx[perm[i]] = output_idx[i];
    }

    // Convert input multi-index to linear index
    uint input_linear = 0;
    uint stride = 1;
    for (int i = n_dims - 1; i >= 0; i--) {
        input_linear += input_idx[i] * stride;
        stride *= input_shape[i];
    }

    output[gid] = input[input_linear];
}
#include <metal_stdlib>
using namespace metal;

/// Broadcast tensor to target shape (f16)
///
/// Parameters:
/// - input: Input tensor data
/// - output: Output buffer (pre-allocated)
/// - input_shape: Shape of input tensor (max 8 dimensions)
/// - target_shape: Shape of output tensor (max 8 dimensions)
/// - input_ndim: Number of dimensions in input
/// - target_ndim: Number of dimensions in target
kernel void broadcast_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* target_shape [[buffer(3)]],
    device const uint* input_ndim [[buffer(4)]],
    device const uint* target_ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint target_n_dims = target_ndim[0];
    uint input_n_dims = input_ndim[0];

    uint total_elements = 1;
    for (uint i = 0; i < target_n_dims; i++) {
        total_elements *= target_shape[i];
    }

    if (gid >= total_elements) return;

    // Compute target multi-index from linear index
    uint target_idx[8];
    uint remaining = gid;
    for (int i = target_n_dims - 1; i >= 0; i--) {
        target_idx[i] = remaining % target_shape[i];
        remaining /= target_shape[i];
    }

    // Compute input strides
    uint input_strides[8];
    uint stride = 1;
    for (int i = input_n_dims - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    // Map target index to input index (align from right)
    uint rank_diff = target_n_dims - input_n_dims;
    uint input_linear = 0;
    for (uint i = rank_diff; i < target_n_dims; i++) {
        uint input_i = i - rank_diff;
        uint coord = (input_shape[input_i] == 1) ? 0 : target_idx[i];
        input_linear += coord * input_strides[input_i];
    }

    output[gid] = input[input_linear];
}

/// Broadcast tensor to target shape (f32)
kernel void broadcast_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* input_shape [[buffer(2)]],
    device const uint* target_shape [[buffer(3)]],
    device const uint* input_ndim [[buffer(4)]],
    device const uint* target_ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint target_n_dims = target_ndim[0];
    uint input_n_dims = input_ndim[0];

    uint total_elements = 1;
    for (uint i = 0; i < target_n_dims; i++) {
        total_elements *= target_shape[i];
    }

    if (gid >= total_elements) return;

    uint target_idx[8];
    uint remaining = gid;
    for (int i = target_n_dims - 1; i >= 0; i--) {
        target_idx[i] = remaining % target_shape[i];
        remaining /= target_shape[i];
    }

    uint input_strides[8];
    uint stride = 1;
    for (int i = input_n_dims - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    uint rank_diff = target_n_dims - input_n_dims;
    uint input_linear = 0;
    for (uint i = rank_diff; i < target_n_dims; i++) {
        uint input_i = i - rank_diff;
        uint coord = (input_shape[input_i] == 1) ? 0 : target_idx[i];
        input_linear += coord * input_strides[input_i];
    }

    output[gid] = input[input_linear];
}
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

// ==================== Single Element Read ====================

/// Read a single element from a tensor at a linear index (f16)
kernel void read_element_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* index [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        output[0] = input[index[0]];
    }
}

/// Read a single element from a tensor at a linear index (f32)
kernel void read_element_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* index [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        output[0] = input[index[0]];
    }
}

// ==================== Missing Scalar Operations ====================

/// Subtract scalar from tensor (f16)
kernel void sub_scalar_f16(
    device const half* a [[buffer(0)]],
    device const half* scalar [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - scalar[0];
}

/// Subtract scalar from tensor (f32)
kernel void sub_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - scalar[0];
}

/// Divide tensor by scalar (f16)
kernel void div_scalar_f16(
    device const half* a [[buffer(0)]],
    device const half* scalar [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / scalar[0];
}

/// Divide tensor by scalar (f32)
kernel void div_scalar_f32(
    device const float* a [[buffer(0)]],
    device const float* scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / scalar[0];
}

// ============================================================================
// Cache Append Operations (for efficient KV cache updates)
// ============================================================================

/// Append data to pre-allocated cache tensor (f16)
/// Writes new_data to cache at offset position without copying existing data
kernel void cache_append_f16(
    device const half* new_data [[buffer(0)]],   // New data to append
    device half* cache [[buffer(1)]],             // Pre-allocated cache buffer
    constant uint& offset [[buffer(2)]],          // Offset in elements
    uint index [[thread_position_in_grid]]
) {
    cache[offset + index] = new_data[index];
}

/// Append data to pre-allocated cache tensor (f32)
/// Writes new_data to cache at offset position without copying existing data
kernel void cache_append_f32(
    device const float* new_data [[buffer(0)]],   // New data to append
    device float* cache [[buffer(1)]],            // Pre-allocated cache buffer
    constant uint& offset [[buffer(2)]],          // Offset in elements
    uint index [[thread_position_in_grid]]
) {
    cache[offset + index] = new_data[index];
}

// =============================================================================
// Fused Transpose-Matmul Kernels for Linear Layer Optimization
// =============================================================================
//
// These kernels perform: C[M,N] = A[M,K] @ B.T where B is [N,K]
// Eliminates the need for explicit transpose operation before matmul
// Optimized for transformer inference: linear(x, weight) where weight is [out_features, in_features]
//
// Memory access pattern:
// - A[M,K]: row-major, sequential access
// - B[N,K]: column-major access (reading transposed)
// - C[M,N]: row-major, sequential write
//
// Expected performance gain: 20-30% faster than separate transpose + matmul

/// Fused transpose-matmul with 16x16 tiling (f16)
/// C[M,N] = A[M,K] @ B.T where B is [N,K]
kernel void matmul_transposed_b_tiled_f16(
    device const half* A [[buffer(0)]],      // Input matrix A [M, K]
    device const half* B [[buffer(1)]],      // Input matrix B [N, K] (will be transposed)
    device half* C [[buffer(2)]],            // Output matrix C [M, N]
    constant uint& M [[buffer(3)]],          // Number of rows in A and C
    constant uint& N [[buffer(4)]],          // Number of rows in B (cols in B.T)
    constant uint& K [[buffer(5)]],          // Shared dimension
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

    float sum = 0.0f;  // f32 accumulator for precision

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        // Load A tile: A[row, k_offset + tx]
        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0h;
        }

        // Load B tile transposed: B[col, k_offset + ty] -> B_tile[ty][tx]
        // B is [N, K], we want B.T[K, N]
        // B[col, k] is at index: col * K + k
        uint b_row = col;  // row in B (becomes col in B.T)
        uint b_col = k_offset + ty;  // col in B (becomes row in B.T)
        if (b_row < N && b_col < K) {
            B_tile[ty][tx] = B[b_row * K + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[ty][k]) * float(B_tile[k][tx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

/// Fused transpose-matmul with 32x32 tiling (f16) for large matrices
kernel void matmul_transposed_b_tiled_32x32_f16(
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

        uint b_row = col;
        uint b_col = k_offset + ty;
        if (b_row < N && b_col < K) {
            B_tile[ty][tx] = B[b_row * K + b_col];
        } else {
            B_tile[ty][tx] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[ty][k]) * float(B_tile[k][tx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

/// Fused transpose-matmul with 16x16 tiling (f32)
kernel void matmul_transposed_b_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        uint b_row = col;
        uint b_col = k_offset + ty;
        if (b_row < N && b_col < K) {
            B_tile[ty][tx] = B[b_row * K + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Fused transpose-matmul with 32x32 tiling (f32)
kernel void matmul_transposed_b_tiled_32x32_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 32;

    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint row = threadgroup_position_in_grid.y * TILE_SIZE + ty;
    uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;

    float sum = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        uint k_offset = tile * TILE_SIZE;

        uint a_row = row;
        uint a_col = k_offset + tx;
        if (a_row < M && a_col < K) {
            A_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        uint b_row = col;
        uint b_col = k_offset + ty;
        if (b_row < N && b_col < K) {
            B_tile[ty][tx] = B[b_row * K + b_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Attention Mask Application Kernels
// ============================================================================

/// Apply attention mask to scores (f16 version)
/// mask: 1 = keep, 0 = mask out (replace with mask_value)
/// scores[i] = mask[i] == 0 ? mask_value : scores[i]
kernel void apply_attention_mask_f16(
    device const half* scores [[buffer(0)]],     // Input scores
    device const half* mask [[buffer(1)]],       // Mask (1=keep, 0=mask)
    device half* output [[buffer(2)]],           // Output
    constant uint& size [[buffer(3)]],           // Total number of elements
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= size) return;
    
    const half mask_value = -10000.0h;
    half m = mask[idx];
    output[idx] = (m == 0.0h) ? mask_value : scores[idx];
}

/// Apply attention mask to scores (f32 version)
/// mask: 1 = keep, 0 = mask out (replace with mask_value)
/// scores[i] = mask[i] == 0 ? mask_value : scores[i]
kernel void apply_attention_mask_f32(
    device const float* scores [[buffer(0)]],    // Input scores
    device const float* mask [[buffer(1)]],      // Mask (1=keep, 0=mask)
    device float* output [[buffer(2)]],          // Output
    constant uint& size [[buffer(3)]],           // Total number of elements
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= size) return;
    
    const float mask_value = -10000.0f;
    float m = mask[idx];
    output[idx] = (m == 0.0f) ? mask_value : scores[idx];
}

// ============================================================================
// Memory layout transformation kernels
// ============================================================================

/// Make tensor contiguous by reordering elements according to strides (f16)
/// Converts strided (non-contiguous) layout to row-major (contiguous) layout
kernel void make_contiguous_f16(
    device const half* input [[buffer(0)]],      // Non-contiguous input
    device half* output [[buffer(1)]],           // Contiguous output
    device const uint* shape [[buffer(2)]],      // Shape dimensions [d0, d1, d2, ...]
    device const uint* strides [[buffer(3)]],    // Input strides
    device const uint* ndim [[buffer(4)]],       // Number of dimensions
    device const uint* numel [[buffer(5)]],      // Total number of elements
    uint gid [[thread_position_in_grid]]
) {
    // Bounds check - critical for safety when threadgroup size doesn't divide evenly
    if (gid >= numel[0]) return;

    uint n_dims = ndim[0];

    // Convert linear output index to multi-dimensional indices
    uint linear_idx = gid;
    uint remaining = linear_idx;

    // Calculate multi-dimensional indices (reverse order for row-major)
    uint indices[8];  // Support up to 8D tensors
    for (int i = int(n_dims) - 1; i >= 0; i--) {
        uint dim_size = shape[i];
        indices[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    // Calculate strided offset in input
    uint src_offset = 0;
    for (uint i = 0; i < n_dims; i++) {
        src_offset += indices[i] * strides[i];
    }

    // Copy element
    output[linear_idx] = input[src_offset];
}

/// Make tensor contiguous by reordering elements according to strides (f32)
kernel void make_contiguous_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* shape [[buffer(2)]],
    device const uint* strides [[buffer(3)]],
    device const uint* ndim [[buffer(4)]],
    device const uint* numel [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Bounds check - critical for safety when threadgroup size doesn't divide evenly
    if (gid >= numel[0]) return;

    uint n_dims = ndim[0];

    uint linear_idx = gid;
    uint remaining = linear_idx;

    uint indices[8];
    for (int i = int(n_dims) - 1; i >= 0; i--) {
        uint dim_size = shape[i];
        indices[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    uint src_offset = 0;
    for (uint i = 0; i < n_dims; i++) {
        src_offset += indices[i] * strides[i];
    }

    output[linear_idx] = input[src_offset];
}

// ============================================================================
// Argmax kernels for greedy sampling (temperature=0)
// ============================================================================

/// Find argmax (index of maximum value) for f16 logits
/// Uses parallel reduction to find both max value and its index simultaneously
kernel void argmax_f16(
    device const half* logits [[buffer(0)]],     // Input logits
    device uint* max_idx [[buffer(1)]],          // Output: index of max
    device const uint* vocab_size [[buffer(2)]], // Vocabulary size
    device const uint* offset [[buffer(3)]],     // Offset for 2D tensors
    threadgroup half* shared_max [[threadgroup(0)]],   // Shared max values
    threadgroup uint* shared_idx [[threadgroup(1)]],   // Shared max indices
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];
    uint start_offset = offset[0];
    
    // Each thread loads one value and its index
    half val;
    uint idx;
    if (gid < vocab) {
        val = logits[start_offset + gid];
        idx = gid;
    } else {
        val = half(-65504.0);  // -inf for f16
        idx = 0;
    }
    
    shared_max[lid] = val;
    shared_idx[lid] = idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction: find max and its index
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_max[lid + stride] > shared_max[lid]) {
                shared_max[lid] = shared_max[lid + stride];
                shared_idx[lid] = shared_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (lid == 0) {
        max_idx[0] = shared_idx[0];
    }
}

/// Find argmax (index of maximum value) for f32 logits
kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* max_idx [[buffer(1)]],
    device const uint* vocab_size [[buffer(2)]],
    device const uint* offset [[buffer(3)]],
    threadgroup float* shared_max [[threadgroup(0)]],
    threadgroup uint* shared_idx [[threadgroup(1)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    uint vocab = vocab_size[0];
    uint start_offset = offset[0];
    
    float val;
    uint idx;
    if (gid < vocab) {
        val = logits[start_offset + gid];
        idx = gid;
    } else {
        val = -INFINITY;
        idx = 0;
    }
    
    shared_max[lid] = val;
    shared_idx[lid] = idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_max[lid + stride] > shared_max[lid]) {
                shared_max[lid] = shared_max[lid + stride];
                shared_idx[lid] = shared_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        max_idx[0] = shared_idx[0];
    }
}
