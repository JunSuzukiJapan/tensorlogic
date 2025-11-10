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
