// ============================================
// Single-Head Scaled Dot-Product Attention
// ============================================
// 
// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Args:
//   Q: [seq_len, d_k] - Query matrix
//   K: [seq_len, d_k] - Key matrix  
//   V: [seq_len, d_v] - Value matrix
//   mask: Optional causal mask
//
// Returns:
//   output: [seq_len, d_v]
//

fn scaled_dot_product_attention(Q, K, V, mask) {
    // Q: [seq_len_q, d_k]
    // K: [seq_len_k, d_k]
    // V: [seq_len_k, d_v]
    
    let d_k = shape(K)[1]  // Key dimension
    let scale = 1.0 / sqrt(d_k)
    
    // Compute attention scores: Q @ K^T
    // [seq_len_q, d_k] @ [d_k, seq_len_k] = [seq_len_q, seq_len_k]
    let K_T = transpose(K)
    let scores = matmul(Q, K_T)
    
    // Scale scores
    let scaled_scores = scores * scale
    
    // Apply causal mask if provided
    // (mask contains -inf for positions to mask out)
    let masked_scores = if mask {
        apply_mask(scaled_scores, mask)
    } else {
        scaled_scores
    }
    
    // Apply softmax to get attention weights
    let attn_weights = softmax(masked_scores, -1)  // dim=-1 (last dimension)
    
    // Compute weighted sum of values
    // [seq_len_q, seq_len_k] @ [seq_len_k, d_v] = [seq_len_q, d_v]
    let output = matmul(attn_weights, V)
    
    return output
}


// ============================================
// Multi-Head Attention (Standard)
// ============================================
//
// Args:
//   x: [seq_len, d_model] - Input
//   W_q: [d_model, d_model] - Query projection weight
//   W_k: [d_model, d_model] - Key projection weight
//   W_v: [d_model, d_model] - Value projection weight  
//   W_o: [d_model, d_model] - Output projection weight
//   num_heads: Number of attention heads
//   mask: Optional causal mask
//
fn multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads, mask) {
    let seq_len = shape(x)[0]
    let d_model = shape(x)[1]
    let d_k = d_model / num_heads  // Head dimension
    
    // Linear projections
    let Q = matmul(x, W_q)  // [seq_len, d_model]
    let K = matmul(x, W_k)  // [seq_len, d_model]
    let V = matmul(x, W_v)  // [seq_len, d_model]
    
    // Split into multiple heads
    // [seq_len, d_model] -> [seq_len, num_heads, d_k]
    let Q_heads = reshape(Q, [seq_len, num_heads, d_k])
    let K_heads = reshape(K, [seq_len, num_heads, d_k])
    let V_heads = reshape(V, [seq_len, num_heads, d_k])
    
    // Transpose to [num_heads, seq_len, d_k] for parallel processing
    let Q_heads = permute(Q_heads, [1, 0, 2])
    let K_heads = permute(K_heads, [1, 0, 2])
    let V_heads = permute(V_heads, [1, 0, 2])
    
    // Apply attention for each head
    // For now, we'll process head by head
    // TODO: Batch process all heads in parallel
    let outputs = []
    for i in 0..num_heads {
        let Q_i = Q_heads[i]  // [seq_len, d_k]
        let K_i = K_heads[i]  // [seq_len, d_k]
        let V_i = V_heads[i]  // [seq_len, d_k]
        
        let head_out = scaled_dot_product_attention(Q_i, K_i, V_i, mask)
        outputs.push(head_out)
    }
    
    // Concatenate all heads
    // [num_heads, seq_len, d_k] -> [seq_len, d_model]
    let concat_output = concat(outputs, -1)
    
    // Final linear projection
    let output = matmul(concat_output, W_o)
    
    return output
}


// ============================================
// Grouped-Query Attention (GQA) - TinyLlama
// ============================================
//
// GQA uses fewer KV heads than Q heads for efficiency
// TinyLlama: 32 query heads, 4 KV heads
// Each KV head is shared across 8 query heads (32/4=8)
//
// Args:
//   x: [seq_len, d_model] - Input
//   W_q: [d_model, d_model] - Query projection (32 heads)
//   W_k: [d_model, d_kv] - Key projection (4 heads)
//   W_v: [d_model, d_kv] - Value projection (4 heads)
//   W_o: [d_model, d_model] - Output projection
//   num_q_heads: Number of query heads (32)
//   num_kv_heads: Number of KV heads (4)
//   mask: Optional causal mask
//
fn grouped_query_attention(x, W_q, W_k, W_v, W_o, num_q_heads, num_kv_heads, mask) {
    let seq_len = shape(x)[0]
    let d_model = shape(x)[1]
    let d_k = d_model / num_q_heads  // Head dimension (64 for TinyLlama)
    
    // Query projection: [seq_len, d_model]
    let Q = matmul(x, W_q)
    
    // KV projections: [seq_len, d_kv] where d_kv = num_kv_heads * d_k
    let K = matmul(x, W_k)
    let V = matmul(x, W_v)
    
    // Split Q into query heads: [seq_len, num_q_heads, d_k]
    let Q_heads = reshape(Q, [seq_len, num_q_heads, d_k])
    
    // Split K, V into KV heads: [seq_len, num_kv_heads, d_k]
    let K_heads = reshape(K, [seq_len, num_kv_heads, d_k])
    let V_heads = reshape(V, [seq_len, num_kv_heads, d_k])
    
    // Repeat each KV head to match number of Q heads
    // Each KV head serves num_q_heads / num_kv_heads query heads
    let kv_group_size = num_q_heads / num_kv_heads
    
    // Expand K and V: [seq_len, num_kv_heads, d_k] -> [seq_len, num_q_heads, d_k]
    let K_expanded = repeat_interleave(K_heads, kv_group_size, 1)
    let V_expanded = repeat_interleave(V_heads, kv_group_size, 1)
    
    // Transpose for parallel processing: [num_q_heads, seq_len, d_k]
    let Q_heads = permute(Q_heads, [1, 0, 2])
    let K_expanded = permute(K_expanded, [1, 0, 2])
    let V_expanded = permute(V_expanded, [1, 0, 2])
    
    // Apply attention for each head
    let outputs = []
    for i in 0..num_q_heads {
        let Q_i = Q_heads[i]
        let K_i = K_expanded[i]
        let V_i = V_expanded[i]
        
        let head_out = scaled_dot_product_attention(Q_i, K_i, V_i, mask)
        outputs.push(head_out)
    }
    
    // Concatenate: [seq_len, d_model]
    let concat_output = concat(outputs, -1)
    
    // Output projection
    let output = matmul(concat_output, W_o)
    
    return output
}


// ============================================
// SwiGLU Feed-Forward Network
// ============================================
//
// SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up) W_down
// where SiLU(x) = x * sigmoid(x)
//
// TinyLlama config:
//   d_model = 2048
//   d_ff = 5632 (intermediate dimension)
//
// Args:
//   x: [seq_len, d_model] - Input
//   W_gate: [d_model, d_ff] - Gate projection
//   W_up: [d_model, d_ff] - Up projection
//   W_down: [d_ff, d_model] - Down projection
//
fn swiglu_ffn(x, W_gate, W_up, W_down) {
    // Gate projection and activation
    let gate = matmul(x, W_gate)  // [seq_len, d_ff]
    let gate_activated = sigmoid(gate) * gate  // SiLU activation
    
    // Up projection
    let up = matmul(x, W_up)  // [seq_len, d_ff]
    
    // Element-wise product
    let intermediate = gate_activated * up  // [seq_len, d_ff]
    
    // Down projection
    let output = matmul(intermediate, W_down)  // [seq_len, d_model]
    
    return output
}


// ============================================
// Transformer Layer
// ============================================
//
// Single transformer layer with:
//   - Pre-attention RMSNorm
//   - Grouped-Query Attention
//   - Residual connection
//   - Pre-FFN RMSNorm  
//   - SwiGLU FFN
//   - Residual connection
//
fn transformer_layer(
    x, 
    attn_norm_weight,
    W_q, W_k, W_v, W_o,
    ffn_norm_weight,
    W_gate, W_up, W_down,
    num_q_heads,
    num_kv_heads,
    mask
) {
    // Pre-attention normalization
    let x_norm = rms_norm(x, attn_norm_weight)
    
    // Self-attention with residual
    let attn_out = grouped_query_attention(
        x_norm, W_q, W_k, W_v, W_o,
        num_q_heads, num_kv_heads, mask
    )
    let x = x + attn_out  // Residual connection
    
    // Pre-FFN normalization
    let x_norm = rms_norm(x, ffn_norm_weight)
    
    // Feed-forward network with residual
    let ffn_out = swiglu_ffn(x_norm, W_gate, W_up, W_down)
    let x = x + ffn_out  // Residual connection
    
    return x
}


main {
    print("=== TensorLogic Transformer Implementation ===")
    print("")
    print("✅ Defined functions:")
    print("  • scaled_dot_product_attention(Q, K, V, mask)")
    print("  • multi_head_attention(...)")
    print("  • grouped_query_attention(...) - TinyLlama GQA")
    print("  • swiglu_ffn(x, W_gate, W_up, W_down)")
    print("  • transformer_layer(...)")
    print("")
    print("📋 TinyLlama Architecture:")
    print("  • 22 transformer layers")
    print("  • 32 query heads, 4 KV heads (GQA)")
    print("  • d_model = 2048")
    print("  • d_ff = 5632")
    print("  • vocab_size = 32000")
    print("")
    print("⚠️  Note: Some functions (repeat_interleave, array indexing)")
    print("   need to be implemented or worked around")
}
