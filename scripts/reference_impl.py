#!/usr/bin/env python3
"""
Reference implementation for TinyLlama layer 0 forward pass
Compares with TensorLogic implementation to find numerical differences
"""

import numpy as np
import struct
from pathlib import Path
import json

def load_gguf_tensor(filepath, tensor_name):
    """Load a specific tensor from GGUF file"""
    # This is a simplified loader - for production use gguf library
    print(f"Loading tensor: {tensor_name}")
    # Placeholder - would need full GGUF parser
    return None

def load_tokenizer(tokenizer_path):
    """Load tokenizer JSON"""
    with open(tokenizer_path, 'r') as f:
        return json.load(f)

def tokenize(tokenizer, text):
    """Simple tokenization"""
    # For "Hello", TinyLlama uses token ID 15043
    # This is from the previous test results
    return [15043]

def apply_rope(x, position=0, rope_base=10000.0):
    """
    Apply RoPE to tensor
    x: [seq_len, n_heads, head_dim]
    """
    seq_len, n_heads, head_dim = x.shape

    result = np.zeros_like(x)

    for pos_idx in range(seq_len):
        pos = pos_idx + position
        for head in range(n_heads):
            for pair_idx in range(head_dim // 2):
                # Calculate frequency
                exponent = (2 * pair_idx) / head_dim
                freq = 1.0 / (rope_base ** exponent)

                # Calculate angle
                theta = pos * freq
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                # Get dimension indices
                dim0 = pair_idx * 2
                dim1 = pair_idx * 2 + 1

                # Get input values
                x0 = x[pos_idx, head, dim0]
                x1 = x[pos_idx, head, dim1]

                # Apply rotation
                result[pos_idx, head, dim0] = x0 * cos_theta - x1 * sin_theta
                result[pos_idx, head, dim1] = x0 * sin_theta + x1 * cos_theta

    return result

def expand_kv_heads(kv, n_q_heads=32):
    """
    Expand K/V from 4 heads to 32 heads (GQA)
    kv: [seq_len, 4, head_dim]
    returns: [seq_len, 32, head_dim]
    """
    seq_len, n_kv_heads, head_dim = kv.shape
    expansion_factor = n_q_heads // n_kv_heads  # 8

    # Reshape and broadcast approach (matching TensorLogic)
    # [seq, 4, 64] -> [seq, 4, 1, 64] -> [seq, 4, 8, 64] -> [seq, 32, 64]
    kv_reshaped = kv[:, :, np.newaxis, :]  # [seq, 4, 1, 64]
    kv_expanded = np.broadcast_to(kv_reshaped, (seq_len, n_kv_heads, expansion_factor, head_dim))
    kv_final = kv_expanded.reshape(seq_len, n_q_heads, head_dim)

    return kv_final

def attention(Q, K, V):
    """
    Compute attention: softmax(Q @ K.T / sqrt(d_k)) @ V
    Q, K, V: [seq_len, n_heads, head_dim]
    """
    seq_len, n_heads, head_dim = Q.shape

    # Scaling factor
    scale = 1.0 / np.sqrt(head_dim)  # 1/sqrt(64) = 0.125

    # Compute attention scores: [seq, heads, seq]
    # For each head: Q[seq, d] @ K[seq, d].T = [seq, seq]
    scores = np.zeros((seq_len, n_heads, seq_len))
    for head in range(n_heads):
        Q_h = Q[:, head, :]  # [seq, d]
        K_h = K[:, head, :]  # [seq, d]
        scores[:, head, :] = Q_h @ K_h.T  # [seq, seq]

    # Scale
    scores = scores * scale

    print(f"\nAttention scores (before softmax):")
    print(f"  Shape: {scores.shape}")
    print(f"  Head 0, position 0: {scores[0, 0, :]}")
    print(f"  Max: {scores.max():.6f}, Min: {scores.min():.6f}")

    # Softmax over last dimension
    # Numerical stability: subtract max
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = scores_exp.sum(axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum

    print(f"\nAttention weights (after softmax):")
    print(f"  Shape: {attn_weights.shape}")
    print(f"  Head 0, position 0: {attn_weights[0, 0, :]}")
    print(f"  Sum (should be 1.0): {attn_weights[0, 0, :].sum():.6f}")

    # Weighted sum: [seq, heads, seq] @ [seq, heads, d] = [seq, heads, d]
    output = np.zeros_like(Q)
    for head in range(n_heads):
        weights_h = attn_weights[:, head, :]  # [seq, seq]
        V_h = V[:, head, :]  # [seq, d]
        output[:, head, :] = weights_h @ V_h  # [seq, d]

    return output

def rms_norm(x, weight, eps=1e-5):
    """RMS Normalization"""
    # x: [seq_len, hidden_dim]
    # weight: [hidden_dim]
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight

def linear(x, weight):
    """Linear layer: x @ W.T"""
    return x @ weight.T

def swiglu(x, W_gate, W_up, W_down):
    """SwiGLU FFN"""
    gate = linear(x, W_gate)
    up = linear(x, W_up)
    # SiLU: x * sigmoid(x)
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    intermediate = silu_gate * up
    return linear(intermediate, W_down)

def main():
    print("=" * 60)
    print("TinyLlama Layer 0 Reference Implementation (Python)")
    print("=" * 60)
    print()

    # Configuration
    n_heads = 32
    n_kv_heads = 4
    head_dim = 64
    hidden_dim = 2048

    print("Configuration:")
    print(f"  Q heads: {n_heads}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print()

    # For now, use dummy weights since we don't have a GGUF parser
    # In production, these would be loaded from the model file
    print("⚠️  WARNING: Using random weights for demonstration")
    print("    To get accurate comparison, need to load actual GGUF weights")
    print()

    # Create dummy weights (should be loaded from GGUF)
    np.random.seed(42)
    embedding = np.random.randn(1, hidden_dim).astype(np.float32) * 0.1

    W_q = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.01
    W_k = np.random.randn(n_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.01
    W_v = np.random.randn(n_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.01
    W_o = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.01

    attn_norm = np.ones(hidden_dim).astype(np.float32)
    ffn_norm = np.ones(hidden_dim).astype(np.float32)

    W_gate = np.random.randn(5632, hidden_dim).astype(np.float32) * 0.01
    W_up = np.random.randn(5632, hidden_dim).astype(np.float32) * 0.01
    W_down = np.random.randn(hidden_dim, 5632).astype(np.float32) * 0.01

    print("Step 1: Input embedding [1, 2048]")
    h = embedding
    print(f"  Shape: {h.shape}")
    print(f"  Sample values: {h[0, :5]}")
    print()

    # Attention norm
    print("Step 2: RMS normalization")
    h_norm = rms_norm(h, attn_norm)
    print(f"  Shape: {h_norm.shape}")
    print(f"  Sample values: {h_norm[0, :5]}")
    print()

    # Q, K, V projections
    print("Step 3: Q, K, V projections")
    Q = linear(h_norm, W_q)  # [1, 2048]
    K = linear(h_norm, W_k)  # [1, 256]
    V = linear(h_norm, W_v)  # [1, 256]

    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print()

    # Reshape to heads
    print("Step 4: Reshape to heads")
    Q_heads = Q.reshape(1, n_heads, head_dim)  # [1, 32, 64]
    K_heads = K.reshape(1, n_kv_heads, head_dim)  # [1, 4, 64]
    V_heads = V.reshape(1, n_kv_heads, head_dim)  # [1, 4, 64]

    print(f"  Q_heads: {Q_heads.shape}")
    print(f"  K_heads: {K_heads.shape}")
    print(f"  V_heads: {V_heads.shape}")
    print()

    # Apply RoPE
    print("Step 5: Apply RoPE")
    Q_rope = apply_rope(Q_heads, position=0)
    K_rope = apply_rope(K_heads, position=0)

    print(f"  Q_rope: {Q_rope.shape}")
    print(f"  K_rope: {K_rope.shape}")
    print(f"  Position 0 should preserve input (cos(0)=1, sin(0)=0)")
    print(f"  Q difference: {np.abs(Q_rope - Q_heads).max():.6f}")
    print(f"  K difference: {np.abs(K_rope - K_heads).max():.6f}")
    print()

    # GQA expansion
    print("Step 6: GQA expansion (4 heads -> 32 heads)")
    K_expanded = expand_kv_heads(K_rope)
    V_expanded = expand_kv_heads(V_heads)

    print(f"  K_expanded: {K_expanded.shape}")
    print(f"  V_expanded: {V_expanded.shape}")

    # Verify expansion
    print(f"  Verification: K_expanded[0, 0, :5] = {K_expanded[0, 0, :5]}")
    print(f"  Should match:  K_rope[0, 0, :5] = {K_rope[0, 0, :5]}")
    print(f"  Match: {np.allclose(K_expanded[0, 0, :], K_rope[0, 0, :])}")
    print()

    # Attention
    print("Step 7: Compute attention")
    attn_output = attention(Q_rope, K_expanded, V_expanded)
    print(f"  Output shape: {attn_output.shape}")
    print()

    # Reshape and project
    print("Step 8: Reshape and output projection")
    attn_reshaped = attn_output.reshape(1, hidden_dim)
    attn_proj = linear(attn_reshaped, W_o)

    print(f"  Reshaped: {attn_reshaped.shape}")
    print(f"  After projection: {attn_proj.shape}")
    print()

    # Residual
    print("Step 9: Residual connection")
    h1 = h + attn_proj
    print(f"  Shape: {h1.shape}")
    print()

    # FFN
    print("Step 10: FFN with SwiGLU")
    h1_norm = rms_norm(h1, ffn_norm)
    ffn_output = swiglu(h1_norm, W_gate, W_up, W_down)
    print(f"  FFN output: {ffn_output.shape}")
    print()

    # Final output
    print("Step 11: Final residual")
    h2 = h1 + ffn_output
    print(f"  Layer 0 output: {h2.shape}")
    print(f"  Sample values: {h2[0, :5]}")
    print()

    print("=" * 60)
    print("✅ Reference implementation complete")
    print()
    print("📝 Next steps:")
    print("  1. Load actual GGUF weights instead of random weights")
    print("  2. Run TensorLogic with same input and extract values")
    print("  3. Compare step-by-step to find where divergence occurs")
    print("=" * 60)

if __name__ == "__main__":
    main()
