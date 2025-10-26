#!/usr/bin/env python3
"""
Reference implementation with actual GGUF weights
Loads TinyLlama weights and computes layer 0 forward pass
"""

import numpy as np
import gguf
from pathlib import Path
import json

def dequantize_q4_0(data, shape):
    """Dequantize Q4_0 format to float32"""
    # Q4_0: 32 values per block, 4 bits per value
    # Block structure: f16 scale (2 bytes) + 16 bytes of quantized values (32 values)

    block_size = 32
    bytes_per_block = 2 + 16  # 2 bytes scale + 16 bytes quantized data
    expected_elements = int(np.prod(shape))
    n_blocks = (expected_elements + block_size - 1) // block_size

    print(f"  Q4_0 Debug: shape={shape}, expected_elements={expected_elements}")
    print(f"  Q4_0 Debug: data_bytes={len(data)}, n_blocks={n_blocks}, bytes_per_block={bytes_per_block}")

    result = np.zeros(expected_elements, dtype=np.float32)

    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data):
            break

        # Read scale as f16 (2 bytes)
        scale_bytes = data[offset:offset+2]
        scale = float(np.frombuffer(scale_bytes, dtype=np.float16)[0])

        # Read 16 bytes of 4-bit quantized values (32 values)
        quant_bytes = bytes(data[offset+2:offset+18])

        for i in range(16):
            byte = quant_bytes[i]
            # Each byte contains 2 4-bit values (low nibble, high nibble)
            # Layout: GROUPED [low0-15, high0-15], not interleaved
            val0 = (byte & 0x0F) - 8  # Low 4 bits, signed
            val1 = ((byte >> 4) & 0x0F) - 8  # High 4 bits, signed

            # Grouped layout: lower nibbles in first half, upper nibbles in second half
            idx0 = block_idx * block_size + i
            idx1 = block_idx * block_size + i + 16

            if idx0 < expected_elements:
                result[idx0] = float(val0) * scale
            if idx1 < expected_elements:
                result[idx1] = float(val1) * scale

    return result.reshape(shape)

def dequantize_q6_k(data, shape):
    """Dequantize Q6_K format to float32"""
    # Q6_K: 256 values per block, 6 bits per value
    # Block structure (210 bytes):
    #   - ql: 128 bytes (lower 4 bits of quantized values)
    #   - qh: 64 bytes (upper 2 bits of quantized values)
    #   - scales: 16 bytes (8-bit quantized scales)
    #   - d: 2 bytes (f16 super-block scale)

    block_size = 256
    bytes_per_block = 210  # 128 + 64 + 16 + 2

    expected_values = int(np.prod(shape))
    data_bytes = len(data)

    print(f"  Q6_K Debug: shape={shape}, expected_values={expected_values}, data_bytes={data_bytes}")
    print(f"  Q6_K Debug: bytes_per_block={bytes_per_block}, expected_blocks={expected_values // block_size}")
    print(f"  Q6_K Debug: actual_bytes_per_value={data_bytes / expected_values:.2f}")

    n_blocks = len(data) // bytes_per_block
    total_values = n_blocks * block_size

    result = np.zeros(total_values, dtype=np.float32)

    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block

        # Read block data
        ql = bytes(data[offset:offset+128])           # Lower 4 bits
        qh = bytes(data[offset+128:offset+192])       # Upper 2 bits
        scales = bytes(data[offset+192:offset+208])   # Scales (int8)
        d_bytes = data[offset+208:offset+210]         # Super-block scale (f16)

        d = float(np.frombuffer(d_bytes, dtype=np.float16)[0])

        # Dequantize each value
        for i in range(block_size):
            # Get 6-bit quantized value (4 bits from ql + 2 bits from qh)
            ql_idx = i // 2
            qh_idx = i // 4

            if ql_idx >= len(ql) or qh_idx >= len(qh):
                break

            # Lower 4 bits
            ql_byte = ql[ql_idx]
            if i % 2 == 0:
                q_low = ql_byte & 0x0F
            else:
                q_low = (ql_byte >> 4) & 0x0F

            # Upper 2 bits
            qh_byte = qh[qh_idx]
            shift = (i % 4) * 2
            q_high = (qh_byte >> shift) & 0x03

            # Combine to get 6-bit value (0-63)
            q = q_low | (q_high << 4)

            # Get scale for this group (16 values per scale)
            scale_idx = i // 16
            if scale_idx >= len(scales):
                scale_idx = len(scales) - 1

            # scales contains signed int8 values
            scale_byte = scales[scale_idx] if scale_idx < len(scales) else 1
            # Convert unsigned byte to signed int8
            scale = scale_byte if scale_byte < 128 else scale_byte - 256

            # Dequantize: (q - 32) * scale * d
            result[block_idx * block_size + i] = float(q - 32) * float(scale) * d / 16.0

    # Reshape to target shape
    total_expected = int(np.prod(shape))
    return result[:total_expected].reshape(shape)

def load_gguf_weights(model_path):
    """Load weights from GGUF file"""
    reader = gguf.GGUFReader(model_path)

    weights = {}

    for tensor in reader.tensors:
        name = str(tensor.name)
        shape = tensor.shape
        data = tensor.data
        tensor_type = tensor.tensor_type

        print(f"  Tensor: {name}, shape={shape}, type={tensor_type}, data_size={len(data)}")

        # Convert to numpy
        if tensor_type == gguf.GGMLQuantizationType.F32:
            weights[name] = np.frombuffer(data, dtype=np.float32).reshape(shape)
        elif tensor_type == gguf.GGMLQuantizationType.F16:
            weights[name] = np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
        elif tensor_type == gguf.GGMLQuantizationType.Q4_0:
            weights[name] = dequantize_q4_0(data, shape)
        elif tensor_type == gguf.GGMLQuantizationType.Q6_K:
            print(f"  WARNING: Skipping {name} temporarily (Q6_K needs proper implementation)")
            # weights[name] = dequantize_q6_k(data, shape)
            continue
        else:
            print(f"  Skipping {name} (type {tensor_type} not implemented)")
            continue

    return weights

def apply_rope(x, position=0, rope_base=10000.0):
    """Apply RoPE to tensor"""
    seq_len, n_heads, head_dim = x.shape
    result = np.zeros_like(x)

    for pos_idx in range(seq_len):
        pos = pos_idx + position
        for head in range(n_heads):
            for pair_idx in range(head_dim // 2):
                exponent = (2 * pair_idx) / head_dim
                freq = 1.0 / (rope_base ** exponent)
                theta = pos * freq
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                dim0 = pair_idx * 2
                dim1 = pair_idx * 2 + 1

                x0 = x[pos_idx, head, dim0]
                x1 = x[pos_idx, head, dim1]

                result[pos_idx, head, dim0] = x0 * cos_theta - x1 * sin_theta
                result[pos_idx, head, dim1] = x0 * sin_theta + x1 * cos_theta

    return result

def expand_kv_heads(kv, n_q_heads=32):
    """Expand K/V from 4 heads to 32 heads (GQA)"""
    seq_len, n_kv_heads, head_dim = kv.shape
    expansion_factor = n_q_heads // n_kv_heads

    kv_reshaped = kv[:, :, np.newaxis, :]
    kv_expanded = np.broadcast_to(kv_reshaped, (seq_len, n_kv_heads, expansion_factor, head_dim))
    kv_final = kv_expanded.reshape(seq_len, n_q_heads, head_dim)

    return kv_final

def attention(Q, K, V):
    """Compute attention"""
    seq_len, n_heads, head_dim = Q.shape
    scale = 1.0 / np.sqrt(head_dim)

    scores = np.zeros((seq_len, n_heads, seq_len))
    for head in range(n_heads):
        Q_h = Q[:, head, :]
        K_h = K[:, head, :]
        scores[:, head, :] = Q_h @ K_h.T

    scores = scores * scale

    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = scores_exp.sum(axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum

    output = np.zeros_like(Q)
    for head in range(n_heads):
        weights_h = attn_weights[:, head, :]
        V_h = V[:, head, :]
        output[:, head, :] = weights_h @ V_h

    return output, attn_weights

def rms_norm(x, weight, eps=1e-5):
    """RMS Normalization"""
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
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    intermediate = silu_gate * up
    return linear(intermediate, W_down)

def main():
    print("=" * 70)
    print("TinyLlama Layer 0 Forward Pass with Actual GGUF Weights")
    print("=" * 70)
    print()

    # Paths
    model_path = Path.home() / ".llm/models/tinyllama-1.1b-chat-q4_0.gguf"

    print(f"Loading model from: {model_path}")
    weights = load_gguf_weights(str(model_path))
    print(f"Loaded {len(weights)} tensors")
    print()

    # Configuration
    n_heads = 32
    n_kv_heads = 4
    head_dim = 64
    hidden_dim = 2048

    # Token embedding for "Hello" (token ID 15043)
    token_id = 15043
    embedding_table = weights['token_embd.weight']  # [vocab_size, hidden_dim]
    h = embedding_table[token_id:token_id+1, :]  # [1, 2048]

    print("Step 1: Token embedding")
    print(f"  Token ID: {token_id}")
    print(f"  Embedding shape: {h.shape}")
    print(f"  First 5 values: {h[0, :5]}")
    print()

    # Layer 0 weights
    W_q = weights['blk.0.attn_q.weight']
    W_k = weights['blk.0.attn_k.weight']
    W_v = weights['blk.0.attn_v.weight']
    W_o = weights['blk.0.attn_output.weight']
    attn_norm = weights['blk.0.attn_norm.weight']
    ffn_norm = weights['blk.0.ffn_norm.weight']
    W_gate = weights['blk.0.ffn_gate.weight']
    W_up = weights['blk.0.ffn_up.weight']
    W_down = weights['blk.0.ffn_down.weight']

    # Attention norm
    print("Step 2: RMS normalization")
    h_norm = rms_norm(h, attn_norm)
    print(f"  Shape: {h_norm.shape}")
    print(f"  First 5 values: {h_norm[0, :5]}")
    print()

    # Q, K, V projections
    print("Step 3: Q, K, V projections")
    Q = linear(h_norm, W_q)
    K = linear(h_norm, W_k)
    V = linear(h_norm, W_v)

    print(f"  Q: {Q.shape}, first 5: {Q[0, :5]}")
    print(f"  K: {K.shape}, first 5: {K[0, :5]}")
    print(f"  V: {V.shape}, first 5: {V[0, :5]}")
    print()

    # Reshape
    print("Step 4: Reshape to heads")
    Q_heads = Q.reshape(1, n_heads, head_dim)
    K_heads = K.reshape(1, n_kv_heads, head_dim)
    V_heads = V.reshape(1, n_kv_heads, head_dim)
    print(f"  Q_heads: {Q_heads.shape}")
    print(f"  K_heads: {K_heads.shape}")
    print(f"  V_heads: {V_heads.shape}")
    print()

    # RoPE
    print("Step 5: Apply RoPE (position=0)")
    Q_rope = apply_rope(Q_heads, position=0)
    K_rope = apply_rope(K_heads, position=0)

    print(f"  Q_rope: {Q_rope.shape}")
    print(f"  K_rope: {K_rope.shape}")
    print(f"  Q change: {np.abs(Q_rope - Q_heads).max():.10f}")
    print(f"  K change: {np.abs(K_rope - K_heads).max():.10f}")
    print()

    # GQA expansion
    print("Step 6: GQA expansion (4 -> 32 heads)")
    K_expanded = expand_kv_heads(K_rope)
    V_expanded = expand_kv_heads(V_heads)
    print(f"  K_expanded: {K_expanded.shape}")
    print(f"  V_expanded: {V_expanded.shape}")
    print()

    # Attention
    print("Step 7: Compute attention")
    attn_output, attn_weights = attention(Q_rope, K_expanded, V_expanded)
    print(f"  Output: {attn_output.shape}")
    print(f"  Attention weights (head 0): {attn_weights[0, 0, :]}")
    print()

    # Output projection
    print("Step 8: Output projection")
    attn_reshaped = attn_output.reshape(1, hidden_dim)
    attn_proj = linear(attn_reshaped, W_o)
    print(f"  After projection: {attn_proj.shape}")
    print(f"  First 5 values: {attn_proj[0, :5]}")
    print()

    # Residual
    print("Step 9: Residual connection")
    h1 = h + attn_proj
    print(f"  h1 shape: {h1.shape}")
    print(f"  First 5 values: {h1[0, :5]}")
    print()

    # FFN
    print("Step 10: FFN normalization")
    h1_norm = rms_norm(h1, ffn_norm)
    print(f"  h1_norm: {h1_norm.shape}")
    print()

    print("Step 11: SwiGLU FFN")
    ffn_output = swiglu(h1_norm, W_gate, W_up, W_down)
    print(f"  FFN output: {ffn_output.shape}")
    print(f"  First 5 values: {ffn_output[0, :5]}")
    print()

    # Final residual
    print("Step 12: Final residual")
    h2 = h1 + ffn_output
    print(f"  Layer 0 output: {h2.shape}")
    print(f"  First 5 values: {h2[0, :5]}")
    print()

    # Continue to output projection
    print("=" * 70)
    print("Computing final logits...")
    print("=" * 70)
    print()

    output_norm = weights['output_norm.weight']
    output_weight = weights['output.weight']

    # Output normalization
    h_final_norm = rms_norm(h2, output_norm)

    # Final projection to vocabulary
    logits = linear(h_final_norm, output_weight)  # [1, vocab_size]

    print(f"Logits shape: {logits.shape}")

    # Get top token
    top_token = np.argmax(logits[0])
    top_logit = logits[0, top_token]

    print(f"\nTop token: {top_token}")
    print(f"Top logit: {top_logit:.6f}")

    # Get top 5
    top_5_indices = np.argsort(logits[0])[-5:][::-1]
    print(f"\nTop 5 tokens:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"  {i}. Token {idx}: logit={logits[0, idx]:.6f}")

    print()
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)
    print()
    print("📝 Compare with TensorLogic output:")
    print("   Run: TL_DEBUG_SAMPLING=1 ./target/release/tl run examples/tests/simple_forward.tl")

if __name__ == "__main__":
    main()
