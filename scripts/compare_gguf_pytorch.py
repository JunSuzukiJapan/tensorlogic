#!/usr/bin/env python3
"""
Compare GGUF file directly with PyTorch model to identify discrepancies.
"""

import struct
import numpy as np
from gguf import GGUFReader

def read_gguf_embedding(gguf_path):
    """Read embedding weight from GGUF file."""
    print(f"Reading GGUF file: {gguf_path}")

    reader = GGUFReader(gguf_path)

    # Find token_embd.weight tensor
    embedding_tensor = None
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            embedding_tensor = tensor
            break

    if embedding_tensor is None:
        raise ValueError("token_embd.weight not found in GGUF file")

    print(f"  Tensor name: {embedding_tensor.name}")
    print(f"  Tensor type: {embedding_tensor.tensor_type}")
    print(f"  Shape: {embedding_tensor.shape}")
    print(f"  Data size: {len(embedding_tensor.data)} bytes")

    # Convert bytes to f16 array
    # GGUF stores f16 in little-endian
    num_elements = int(np.prod(embedding_tensor.shape))
    print(f"  Expected elements (from shape): {num_elements}")

    # Read as little-endian f16
    data_np = np.frombuffer(embedding_tensor.data, dtype=np.float16)
    print(f"  Actual elements (from data): {len(data_np)}")

    # Calculate actual shape from data
    embedding_dim = embedding_tensor.shape[0]  # 2048
    actual_vocab_size = len(data_np) // embedding_dim
    print(f"  Calculated vocab_size: {actual_vocab_size}")

    # Reshape using actual data size
    # GGUF stores in row-major as [embedding_dim, vocab_size]
    embedding_weight = data_np.reshape(embedding_dim, actual_vocab_size)
    print(f"  Reshaped to: {embedding_weight.shape}")

    # Transpose to match PyTorch: [vocab_size, embedding_dim]
    embedding_weight = embedding_weight.T
    print(f"  After transpose: {embedding_weight.shape}")

    return embedding_weight

def main():
    print("=" * 80)
    print("GGUF vs PyTorch Embedding Comparison")
    print("=" * 80)
    print()

    # Load GGUF embedding
    print("[1] Loading from GGUF file...")
    gguf_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    gguf_embedding = read_gguf_embedding(gguf_path)
    print()

    # Load PyTorch embedding
    print("[2] Loading from PyTorch model...")
    import torch
    from transformers import AutoModel

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    pytorch_embedding = model.embed_tokens.weight.data.cpu().numpy()
    print(f"  PyTorch shape: {pytorch_embedding.shape}")
    print(f"  PyTorch dtype: {pytorch_embedding.dtype}")
    print()

    # Compare BOS token (ID=1)
    print("[3] Comparing BOS token (ID=1) embedding...")
    bos_gguf = gguf_embedding[1, :]
    bos_pytorch = pytorch_embedding[1, :]

    print(f"  GGUF BOS sum:    {bos_gguf.sum()}")
    print(f"  PyTorch BOS sum: {bos_pytorch.sum()}")
    print(f"  Difference:      {abs(bos_gguf.sum() - bos_pytorch.sum())}")
    print()

    print("  First 10 values comparison:")
    print("  Index | GGUF         | PyTorch      | Diff")
    print("  " + "-" * 50)
    for i in range(10):
        diff = abs(bos_gguf[i] - bos_pytorch[i])
        match = "✓" if diff < 1e-5 else "✗"
        print(f"  [{i:2d}]  | {bos_gguf[i]:12.6f} | {bos_pytorch[i]:12.6f} | {diff:10.8f} {match}")
    print()

    # Full embedding comparison
    print("[4] Full embedding weight comparison...")

    # Check if shapes match
    if gguf_embedding.shape != pytorch_embedding.shape:
        print(f"  ✗ Shape mismatch!")
        print(f"    GGUF:    {gguf_embedding.shape}")
        print(f"    PyTorch: {pytorch_embedding.shape}")
        return

    # Element-wise comparison
    diff = np.abs(gguf_embedding - pytorch_embedding)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  Shape: {gguf_embedding.shape}")
    print(f"  Max difference:  {max_diff}")
    print(f"  Mean difference: {mean_diff}")
    print(f"  Num exact matches: {(diff < 1e-7).sum()} / {diff.size}")
    print()

    # Check if they're identical
    if max_diff < 1e-5:
        print("  ✓ GGUF and PyTorch embeddings are virtually identical!")
    else:
        print("  ✗ Significant differences found!")

        # Find indices with largest differences
        print("\n  Top 10 differences:")
        flat_diff = diff.flatten()
        top_indices = np.argsort(flat_diff)[-10:][::-1]

        for idx in top_indices:
            token_id = idx // gguf_embedding.shape[1]
            emb_dim = idx % gguf_embedding.shape[1]
            gguf_val = gguf_embedding[token_id, emb_dim]
            pytorch_val = pytorch_embedding[token_id, emb_dim]
            print(f"    Token {token_id}, Dim {emb_dim}: GGUF={gguf_val:.6f}, PyTorch={pytorch_val:.6f}, Diff={flat_diff[idx]:.6f}")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
