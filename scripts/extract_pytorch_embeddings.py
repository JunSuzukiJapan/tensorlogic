#!/usr/bin/env python3
"""
Extract embedding weights from PyTorch TinyLlama model and compare with TensorLogic.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

def main():
    print("=" * 80)
    print("PyTorch Embedding Weight Extraction")
    print("=" * 80)
    print()

    # Load model
    print("[1] Loading TinyLlama model from HuggingFace...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"    ✓ Loaded: {model_name}")
    print()

    # Get embedding weight
    print("[2] Extracting embedding weight...")
    embedding_weight = model.embed_tokens.weight.data  # [vocab_size, embedding_dim]
    print(f"    Embedding weight shape: {embedding_weight.shape}")
    print(f"    Dtype: {embedding_weight.dtype}")
    print()

    # Test BOS token (ID=1)
    print("[3] Testing BOS token (ID=1)...")
    bos_embedding = embedding_weight[1, :]  # Shape: [embedding_dim]
    bos_sum = bos_embedding.sum().item()
    print(f"    BOS embedding shape: {bos_embedding.shape}")
    print(f"    BOS embedding sum: {bos_sum}")
    print(f"    First 10 values:")
    for i in range(10):
        print(f"      [{i}]: {bos_embedding[i].item()}")
    print()

    # Test full prompt
    print("[4] Testing full prompt embedding...")
    prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello! How are you?</s>\n<|assistant|>\n"

    # Tokenize with BOS
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"    Num tokens: {len(tokens)}")
    print(f"    Token IDs: {tokens}")

    # Get embeddings for all tokens
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    embeddings = embedding_weight[token_tensor, :]  # Shape: [num_tokens, embedding_dim]
    embeddings_sum = embeddings.sum().item()

    print(f"    Embeddings shape: {embeddings.shape}")
    print(f"    Embeddings sum: {embeddings_sum}")
    print()

    # Individual token tests
    print("[5] Individual token embeddings...")
    test_tokens = [1, 529, 1000]  # BOS, first prompt token, random token
    for token_id in test_tokens:
        emb = embedding_weight[token_id, :]
        emb_sum = emb.sum().item()
        print(f"    Token {token_id}: sum = {emb_sum}")
    print()

    # Save embedding weight to file for detailed comparison
    print("[6] Saving embedding weight to file...")
    output_file = "/tmp/pytorch_embedding_weight.npz"
    np.savez(output_file,
             embedding_weight=embedding_weight.cpu().numpy(),
             bos_embedding=bos_embedding.cpu().numpy(),
             full_prompt_embeddings=embeddings.cpu().numpy(),
             token_ids=np.array(tokens))
    print(f"    ✓ Saved to: {output_file}")
    print()

    # Summary comparison with TensorLogic
    print("=" * 80)
    print("Summary for TensorLogic Comparison")
    print("=" * 80)
    print(f"BOS embedding sum (PyTorch):        {bos_sum}")
    print(f"BOS embedding sum (TensorLogic):    0.08184814453125")
    print(f"Difference:                         {bos_sum - 0.08184814453125}")
    print()
    print(f"Full prompt embedding sum (PyTorch):    {embeddings_sum}")
    print(f"Full prompt embedding sum (TensorLogic): 2.0859375")
    print(f"Difference:                             {embeddings_sum - 2.0859375}")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
