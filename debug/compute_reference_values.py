#!/usr/bin/env python3
"""
Compute reference intermediate values using HuggingFace Transformers
This provides ground truth values to debug TensorLogic's NaN issues
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def main():
    print("=" * 80)
    print("🔍 Computing Reference Values with HuggingFace Transformers")
    print("=" * 80)
    print()

    # Setup
    print("[1/5] Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model = model.to("cpu")
    model.eval()
    print("      ✓ Model loaded (f16, CPU)")
    print()

    # Create prompt
    print("[2/5] Creating prompt...")
    system_prompt = "You are a friendly chatbot."
    user_message = "Hello! How are you?"

    # TinyLlama chat template
    prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
    print(f"      Prompt: {repr(prompt)}")

    # Tokenize
    print()
    print("[3/5] Tokenizing...")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]

    tokens = input_ids[0].tolist()
    print(f"      Tokens: {tokens}")
    print(f"      Num tokens: {len(tokens)}")
    print()

    # Forward pass with hooks to capture intermediate values
    print("[4/5] Running forward pass and capturing intermediate values...")
    print()

    intermediate_values = {}

    # Hook to capture embedding output
    def hook_embedding(module, input, output):
        intermediate_values['embedding'] = output.detach().cpu().float()

    # Hook to capture layer 0 outputs
    def hook_layer_0_attn(module, input, output):
        if isinstance(output, tuple):
            intermediate_values['layer_0_attn_output'] = output[0].detach().cpu().float()
        else:
            intermediate_values['layer_0_attn_output'] = output.detach().cpu().float()

    def hook_layer_0_mlp(module, input, output):
        intermediate_values['layer_0_mlp_output'] = output.detach().cpu().float()

    def hook_layer_0_output(module, input, output):
        if isinstance(output, tuple):
            intermediate_values['layer_0_output'] = output[0].detach().cpu().float()
        else:
            intermediate_values['layer_0_output'] = output.detach().cpu().float()

    # Register hooks
    embedding_hook = model.model.embed_tokens.register_forward_hook(hook_embedding)
    layer_0_attn_hook = model.model.layers[0].self_attn.register_forward_hook(hook_layer_0_attn)
    layer_0_mlp_hook = model.model.layers[0].mlp.register_forward_hook(hook_layer_0_mlp)
    layer_0_hook = model.model.layers[0].register_forward_hook(hook_layer_0_output)

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Remove hooks
    embedding_hook.remove()
    layer_0_attn_hook.remove()
    layer_0_mlp_hook.remove()
    layer_0_hook.remove()

    # Get logits
    logits = outputs.logits[0, -1, :].cpu().float()  # Last token logits

    print("  [Step 1] Embedding")
    emb = intermediate_values['embedding'][0]  # [seq_len, d_model]
    print(f"    Shape: {emb.shape}")
    print(f"    embedding[0, 0:3] = {emb[0, 0:3].tolist()}")
    print()

    print("  [Step 2] Layer 0 - Attention output")
    if 'layer_0_attn_output' in intermediate_values:
        attn_out = intermediate_values['layer_0_attn_output'][0]
        print(f"    Shape: {attn_out.shape}")
        print(f"    attn_output[0, 0:3] = {attn_out[0, 0:3].tolist()}")
    print()

    print("  [Step 3] Layer 0 - MLP output")
    if 'layer_0_mlp_output' in intermediate_values:
        mlp_out = intermediate_values['layer_0_mlp_output'][0]
        print(f"    Shape: {mlp_out.shape}")
        print(f"    mlp_output[0, 0:3] = {mlp_out[0, 0:3].tolist()}")
    print()

    print("  [Step 4] Layer 0 - Final output")
    layer_0_out = intermediate_values['layer_0_output'][0]
    print(f"    Shape: {layer_0_out.shape}")
    print(f"    layer_0_output[0, 0:3] = {layer_0_out[0, 0:3].tolist()}")
    print()

    print("  [Step 5] All hidden states")
    hidden_states = outputs.hidden_states
    print(f"    Number of layers: {len(hidden_states)}")

    # Layer 21 output (before final norm)
    layer_21_out = hidden_states[-1][0, -1, :].cpu().float()
    print(f"    layer_21_output[-1, 0:3] = {layer_21_out[0:3].tolist()}")
    print()

    print("  [Step 6] Final logits")
    print(f"    Logits shape: {logits.shape}")
    print(f"    logits[0:3] = {logits[0:3].tolist()}")
    print()

    # Get top 10 logits
    top_k = torch.topk(logits, k=10)
    top_values = top_k.values.tolist()
    top_indices = top_k.indices.tolist()

    print("    Top 10 logits:")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        token = tokenizer.decode([idx])
        print(f"      #{i+1}: token_id={idx} logit={val:.6f} token={repr(token)}")
    print()

    # Save to JSON
    print("[5/5] Saving results to document...")

    results = {
        "prompt": prompt,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "intermediate_values": {
            "embedding": {
                "sample_0_0_3": emb[0, 0:3].tolist(),
                "shape": list(emb.shape)
            },
            "layer_0": {
                "attn_output": {
                    "sample_0_0_3": attn_out[0, 0:3].tolist() if 'layer_0_attn_output' in intermediate_values else None,
                    "shape": list(attn_out.shape) if 'layer_0_attn_output' in intermediate_values else None
                },
                "mlp_output": {
                    "sample_0_0_3": mlp_out[0, 0:3].tolist() if 'layer_0_mlp_output' in intermediate_values else None,
                    "shape": list(mlp_out.shape) if 'layer_0_mlp_output' in intermediate_values else None
                },
                "layer_output": {
                    "sample_0_0_3": layer_0_out[0, 0:3].tolist(),
                    "shape": list(layer_0_out.shape)
                }
            },
            "layer_21_output": {
                "sample_last_0_3": layer_21_out[0:3].tolist()
            },
            "logits": {
                "sample_0_3": logits[0:3].tolist(),
                "top_10": [
                    {
                        "rank": i + 1,
                        "token_id": int(idx),
                        "logit": float(val),
                        "token": tokenizer.decode([idx])
                    }
                    for i, (idx, val) in enumerate(zip(top_indices, top_values))
                ]
            }
        }
    }

    output_path = "/Users/junsuzuki/Program/Rust/tensorlogic/claudedocs/transformers_reference_values.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"      ✓ Results saved to: {output_path}")
    print()

    print("=" * 80)
    print("✅ Reference values computed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
