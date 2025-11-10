#!/usr/bin/env python3
"""
Candle-style 1-layer transformer test for TinyLlama
Compare with TensorLogic to debug GPU sync issues
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def main():
    print("=" * 80)
    print("Candle-style 1-Layer Transformer Test")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading TinyLlama model...")
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
    print("[2/4] Creating prompt...")
    system_prompt = "You are a friendly chatbot."
    user_message = "Hello! How are you?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"      Prompt: {repr(chat_prompt[:50])}...")

    # Tokenize (no BOS token for consistency with TensorLogic)
    input_ids = tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt")
    print(f"      Tokens: {input_ids.shape[1]} tokens")
    print(f"      Token IDs: {input_ids[0][:10].tolist()}...")
    print()

    # Run forward pass - ONLY LAYER 0
    print("[3/4] Running forward pass (Layer 0 only)...")

    with torch.no_grad():
        # Embedding
        embeddings = model.model.embed_tokens(input_ids)
        print(f"      Embedding shape: {embeddings.shape}")
        print(f"      Embedding[0,0:3]: {embeddings[0, 0, :3].tolist()}")
        print(f"      Embedding sum: {embeddings.sum().item():.6f}")
        print()

        # Layer 0 - Use the full model forward pass
        # But extract output after layer 0 only
        # This is simpler than manual implementation
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Get hidden state after layer 0
        # hidden_states[0] = embedding
        # hidden_states[1] = after layer 0
        hidden = outputs.hidden_states[1]

        print(f"      Layer 0 output shape: {hidden.shape}")
        print(f"      Layer 0 output[0,0:3]: {hidden[0, 0, :3].tolist()}")
        print(f"      Layer 0 output sum: {hidden.sum().item():.6f}")
        print()

        # Final norm and logits
        hidden = model.model.norm(hidden)
        print(f"      Final norm shape: {hidden.shape}")
        print(f"      Final norm[0,0:3]: {hidden[0, 0, :3].tolist()}")
        print(f"      Final norm sum: {hidden.sum().item():.6f}")
        print()

        logits = model.lm_head(hidden)
        print(f"      Logits shape: {logits.shape}")

        # Get last token logits
        last_logits = logits[0, -1, :]
        print(f"      Last token logits shape: {last_logits.shape}")
        print(f"      Last token logits sum: {last_logits.sum().item():.6f}")
        print()

        # Top 10 predictions
        top_k = 10
        top_values, top_indices = torch.topk(last_logits, top_k)

        print(f"      Top {top_k} predictions:")
        for rank, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1):
            token = tokenizer.decode([idx])
            print(f"        #{rank}: token_id={idx:5d} logit={val:8.4f} token={repr(token)}")
        print()

    # Save results
    print("[4/4] Saving results...")

    results = {
        "model": model_name,
        "prompt": chat_prompt,
        "num_tokens": input_ids.shape[1],
        "token_ids": input_ids[0].tolist(),
        "embedding": {
            "shape": list(embeddings.shape),
            "sum": float(embeddings.sum().item()),
            "first_3": embeddings[0, 0, :3].tolist()
        },
        "layer_0": {
            "output_shape": list(hidden.shape),
            "output_sum": float(hidden.sum().item()),
            "output_first_3": hidden[0, 0, :3].tolist()
        },
        "final_norm": {
            "shape": list(hidden.shape),
            "sum": float(hidden.sum().item()),
            "first_3": hidden[0, 0, :3].tolist()
        },
        "logits": {
            "shape": list(logits.shape),
            "last_token_sum": float(last_logits.sum().item())
        },
        "top_predictions": [
            {
                "rank": rank,
                "token_id": int(idx),
                "logit": float(val),
                "token": tokenizer.decode([idx])
            }
            for rank, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1)
        ]
    }

    output_path = "/Users/junsuzuki/Program/Rust/tensorlogic/claudedocs/candle_1layer_reference.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"      ✓ Results saved to: {output_path}")
    print()

    print("=" * 80)
    print("✅ 1-Layer Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
