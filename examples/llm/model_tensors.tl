// Test getting tensors from TinyLlama model

main {
    print("=== TinyLlama Model Tensors Test ===")
    print("")

    print("Step 1: Loading model...")
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("✓ Model loaded")
    print("")

    print("Step 2: Getting token embedding weights")
    let token_embd = get_tensor(model, "token_embd.weight")
    print("✓ token_embd.weight shape:", shape(token_embd))
    print("  Expected: [2048, 32000]")
    print("")

    print("Step 3: Getting layer 0 weights")
    let attn_q = get_tensor(model, "blk.0.attn_q.weight")
    print("✓ blk.0.attn_q.weight shape:", shape(attn_q))
    print("  Expected: [2048, 2048]")

    let attn_k = get_tensor(model, "blk.0.attn_k.weight")
    print("✓ blk.0.attn_k.weight shape:", shape(attn_k))
    print("  Expected: [2048, 256] (GQA)")

    let attn_v = get_tensor(model, "blk.0.attn_v.weight")
    print("✓ blk.0.attn_v.weight shape:", shape(attn_v))
    print("  Expected: [2048, 256] (GQA)")

    let attn_output = get_tensor(model, "blk.0.attn_output.weight")
    print("✓ blk.0.attn_output.weight shape:", shape(attn_output))
    print("  Expected: [2048, 2048]")
    print("")

    print("Step 4: Getting normalization weights")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    print("✓ blk.0.attn_norm.weight shape:", shape(attn_norm))
    print("  Expected: [2048]")

    let ffn_norm = get_tensor(model, "blk.0.ffn_norm.weight")
    print("✓ blk.0.ffn_norm.weight shape:", shape(ffn_norm))
    print("  Expected: [2048]")
    print("")

    print("Step 5: Getting FFN weights (SwiGLU)")
    let ffn_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    print("✓ blk.0.ffn_gate.weight shape:", shape(ffn_gate))
    print("  Expected: [2048, 5632]")

    let ffn_up = get_tensor(model, "blk.0.ffn_up.weight")
    print("✓ blk.0.ffn_up.weight shape:", shape(ffn_up))
    print("  Expected: [2048, 5632]")

    let ffn_down = get_tensor(model, "blk.0.ffn_down.weight")
    print("✓ blk.0.ffn_down.weight shape:", shape(ffn_down))
    print("  Expected: [5632, 2048]")
    print("")

    print("Step 6: Getting output weights")
    let output_norm = get_tensor(model, "output_norm.weight")
    print("✓ output_norm.weight shape:", shape(output_norm))
    print("  Expected: [2048]")

    let output = get_tensor(model, "output.weight")
    print("✓ output.weight shape:", shape(output))
    print("  Expected: [2048, 32000]")
    print("")

    print("✅ All model tensors loaded successfully!")
    print("")
    print("TinyLlama Architecture:")
    print("  • vocab_size: 32000")
    print("  • d_model: 2048")
    print("  • num_layers: 22")
    print("  • num_heads: 32")
    print("  • num_kv_heads: 4 (Grouped Query Attention)")
    print("  • head_dim: 64")
    print("  • ffn_dim: 5632")
}
