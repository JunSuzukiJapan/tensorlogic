main {
    print("=== Testing Layer Weight Shapes ===")

    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("\n--- Layer 0 Attention Weights ---")
    let wq = get_tensor(model, "blk.0.attn_q.weight")
    print("blk.0.attn_q.weight:", shape(wq))

    let wk = get_tensor(model, "blk.0.attn_k.weight")
    print("blk.0.attn_k.weight:", shape(wk))

    let wv = get_tensor(model, "blk.0.attn_v.weight")
    print("blk.0.attn_v.weight:", shape(wv))

    let wo = get_tensor(model, "blk.0.attn_output.weight")
    print("blk.0.attn_output.weight:", shape(wo))

    print("\n--- Layer 0 FFN Weights ---")
    let wgate = get_tensor(model, "blk.0.ffn_gate.weight")
    print("blk.0.ffn_gate.weight:", shape(wgate))

    let wup = get_tensor(model, "blk.0.ffn_up.weight")
    print("blk.0.ffn_up.weight:", shape(wup))

    let wdown = get_tensor(model, "blk.0.ffn_down.weight")
    print("blk.0.ffn_down.weight:", shape(wdown))

    print("\nExpected shapes for d_model=2048, d_ff=5632:")
    print("  Q,K,V weights: [2048, 2048] (for GQA)")
    print("  Output weight: [2048, 2048]")
    print("  Gate/Up weights: [2048, 5632]")
    print("  Down weight: [5632, 2048]")
}
