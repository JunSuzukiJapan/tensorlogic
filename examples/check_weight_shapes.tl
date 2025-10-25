// Check model weight shapes - find transpose issue

main {
    print("=== Weight Shape Investigation ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("[Layer 0 Weights]")
    print("")

    print("Q projection:")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    print("  W_q shape:", shape(W_q))
    print("  Expected: [2048, 2048] (d_model, d_model)")
    print("")

    print("K projection (GQA - should be smaller):")
    let W_k = get_tensor(model, "blk.0.attn_k.weight")
    print("  W_k shape:", shape(W_k))
    print("  Expected: [2048, 256] (d_model, n_kv_heads * head_dim)")
    print("  Actual GQA config: 4 KV heads * 64 dim = 256")
    print("")

    print("V projection (GQA - same as K):")
    let W_v = get_tensor(model, "blk.0.attn_v.weight")
    print("  W_v shape:", shape(W_v))
    print("  Expected: [2048, 256]")
    print("")

    print("Output projection:")
    let W_o = get_tensor(model, "blk.0.attn_output.weight")
    print("  W_o shape:", shape(W_o))
    print("  Expected: [2048, 2048]")
    print("")

    print("[FFN Weights]")
    print("")

    print("Gate:")
    let W_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    print("  W_gate shape:", shape(W_gate))
    print("  Expected: [2048, 5632] (d_model, ffn_dim)")
    print("")

    print("Up:")
    let W_up = get_tensor(model, "blk.0.ffn_up.weight")
    print("  W_up shape:", shape(W_up))
    print("  Expected: [2048, 5632]")
    print("")

    print("Down:")
    let W_down = get_tensor(model, "blk.0.ffn_down.weight")
    print("  W_down shape:", shape(W_down))
    print("  Expected: [5632, 2048] (ffn_dim, d_model)")
    print("")

    print("=== Analysis ===")
    print("TinyLlama 1.1B uses Grouped Query Attention (GQA):")
    print("  - 32 query heads")
    print("  - 4 KV heads (8:1 ratio)")
    print("  - 64 dim per head")
    print("  - Q: 32 * 64 = 2048")
    print("  - K/V: 4 * 64 = 256")
    print("")
    print("If weights are transposed, we need to transpose them before matmul!")
}
