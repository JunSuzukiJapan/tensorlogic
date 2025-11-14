// Check actual values in norm weights

main {
    print("================================================================================")
    print("🔍 Norm Weight Values Analysis")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Check output norm weight values
    let output_norm = get_tensor(model, "output_norm.weight")
    print("Output Norm Weight:")
    print("  Sum:", sum(output_norm))
    print("  Average:", sum(output_norm), "/ 2048 =", sum(output_norm) / 2048.0)
    print("")

    // Check layer 0 attention norm weight values
    let layer0_attn_norm = model.blk[0].attn_norm.weight
    print("Layer 0 Attention Norm Weight:")
    print("  Sum:", sum(layer0_attn_norm))
    print("  Average:", sum(layer0_attn_norm), "/ 2048 =", sum(layer0_attn_norm) / 2048.0)
    print("")

    // Check layer 0 ffn norm weight values
    let layer0_ffn_norm = model.blk[0].ffn_norm.weight
    print("Layer 0 FFN Norm Weight:")
    print("  Sum:", sum(layer0_ffn_norm))
    print("  Average:", sum(layer0_ffn_norm), "/ 2048 =", sum(layer0_ffn_norm) / 2048.0)
    print("")

    // Compare: manually load blk.0.attn_norm.weight with get_tensor
    let manual_load = get_tensor(model, "blk.0.attn_norm.weight")
    print("Manual Load blk.0.attn_norm.weight:")
    print("  Sum:", sum(manual_load))
    print("  Average:", sum(manual_load), "/ 2048 =", sum(manual_load) / 2048.0)
    print("")

    print("================================================================================")
}
