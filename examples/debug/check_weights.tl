// Check weight statistics to understand RMS norm behavior

main {
    print("================================================================================")
    print("🔍 Weight Statistics Analysis")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Check output norm weight
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let weight_sum = sum(output_norm_weight)

    print("Output Norm Weight:")
    print("  Sum:", weight_sum)
    print("  Size: 2048")
    print("  Average:", weight_sum, "/ 2048 =", weight_sum / 2048.0)
    print("")

    // Check layer 0 attention norm weight
    let layer0_attn_norm = model.blk[0].attn_norm.weight
    let layer0_sum = sum(layer0_attn_norm)

    print("Layer 0 Attention Norm Weight:")
    print("  Sum:", layer0_sum)
    print("  Size: 2048")
    print("  Average:", layer0_sum, "/ 2048 =", layer0_sum / 2048.0)
    print("")

    print("================================================================================")
}
