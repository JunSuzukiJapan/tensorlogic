// Check norm weights in Q4_0 model to see if issue is specific to F16 file

main {
    print("================================================================================")
    print("🔍 Checking Q4_0 Model Norm Weights")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Check output norm weight
    let output_norm = get_tensor(model, "output_norm.weight")
    print("Output Norm Weight (Q4_0):")
    print("  Sum:", sum(output_norm))
    print("  Average:", sum(output_norm), "/ 2048 =", sum(output_norm) / 2048.0)
    print("")

    // Check layer 0 attention norm weight
    let layer0_attn_norm = model.blk[0].attn_norm.weight
    print("Layer 0 Attention Norm Weight (Q4_0):")
    print("  Sum:", sum(layer0_attn_norm))
    print("  Average:", sum(layer0_attn_norm), "/ 2048 =", sum(layer0_attn_norm) / 2048.0)
    print("")

    // Check layer 0 ffn norm weight
    let layer0_ffn_norm = model.blk[0].ffn_norm.weight
    print("Layer 0 FFN Norm Weight (Q4_0):")
    print("  Sum:", sum(layer0_ffn_norm))
    print("  Average:", sum(layer0_ffn_norm), "/ 2048 =", sum(layer0_ffn_norm) / 2048.0)
    print("")

    print("================================================================================")
}
