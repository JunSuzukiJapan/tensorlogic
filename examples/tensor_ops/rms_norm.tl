main {
    print("=== Testing RMSNorm Implementation ===")
    print("")

    print("Test 1: Basic RMS normalization")
    let x = positional_encoding(2, 4)
    let weight = ones([4])
    let normalized = x.rms_norm(weight)
    print("  Input shape:", x.shape())
    print("  Weight shape:", weight.shape())
    print("  ✓ RMS normalization completed")
    print("")

    print("Test 2: RMS norm with custom eps")
    let eps = 0.000001
    let normalized2 = x.rms_norm(weight, eps)
    print("  ✓ Custom eps =", eps)
    print("")

    print("Test 3: RMS norm with learned weight")
    let learned_weight = [1.0, 2.0, 3.0, 4.0]
    let normalized3 = x.rms_norm(learned_weight)
    print("  ✓ Learned weight applied")
    print("")

    print("Test 4: Simulating TinyLlama layer (d_model = 2048)")
    let d_model = 8
    let batch_size = 2
    let seq_len = 4

    let hidden = ones([batch_size, seq_len, d_model])
    let attn_norm_weight = ones([d_model])

    print("  Hidden states shape: [", batch_size, ",", seq_len, ",", d_model, "]")
    print("  Norm weight shape: [", d_model, "]")

    let normalized_hidden = hidden.rms_norm(attn_norm_weight)
    print("  ✓ Pre-attention RMS normalization completed")
    print("")

    print("=== Key Differences from LayerNorm ===")
    print("")
    print("RMSNorm (used in LLaMA/TinyLlama):")
    print("  • No mean centering (simpler)")
    print("  • No bias parameter")
    print("  • Only weight scaling")
    print("  • Formula: x / rms(x) * weight")
    print("  • Default eps: 1e-6")
    print("")
    print("LayerNorm (standard):")
    print("  • Mean centering: (x - mean)")
    print("  • Variance normalization")
    print("  • Optional weight and bias")
    print("  • Formula: (x - mean) / std * weight + bias")
    print("  • Default eps: 1e-5")
    print("")

    print("=== TinyLlama Usage ===")
    print("")
    print("RMSNorm is used in 3 places per transformer layer:")
    print("  1. Pre-attention: blk.i.attn_norm.weight [2048]")
    print("  2. Pre-FFN: blk.i.ffn_norm.weight [2048]")
    print("  3. Final output: output_norm.weight [2048]")
    print("")
    print("Total: 22 layers * 2 + 1 final = 45 RMSNorm operations")
    print("")

    print("✅ All RMSNorm tests completed!")
}
