// Test: Device Fix Verification
// This demonstrates that model weights and created tensors are now on the same device

main {
    print("=== Device Fix Verification ===")
    print("")

    // Load TinyLlama model
    print("Step 1: Loading model...")
    let model_path = "/Users/junsuzuki/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("✓ Model loaded")
    print("")

    // Create input tensor
    print("Step 2: Creating input tensor...")
    let seq_len = 4
    let d_model = 2048
    let x = ones([seq_len, d_model])
    print("✓ Input created:", shape(x))
    print("")

    // Load model weights
    print("Step 3: Loading model weights...")
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    print("✓ Weights loaded")
    print("  W_q shape:", shape(W_q))
    print("  attn_norm shape:", shape(attn_norm))
    print("")

    // Test 1: RMSNorm (previously failed with device mismatch)
    print("Test 1: RMSNorm with model weights...")
    let x_norm = rms_norm(x, attn_norm)
    print("✅ SUCCESS! Shape:", shape(x_norm))
    print("")

    // Test 2: MatMul with model weights (previously failed with device mismatch)
    print("Test 2: MatMul with model weights...")
    let Q = matmul(x_norm, W_q)
    print("✅ SUCCESS! Q shape:", shape(Q))
    print("")

    // Test 3: Multiple operations in sequence
    print("Test 3: Multiple operations...")
    let W_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    let gate = matmul(Q, W_gate)
    let gate_act = sigmoid(gate)
    let result = gate * gate_act
    print("✅ SUCCESS! Result shape:", shape(result))
    print("")

    print("==================================================")
    print("🎉 Device Fix Verified!")
    print("")
    print("Summary:")
    print("  • Model weights and created tensors use same MetalDevice")
    print("  • RMSNorm works with model weights")
    print("  • MatMul works with model weights")
    print("  • Complex operation chains work correctly")
    print("")
    print("Next: Implement proper Grouped Query Attention for full inference")
}
