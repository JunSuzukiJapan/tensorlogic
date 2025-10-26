// Dump GGUF weights to verify dequantization
// This helps us confirm TensorLogic loads Q4_0 weights correctly

main {
    print("=== GGUF Weight Dump ===\n")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Get token embedding weight (Q4_0 quantized)
    let token_embd = get_tensor(model, "token_embd.weight")
    print("token_embd.weight shape: ", shape(token_embd), "\n")

    // Get a single embedding (token ID 15043 = "Hello")
    let hello_embd = embedding(token_embd, [15043.0])
    print("Embedding for token 15043 ('Hello'):")
    print("  Shape: ", shape(hello_embd))
    print("  (Cannot easily print full 2048 values in TensorLogic)")
    print("")

    // Get Layer 0 attention Q weight
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    print("blk.0.attn_q.weight shape: ", shape(W_q_0), "\n")

    // Get Layer 0 attention norm weight (F32, not quantized)
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    print("blk.0.attn_norm.weight shape: ", shape(attn_norm_0), "\n")

    print("=== Verification ===")
    print("If TensorLogic loads GGUF correctly:")
    print("  1. Shapes should match expected dimensions")
    print("  2. Values should not be all zeros or NaN")
    print("  3. Q4_0 dequantization should produce reasonable f16 values")
    print("")
    print("To dump actual values, we need to:")
    print("  1. Write weights to a file in TensorLogic")
    print("  2. Or use Rust test to access tensor internals")
}
