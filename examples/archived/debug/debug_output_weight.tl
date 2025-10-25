// Debug: Check if output.weight contains NaN

main {
    print("=== Checking output.weight for NaN ===")
    print("")

    // Load model
    print("[1/2] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Loaded output.weight")
    print("        Shape:", shape(output_weight))
    print("")

    // Check values
    print("[2/2] Checking top values in output.weight...")
    let k = to_int(20.0)
    let dummy = print_top_k(output_weight, k)
    print("")

    print("Diagnosis:")
    print("  If output.weight contains NaN -> Model loading problem")
    print("  If output.weight is OK -> matmul() implementation problem")
    print("")
}
