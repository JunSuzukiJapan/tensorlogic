// Debug: Test each stage of the inference pipeline
// Find where values become zero

main {
    print("=== Testing Inference Pipeline Stage by Stage ===")
    print("")

    // Load model
    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("    ✓ Model loaded")
    print("")

    // Test with simple text
    print("[2] Creating tokens from text...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let tokens = tokenize(tokenizer, "Hello", false)
    print("    Token count:", len(tokens))
    print("")

    // Stage 1: Embedding
    print("[3] Stage 1: Embedding")
    let emb = embedding(emb_weight, tokens)
    print("    Embedding shape:", shape(emb))
    let emb_sample = slice(emb, 0.0, 0.0, 1.0)
    print("    Sample value at [0,0]:", emb_sample)
    print("")

    // Stage 2: Output projection (skip transformer layers for now)
    print("[4] Stage 2: Output Norm and Logits")
    let normed = rms_norm(emb, output_norm_weight)
    print("    After output_norm, shape:", shape(normed))
    let norm_sample = slice(normed, 0.0, 0.0, 1.0)
    print("    Sample value at [0,0]:", norm_sample)
    print("")

    print("[5] Stage 3: Final Linear Projection")
    let logits = linear(normed, output_weight)
    print("    Logits shape:", shape(logits))
    let logit_0 = slice(logits, 0.0, 0.0, 1.0)
    let logit_1 = slice(logits, 0.0, 1.0, 2.0)
    let logit_2 = slice(logits, 0.0, 2.0, 3.0)
    print("    Logit[0]:", logit_0)
    print("    Logit[1]:", logit_1)
    print("    Logit[2]:", logit_2)
    print("")

    let max_logit = max(logits)
    let min_logit = min(logits)
    print("    Logit range: min=", min_logit, ", max=", max_logit)
    print("")

    print("=== Pipeline Test Complete ===")
}
