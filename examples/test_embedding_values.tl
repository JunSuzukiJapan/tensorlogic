// Embedding weight individual values comparison

main {
    print("================================================================================")
    print("Embedding Weight Individual Values Test")
    print("================================================================================")
    print("")

    // Load model
    print("[1] Loading f16 model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("    Embedding weight shape:", shape(emb_weight))
    print("")

    // PyTorch reference values for BOS token (ID=1)
    // From PyTorch:
    //   [0]: -0.00109100341796875
    //   [1]: 0.00193023681640625
    //   [2]: -0.0016632080078125
    //   [3]: 0.0037689208984375
    //   [4]: 0.00107574462890625
    //   [5]: 0.0037841796875
    //   [6]: 0.000682830810546875
    //   [7]: -0.00017261505126953125
    //   [8]: -0.0010833740234375
    //   [9]: -0.000896453857421875
    // Sum: 0.052734375

    print("[2] Testing individual BOS token (ID=1) embedding values...")
    print("    PyTorch reference (first 10 dims):")
    print("      [0]: -0.001091")
    print("      [1]:  0.001930")
    print("      [2]: -0.001663")
    print("      [3]:  0.003769")
    print("      [4]:  0.001076")
    print("      [5]:  0.003784")
    print("      [6]:  0.000683")
    print("      [7]: -0.000173")
    print("      [8]: -0.001083")
    print("      [9]: -0.000896")
    print("")

    print("    TensorLogic values (accessing via embedding):")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let bos_tokens = tokenize(tokenizer, "", true)  // Just BOS
    let bos_emb = embedding(emb_weight, bos_tokens)  // [1, 2048]

    // Access individual dimensions of BOS embedding
    // Note: TL doesn't support tensor[i, j] yet, so we use sum of slices
    print("      (BOS embedding shape:", shape(bos_emb), ")")
    print("      BOS embedding sum:", sum(bos_emb))
    print("      Expected sum: 0.0527")
    print("")

    print("================================================================================")
    print("✅ Test Complete")
    print("================================================================================")
}
