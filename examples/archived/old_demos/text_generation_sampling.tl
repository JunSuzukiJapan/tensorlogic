main {
    print("=== Text Generation with Sampling Strategies ===")
    print("")

    let vocab_size = 1000
    print("Vocabulary size:", vocab_size)
    print("")

    print("Step 1: Generate random logits (simulating model output)")
    let logits_2d = positional_encoding(1, vocab_size)
    let logits = flatten(logits_2d)
    print("  ✓ Logits shape: [", vocab_size, "]")
    print("")

    print("Step 2: Top-k Sampling (k=50)")
    print("  Strategy: Sample from top 50 most likely tokens")
    let k = 50
    let top_k_logits = top_k(logits, k)
    print("  ✓ Applied top-k filter (k =", k, ")")

    let top_k_probs = softmax(top_k_logits, 0)
    let top_k_token = sample(top_k_probs)
    print("  ✓ Sampled token ID:", top_k_token)
    print("")

    print("Step 3: Top-p Sampling (p=0.9)")
    print("  Strategy: Sample from smallest set with cumulative prob >= 90%")
    let p = 0.9
    let top_p_logits = top_p(logits, p)
    print("  ✓ Applied top-p (nucleus) filter (p =", p, ")")

    let top_p_probs = softmax(top_p_logits, 0)
    let top_p_token = sample(top_p_probs)
    print("  ✓ Sampled token ID:", top_p_token)
    print("")

    print("Step 4: Combined Sampling (Top-k + Top-p)")
    print("  This is the recommended approach for chat models")
    let combined_logits_1 = top_k(logits, k)
    let combined_logits_2 = top_p(combined_logits_1, p)
    let combined_probs = softmax(combined_logits_2, 0)
    let combined_token = sample(combined_probs)
    print("  ✓ Applied top-k (", k, ") + top-p (", p, ")")
    print("  ✓ Sampled token ID:", combined_token)
    print("")

    print("=== Sampling Comparison ===")
    print("")
    print("Method                | Characteristics")
    print("----------------------|------------------------------------------")
    print("Greedy (argmax)      | Deterministic, always picks best token")
    print("Top-k (k=50)         | Limits to top k tokens, balances quality")
    print("Top-p (p=0.9)        | Adaptive nucleus, quality over diversity")
    print("Temperature (T<1.0)  | More focused and deterministic")
    print("Temperature (T>1.0)  | More random and creative")
    print("k=50 + p=0.9         | Best for chat (quality + diversity)")
    print("")

    print("=== Production LLM Generation Loop ===")
    print("")
    print("Typical hyperparameters for chat:")
    print("  • Temperature: 0.7 (balanced)")
    print("  • Top-k: 50 (quality control)")
    print("  • Top-p: 0.9 (nucleus sampling)")
    print("  • Max tokens: 100-2048")
    print("")
    print("Pseudocode:")
    print("  for i in 1..max_new_tokens:")
    print("    logits = model.forward(input_ids)")
    print("    next_logits = logits[-1, :] / temperature")
    print("    filtered = top_p(top_k(next_logits, k), p)")
    print("    probs = softmax(filtered)")
    print("    next_token = sample(probs)")
    print("    input_ids = append(input_ids, next_token)")
    print("    if next_token == EOS_TOKEN: break")
    print("")

    print("✅ Sampling strategies demonstration complete!")
}
