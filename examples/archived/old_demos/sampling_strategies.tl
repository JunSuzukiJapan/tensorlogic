// Demonstration of Different Sampling Strategies in TensorLogic
// Shows: Greedy, Temperature, and Top-p (Nucleus) Sampling

main {
    print("=== Sampling Strategies Demonstration ===")
    print("")

    // Load resources
    print("[1/2] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_weight = get_tensor(model, "token_embd.weight")
    let embed_table = transpose(embed_weight)
    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Loaded")
    print("")

    // Prepare prompt
    print("[2/2] Preparing prompt...")
    let prompt = "Hello"
    let tokens = tokenize(tokenizer, prompt, true)
    print("      Prompt:", prompt)
    print("      Initial tokens:", tokens)
    print("")

    print("=================================================")
    print("")

    // Strategy 1: Greedy (Deterministic)
    print("Strategy 1: GREEDY SAMPLING")
    print("  Description: Always pick highest probability token")
    print("  Characteristics: Deterministic, repetitive, safe")
    print("")

    let greedy_tokens = tokens

    let e1 = embedding(embed_table, greedy_tokens)
    let l1 = matmul(e1, output_weight)
    let t1 = to_int(argmax(l1))
    greedy_tokens = append(greedy_tokens, t1)

    let e2 = embedding(embed_table, greedy_tokens)
    let l2 = matmul(e2, output_weight)
    let t2 = to_int(argmax(l2))
    greedy_tokens = append(greedy_tokens, t2)

    let e3 = embedding(embed_table, greedy_tokens)
    let l3 = matmul(e3, output_weight)
    let t3 = to_int(argmax(l3))
    greedy_tokens = append(greedy_tokens, t3)

    let greedy_text = detokenize(tokenizer, greedy_tokens, true)
    print("  Generated:", greedy_text)
    print("  Tokens:", greedy_tokens)
    print("")

    // Strategy 2: Temperature = 0.7 (Slightly Random)
    print("Strategy 2: TEMPERATURE SAMPLING (T=0.7)")
    print("  Description: Controlled randomness, slight diversity")
    print("  Characteristics: Creative but coherent")
    print("")

    let temp_tokens = tokens

    let te1 = embedding(embed_table, temp_tokens)
    let tl1 = matmul(te1, output_weight)
    let tt1 = temperature_sample(tl1, 0.7)
    temp_tokens = append(temp_tokens, tt1)

    let te2 = embedding(embed_table, temp_tokens)
    let tl2 = matmul(te2, output_weight)
    let tt2 = temperature_sample(tl2, 0.7)
    temp_tokens = append(temp_tokens, tt2)

    let te3 = embedding(embed_table, temp_tokens)
    let tl3 = matmul(te3, output_weight)
    let tt3 = temperature_sample(tl3, 0.7)
    temp_tokens = append(temp_tokens, tt3)

    let temp_text = detokenize(tokenizer, temp_tokens, true)
    print("  Generated:", temp_text)
    print("  Tokens:", temp_tokens)
    print("")

    // Strategy 3: Temperature = 1.5 (Very Random)
    print("Strategy 3: HIGH TEMPERATURE (T=1.5)")
    print("  Description: High randomness, very diverse")
    print("  Characteristics: Creative, unpredictable, may be incoherent")
    print("")

    let hot_tokens = tokens

    let he1 = embedding(embed_table, hot_tokens)
    let hl1 = matmul(he1, output_weight)
    let ht1 = temperature_sample(hl1, 1.5)
    hot_tokens = append(hot_tokens, ht1)

    let he2 = embedding(embed_table, hot_tokens)
    let hl2 = matmul(he2, output_weight)
    let ht2 = temperature_sample(hl2, 1.5)
    hot_tokens = append(hot_tokens, ht2)

    let he3 = embedding(embed_table, hot_tokens)
    let hl3 = matmul(he3, output_weight)
    let ht3 = temperature_sample(hl3, 1.5)
    hot_tokens = append(hot_tokens, ht3)

    let hot_text = detokenize(tokenizer, hot_tokens, true)
    print("  Generated:", hot_text)
    print("  Tokens:", hot_tokens)
    print("")

    // Strategy 4: Top-p = 0.9 (Nucleus Sampling)
    print("Strategy 4: TOP-P SAMPLING (p=0.9)")
    print("  Description: Sample from top 90% probability mass")
    print("  Characteristics: Diverse but avoids unlikely tokens")
    print("")

    let nucleus_tokens = tokens

    let ne1 = embedding(embed_table, nucleus_tokens)
    let nl1 = matmul(ne1, output_weight)
    let nt1 = top_p_sample(nl1, 0.9)
    nucleus_tokens = append(nucleus_tokens, nt1)

    let ne2 = embedding(embed_table, nucleus_tokens)
    let nl2 = matmul(ne2, output_weight)
    let nt2 = top_p_sample(nl2, 0.9)
    nucleus_tokens = append(nucleus_tokens, nt2)

    let ne3 = embedding(embed_table, nucleus_tokens)
    let nl3 = matmul(ne3, output_weight)
    let nt3 = top_p_sample(nl3, 0.9)
    nucleus_tokens = append(nucleus_tokens, nt3)

    let nucleus_text = detokenize(tokenizer, nucleus_tokens, true)
    print("  Generated:", nucleus_text)
    print("  Tokens:", nucleus_tokens)
    print("")

    print("=================================================")
    print("")
    print("Summary:")
    print("  • Greedy: Safest, most predictable, can be repetitive")
    print("  • Temperature: Controls randomness (low=safe, high=creative)")
    print("  • Top-p: Balances diversity and quality")
    print("")
    print("Best Practices:")
    print("  • Creative writing: T=0.7-0.9 or top-p=0.9")
    print("  • Code generation: T=0.1-0.3 or greedy")
    print("  • Chat/dialogue: T=0.7 + top-p=0.9")
    print("  • Factual tasks: Greedy or T=0.1")
    print("")
    print("✅ Sampling strategies demonstration complete!")
}
