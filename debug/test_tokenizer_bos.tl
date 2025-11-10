// Test BOS token (Beginning Of Sequence) behavior

main {
    print("================================================================================")
    print("🔍 BOS Token Test - tokenize() Third Argument")
    print("================================================================================")
    print("")

    let tokenizer_path = "/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"

    print("[Test 1] tokenize(..., false) - No BOS")
    let tokens_without_bos = tokenize(tokenizer, prompt, false)
    print("  Token count:", len(tokens_without_bos))
    print("  Tokens:", tokens_without_bos)
    print("")

    print("[Test 2] tokenize(..., true) - With BOS")
    let tokens_with_bos = tokenize(tokenizer, prompt, true)
    print("  Token count:", len(tokens_with_bos))
    print("  Tokens:", tokens_with_bos)
    print("")

    print("================================================================================")
    print("📊 Analysis")
    print("================================================================================")
    print("")

    if len(tokens_without_bos) == 38 {
        print("  ✅ Without BOS: 38 tokens (matches HuggingFace)")
    }

    if len(tokens_with_bos) == 39 {
        print("  ✅ With BOS: 39 tokens (matches chat_demo_22layers.tl)")
        print("")
        print("  🎯 Explanation:")
        print("    - BOS token (likely token_id=1) is prepended")
        print("    - HuggingFace uses add_special_tokens=False")
        print("    - chat_demo_22layers.tl uses add_special_tokens=True")
        print("")
        print("  ⚠️  Problem:")
        print("    - BOS token shifts all positions by 1")
        print("    - RoPE (Rotary Position Embedding) will be misaligned")
        print("    - This could cause NaN in attention scores")
    }

    print("")
    print("================================================================================")
    print("✅ BOS token test complete")
    print("================================================================================")
}
