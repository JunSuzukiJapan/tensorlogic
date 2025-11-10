// Test tokenizer round-trip: tokenize -> detokenize
// Verify that text can be converted to tokens and back to the same text

main {
    print("=== Tokenizer Round-Trip Test ===")
    print("")

    // Load tokenizer
    print("[1/3] Loading tokenizer...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")
    print("")

    // Test cases
    print("[2/3] Testing various inputs...")
    print("")

    // Test 1: Simple word
    print("Test 1: Simple word")
    let text1 = "Hello"
    print("  Original: \"", text1, "\"")
    let tokens1 = tokenize(tokenizer, text1, false)
    print("  Tokens:  ", tokens1)
    let decoded1 = detokenize(tokenizer, tokens1, false)
    print("  Decoded:  \"", decoded1, "\"")
    if text1 == decoded1 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 2: Multiple words
    print("Test 2: Multiple words")
    let text2 = "Hello world"
    print("  Original: \"", text2, "\"")
    let tokens2 = tokenize(tokenizer, text2, false)
    print("  Tokens:  ", tokens2)
    let decoded2 = detokenize(tokenizer, tokens2, false)
    print("  Decoded:  \"", decoded2, "\"")
    if text2 == decoded2 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 3: Sentence with punctuation
    print("Test 3: Sentence with punctuation")
    let text3 = "How are you?"
    print("  Original: \"", text3, "\"")
    let tokens3 = tokenize(tokenizer, text3, false)
    print("  Tokens:  ", tokens3)
    let decoded3 = detokenize(tokenizer, tokens3, false)
    print("  Decoded:  \"", decoded3, "\"")
    if text3 == decoded3 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 4: Longer sentence
    print("Test 4: Longer sentence")
    let text4 = "The quick brown fox jumps over the lazy dog."
    print("  Original: \"", text4, "\"")
    let tokens4 = tokenize(tokenizer, text4, false)
    print("  Tokens:  ", tokens4)
    print("  Token count:", len(tokens4))
    let decoded4 = detokenize(tokenizer, tokens4, false)
    print("  Decoded:  \"", decoded4, "\"")
    if text4 == decoded4 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 5: Numbers and special characters
    print("Test 5: Numbers and special characters")
    let text5 = "2024-01-15 @ 3:30pm"
    print("  Original: \"", text5, "\"")
    let tokens5 = tokenize(tokenizer, text5, false)
    print("  Tokens:  ", tokens5)
    let decoded5 = detokenize(tokenizer, tokens5, false)
    print("  Decoded:  \"", decoded5, "\"")
    if text5 == decoded5 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 6: With special tokens
    print("Test 6: With special tokens (add_special_tokens=true)")
    let text6 = "Test with special tokens"
    print("  Original: \"", text6, "\"")
    let tokens6 = tokenize(tokenizer, text6, true)
    print("  Tokens:  ", tokens6)
    let decoded6 = detokenize(tokenizer, tokens6, true)
    print("  Decoded:  \"", decoded6, "\"")
    print("  Note: May include <s> or </s> tokens")
    print("")

    // Test 7: Single character
    print("Test 7: Single character")
    let text7 = "a"
    print("  Original: \"", text7, "\"")
    let tokens7 = tokenize(tokenizer, text7, false)
    print("  Tokens:  ", tokens7)
    let decoded7 = detokenize(tokenizer, tokens7, false)
    print("  Decoded:  \"", decoded7, "\"")
    if text7 == decoded7 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    // Test 8: Empty string edge case
    print("Test 8: Empty string")
    let text8 = ""
    print("  Original: \"", text8, "\"")
    let tokens8 = tokenize(tokenizer, text8, false)
    print("  Tokens:  ", tokens8)
    print("  Token count:", len(tokens8))
    let decoded8 = detokenize(tokenizer, tokens8, false)
    print("  Decoded:  \"", decoded8, "\"")
    if text8 == decoded8 {
        print("  ✅ PASS")
    } else {
        print("  ❌ FAIL")
    }
    print("")

    print("[3/3] Summary")
    print("All tests completed!")
    print("")
}
