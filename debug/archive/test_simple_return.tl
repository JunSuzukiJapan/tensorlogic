// Test: Simple function return to isolate the bug
//
// This test checks if the problem is specific to:
// 1. Direct return of binary operations
// 2. Specific to multiplication
// 3. Related to function call overhead

// Test 1: Direct return of addition
fn test_add_direct(x: float16[?, ?], y: float16[?, ?]) -> float16[?, ?] {
    x + y
}

// Test 2: Addition with intermediate variable
fn test_add_indirect(x: float16[?, ?], y: float16[?, ?]) -> float16[?, ?] {
    let result = x + y
    result
}

// Test 3: Direct return of multiplication
fn test_mul_direct(x: float16[?, ?], y: float16[?, ?]) -> float16[?, ?] {
    x * y
}

// Test 4: Multiplication with intermediate variable
fn test_mul_indirect(x: float16[?, ?], y: float16[?, ?]) -> float16[?, ?] {
    let result = x * y
    result
}

// Test 5: Direct return of unary operation
fn test_sigmoid_direct(x: float16[?, ?]) -> float16[?, ?] {
    sigmoid(x)
}

// Test 6: Unary with intermediate variable
fn test_sigmoid_indirect(x: float16[?, ?]) -> float16[?, ?] {
    let result = sigmoid(x)
    result
}

main {
    print("================================================================================")
    print("Simple Function Return Bug Test")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")

    // Create simple test tensor
    let tokens = tokenize(tokenizer, "Hello", false)
    let z = embedding(embed_table, tokens)

    print("[1/6] Testing addition (direct return)...")
    let result1 = test_add_direct(z, z)
    let sum1 = sum(result1)
    print("  Result sum:", sum1)
    if sum1 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum1 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[2/6] Testing addition (indirect return)...")
    let result2 = test_add_indirect(z, z)
    let sum2 = sum(result2)
    print("  Result sum:", sum2)
    if sum2 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum2 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[3/6] Testing multiplication (direct return)...")
    let result3 = test_mul_direct(z, z)
    let sum3 = sum(result3)
    print("  Result sum:", sum3)
    if sum3 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum3 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[4/6] Testing multiplication (indirect return)...")
    let result4 = test_mul_indirect(z, z)
    let sum4 = sum(result4)
    print("  Result sum:", sum4)
    if sum4 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum4 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[5/6] Testing sigmoid (direct return)...")
    let result5 = test_sigmoid_direct(z)
    let sum5 = sum(result5)
    print("  Result sum:", sum5)
    if sum5 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum5 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[6/6] Testing sigmoid (indirect return)...")
    let result6 = test_sigmoid_indirect(z)
    let sum6 = sum(result6)
    print("  Result sum:", sum6)
    if sum6 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum6 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("================================================================================")
    print("✅ Test Complete")
    print("================================================================================")
}
