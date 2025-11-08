// GPU tensor operations with for-loop
// Progressively complex tests to isolate the issue

test test1_simple_loop {
    print("=== Test 1: Simple for-loop without GPU ===")

    let sum = 0
    for i in range(5) {
        sum = sum + 1
    }

    print("Sum: {}", sum)
    if sum == 5 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test2_loop_with_model_load {
    print("=== Test 2: For-loop with model loaded ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")

    let sum = 0
    for i in range(3) {
        sum = sum + 1
    }

    print("Sum: {}", sum)
    if sum == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test3_loop_with_embedding {
    print("=== Test 3: For-loop with embedding ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)

    print("Embedding shape: {}", shape(x))

    let sum = 0
    for i in range(3) {
        sum = sum + 1
    }

    print("Sum: {}", sum)
    if sum == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test4_loop_with_linear {
    print("=== Test 4: For-loop with linear operation ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)

    print("Embedding shape: {}", shape(x))

    // Perform linear BEFORE loop
    let Q = linear(x, L0.attn_q.weight)
    print("Q shape: {}", shape(Q))

    let sum = 0
    for i in range(3) {
        sum = sum + 1
    }

    print("Sum: {}", sum)
    if sum == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test5_linear_inside_loop {
    print("=== Test 5: Linear operation INSIDE for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)

    print("Embedding shape: {}", shape(x))

    let count = 0
    for i in range(3) {
        let Q = linear(x, L0.attn_q.weight)
        print("  Iteration {}: Q shape = {}", i, shape(Q))
        count = count + 1
    }

    print("Count: {}", count)
    if count == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test6_rms_norm_inside_loop {
    print("=== Test 6: RMS norm INSIDE for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)

    print("Embedding shape: {}", shape(x))

    let count = 0
    for i in range(3) {
        let normed = rms_norm(x, L0.attn_norm.weight)
        print("  Iteration {}: normed shape = {}", i, shape(normed))
        count = count + 1
    }

    print("Count: {}", count)
    if count == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}

test test7_multiple_ops_inside_loop {
    print("=== Test 7: Multiple GPU ops INSIDE for-loop ===")

    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-f16.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]
    let tokens = tokenizer.tokenize("Hello", true)
    let x = embedding(tok_embd, tokens)

    print("Embedding shape: {}", shape(x))

    let count = 0
    for i in range(3) {
        let normed = rms_norm(x, L0.attn_norm.weight)
        let Q = linear(normed, L0.attn_q.weight)
        let K = linear(normed, L0.attn_k.weight)
        print("  Iteration {}: Q={}, K={}", i, shape(Q), shape(K))
        count = count + 1
    }

    print("Count: {}", count)
    if count == 3 {
        print("✓ PASS")
    } else {
        print("❌ FAIL")
    }
}
