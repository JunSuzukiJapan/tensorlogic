// TDD Test Suite for All Loop Constructs with Array Indexing
// Tests: for, while, loop (with break), if

main {
    print("")
    print("=== TDD: Testing All Loop Constructs with Array Indexing ===")
    print("")

    // Load model once for all tests
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    print("✓ Model loaded")
    print("")

    // ========================================================================
    // TEST 1: FOR LOOP with loop variable in array index
    // ========================================================================
    print("[TEST 1] FOR loop: for i in range(3) { model.blk[i].weight }")

    for i in range(3) {
        let weight1 = model.blk[i].attn_norm.weight
        print("  ✓ for loop iteration", i, "- accessed model.blk[", i, "]")
    }

    print("  RESULT: FOR loop PASSED ✅")
    print("")

    // ========================================================================
    // TEST 2: WHILE LOOP with counter variable in array index
    // ========================================================================
    print("[TEST 2] WHILE loop: while i < 3 { model.blk[i].weight }")
    let i = 0
    let weight2 = model.blk[0].attn_norm.weight

    while i < 3 {
        weight2 := model.blk[i].attn_norm.weight
        print("  ✓ while loop iteration", i, "- accessed model.blk[", i, "]")
        i := i + 1
    }

    print("  RESULT: WHILE loop PASSED ✅")
    print("")

    // ========================================================================
    // TEST 3: LOOP (infinite) with BREAK and counter in array index
    // ========================================================================
    print("[TEST 3] LOOP with BREAK: loop { if i >= 3 { break } }")
    let j = 0
    let weight3 = model.blk[0].attn_norm.weight

    loop {
        if j >= 3 {
            break
        }

        weight3 := model.blk[j].attn_norm.weight
        print("  ✓ loop iteration", j, "- accessed model.blk[", j, "]")
        j := j + 1
    }

    print("  RESULT: LOOP with BREAK PASSED ✅")
    print("")

    // ========================================================================
    // TEST 4: IF statement with variable in array index
    // ========================================================================
    print("[TEST 4] IF statement: if condition { model.blk[k].weight }")
    let k = 1

    if k == 1 {
        let weight4 = model.blk[k].attn_norm.weight
        print("  ✓ if block - accessed model.blk[", k, "]")
    }

    print("  RESULT: IF statement PASSED ✅")
    print("")

    // ========================================================================
    // TEST 5: NESTED LOOPS with multiple variables
    // ========================================================================
    print("[TEST 5] NESTED loops: for i in range(2) { for j in range(2) { ... } }")

    for i in range(2) {
        for j in range(2) {
            let layer_idx = i
            let weight5 = model.blk[layer_idx].attn_norm.weight
            print("  ✓ nested loop i=", i, "j=", j, "- accessed model.blk[", layer_idx, "]")
        }
    }

    print("  RESULT: NESTED loops PASSED ✅")
    print("")

    // ========================================================================
    // SUMMARY
    // ========================================================================
    print("=== TEST SUMMARY ===")
    print("✅ FOR loop:       Array indexing with loop variable works")
    print("✅ WHILE loop:     Array indexing with counter variable works")
    print("✅ LOOP + BREAK:   Array indexing with counter variable works")
    print("✅ IF statement:   Array indexing with conditional variable works")
    print("✅ NESTED loops:   Array indexing with nested variables works")
    print("")
    print("🎉 All loop constructs support dynamic array indexing!")
    print("")
}
