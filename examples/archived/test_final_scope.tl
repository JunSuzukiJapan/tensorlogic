// Final comprehensive scope test for all loop types

main {
    print("")
    print("=== Final Comprehensive Scope Test ===")
    print("")

    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    // ========================================================================
    // TEST 1: FOR loop with model.blk[i] in each iteration
    // ========================================================================
    print("[TEST 1] FOR loop scope:")
    for i in range(3) {
        let layer_weight = model.blk[i].attn_norm.weight
        print("  Iteration", i, ": layer_weight declared")
    }
    print("  ✅ PASSED: Each iteration can declare 'layer_weight'")
    print("")

    // ========================================================================
    // TEST 2: WHILE loop with model.blk[i] in each iteration
    // ========================================================================
    print("[TEST 2] WHILE loop scope:")
    let i = 0
    while i < 3 {
        let layer_weight = model.blk[i].attn_norm.weight
        print("  Iteration", i, ": layer_weight declared")
        i = i + 1
    }
    print("  ✅ PASSED: Each iteration can declare 'layer_weight'")
    print("")

    // ========================================================================
    // TEST 3: LOOP with model.blk[j] in each iteration
    // ========================================================================
    print("[TEST 3] LOOP scope:")
    let j = 0
    loop {
        if j >= 3 {
            break
        }
        let layer_weight = model.blk[j].attn_norm.weight
        print("  Iteration", j, ": layer_weight declared")
        j = j + 1
    }
    print("  ✅ PASSED: Each iteration can declare 'layer_weight'")
    print("")

    // ========================================================================
    // TEST 4: Outer variables accessible in loops
    // ========================================================================
    print("[TEST 4] Outer variables accessible:")
    let outer = 42

    for i in range(2) {
        print("  FOR: outer =", outer)
    }

    let k = 0
    while k < 2 {
        print("  WHILE: outer =", outer)
        k = k + 1
    }

    let m = 0
    loop {
        if m >= 2 {
            break
        }
        print("  LOOP: outer =", outer)
        m = m + 1
    }
    print("  ✅ PASSED: Outer variables accessible in all loops")
    print("")

    // ========================================================================
    // SUMMARY
    // ========================================================================
    print("=== ALL TESTS PASSED ===")
    print("")
    print("✅ FOR loop: Correct scope (iteration-local variables)")
    print("✅ WHILE loop: Correct scope (iteration-local variables)")
    print("✅ LOOP: Correct scope (iteration-local variables)")
    print("✅ All loops: Outer variables accessible")
    print("")
    print("🎉 Scope rules are now consistent across all loop types!")
    print("")
}
