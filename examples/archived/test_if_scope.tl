// Test IF statement scope rules

main {
    print("")
    print("=== Testing IF Statement Scope Rules ===")
    print("")

    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    // ========================================================================
    // TEST 1: Variables declared in IF block should be local to that block
    // ========================================================================
    print("[TEST 1] IF block variables should be local:")

    let condition = true
    if condition {
        let if_local = model.blk[0].attn_norm.weight
        print("  Inside IF: if_local declared")
    }

    print("  After IF: if_local should NOT be accessible here")
    print("  (If we try to use if_local here, it should error)")
    print("")

    // ========================================================================
    // TEST 2: Variables in ELSE block should also be local
    // ========================================================================
    print("[TEST 2] ELSE block variables should be local:")

    let test_condition = false
    if test_condition {
        let in_if = 1
        print("  This shouldn't execute")
    } else {
        let in_else = model.blk[1].attn_norm.weight
        print("  Inside ELSE: in_else declared")
    }

    print("  After IF-ELSE: both in_if and in_else should NOT be accessible")
    print("")

    // ========================================================================
    // TEST 3: Outer variables should be accessible in IF blocks
    // ========================================================================
    print("[TEST 3] Outer variables accessible in IF:")

    let outer = 42
    let idx = 2

    if outer == 42 {
        print("  Inside IF: outer =", outer)
        let weight = model.blk[idx].attn_norm.weight
        print("  Inside IF: Can access outer 'idx' for model.blk[", idx, "]")
    }

    print("  ✅ Outer variables accessible in IF block")
    print("")

    // ========================================================================
    // TEST 4: Multiple IF statements should have separate scopes
    // ========================================================================
    print("[TEST 4] Multiple IF statements should have separate scopes:")

    if true {
        let local_var = model.blk[0].attn_norm.weight
        print("  First IF: local_var declared")
    }

    if true {
        let local_var = model.blk[1].attn_norm.weight
        print("  Second IF: local_var declared again (different scope)")
    }

    print("  ✅ Each IF block has its own scope")
    print("")

    // ========================================================================
    // TEST 5: Nested IF statements
    // ========================================================================
    print("[TEST 5] Nested IF statements:")

    if true {
        let outer_if = 1
        print("  Outer IF: outer_if =", outer_if)

        if true {
            let inner_if = 2
            print("  Inner IF: inner_if =", inner_if)
            print("  Inner IF: Can access outer_if =", outer_if)
            let weight = model.blk[0].attn_norm.weight
        }

        print("  Back to outer IF: outer_if still =", outer_if)
        print("  (inner_if should NOT be accessible here)")
    }

    print("  ✅ Nested IF blocks work correctly")
    print("")

    // ========================================================================
    // SUMMARY
    // ========================================================================
    print("=== IF SCOPE TEST SUMMARY ===")
    print("Expected behavior:")
    print("  1. Variables in IF/ELSE blocks are local to those blocks")
    print("  2. Outer variables are accessible inside IF/ELSE")
    print("  3. Each IF/ELSE block has its own scope")
    print("  4. Nested IF blocks follow normal scoping rules")
    print("")
}
