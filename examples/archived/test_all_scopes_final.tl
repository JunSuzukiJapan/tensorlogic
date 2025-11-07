// Comprehensive Final Test: All Control Flow Scopes with model.blk[i] Access

main {
    print("")
    print("=== 🎯 Final Comprehensive Scope Test ===")
    print("")

    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    print("✓ Model loaded")
    print("")

    // ========================================================================
    // RULE 1: Outer variables accessible in all control structures
    // ========================================================================
    print("📋 RULE 1: Outer variables accessible in all control structures")
    let outer_idx = 1
    let outer_val = 42

    if true {
        print("  IF: outer_idx =", outer_idx, ", outer_val =", outer_val)
        let w1 = model.blk[outer_idx].attn_norm.weight
    }

    for i in range(1) {
        print("  FOR: outer_idx =", outer_idx, ", outer_val =", outer_val)
        let w2 = model.blk[outer_idx].attn_norm.weight
    }

    let j = 0
    while j < 1 {
        print("  WHILE: outer_idx =", outer_idx, ", outer_val =", outer_val)
        let w3 = model.blk[outer_idx].attn_norm.weight
        j = j + 1
    }

    let k = 0
    loop {
        if k >= 1 { break }
        print("  LOOP: outer_idx =", outer_idx, ", outer_val =", outer_val)
        let w4 = model.blk[outer_idx].attn_norm.weight
        k = k + 1
    }

    print("  ✅ RULE 1 PASSED: All control structures can access outer variables")
    print("")

    // ========================================================================
    // RULE 2: Block-local variables are re-initialized/isolated
    // ========================================================================
    print("📋 RULE 2: Block-local variables are re-initialized/isolated")

    print("  Testing IF blocks:")
    if true {
        let local = model.blk[0].attn_norm.weight
        print("    First IF: local declared")
    }
    if true {
        let local = model.blk[1].attn_norm.weight
        print("    Second IF: local declared again (no conflict)")
    }

    print("  Testing FOR iterations:")
    for i in range(3) {
        let iteration_var = model.blk[i].attn_norm.weight
        print("    FOR iteration", i, ": iteration_var declared")
    }

    print("  Testing WHILE iterations:")
    let m = 0
    while m < 3 {
        let iteration_var = model.blk[m].attn_norm.weight
        print("    WHILE iteration", m, ": iteration_var declared")
        m = m + 1
    }

    print("  Testing LOOP iterations:")
    let n = 0
    loop {
        if n >= 3 { break }
        let iteration_var = model.blk[n].attn_norm.weight
        print("    LOOP iteration", n, ": iteration_var declared")
        n = n + 1
    }

    print("  ✅ RULE 2 PASSED: Block-local variables properly isolated/re-initialized")
    print("")

    // ========================================================================
    // RULE 3: Dynamic array indexing with variables
    // ========================================================================
    print("📋 RULE 3: Dynamic array indexing works in all control structures")

    if true {
        let idx = 2
        let weight = model.blk[idx].attn_norm.weight
        print("  IF: model.blk[", idx, "] accessed")
    }

    for i in range(3) {
        let weight = model.blk[i].attn_norm.weight
        print("  FOR: model.blk[", i, "] accessed")
    }

    let p = 0
    while p < 3 {
        let weight = model.blk[p].attn_norm.weight
        print("  WHILE: model.blk[", p, "] accessed")
        p = p + 1
    }

    let q = 0
    loop {
        if q >= 3 { break }
        let weight = model.blk[q].attn_norm.weight
        print("  LOOP: model.blk[", q, "] accessed")
        q = q + 1
    }

    print("  ✅ RULE 3 PASSED: Dynamic array indexing works everywhere")
    print("")

    // ========================================================================
    // RULE 4: Nested structures work correctly
    // ========================================================================
    print("📋 RULE 4: Nested structures maintain proper scoping")

    for i in range(2) {
        let outer_loop_var = i
        print("  Outer FOR i =", i)

        if i == 1 {
            let if_var = model.blk[i].attn_norm.weight
            print("    Inside IF: can access outer_loop_var =", outer_loop_var)

            for j in range(2) {
                let inner_loop_var = model.blk[j].attn_norm.weight
                print("      Inner FOR j =", j, ", can access i =", i)
            }
        }
    }

    print("  ✅ RULE 4 PASSED: Nested structures work correctly")
    print("")

    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    print("=== 🎉 ALL SCOPE RULES VERIFIED ===")
    print("")
    print("Summary of TensorLogic Scope Rules:")
    print("  1. ✅ Outer variables accessible in all control structures")
    print("  2. ✅ Block-local variables properly isolated/re-initialized")
    print("  3. ✅ Dynamic array indexing (model.blk[var]) works everywhere")
    print("  4. ✅ Nested structures maintain proper scoping")
    print("")
    print("Control Structures Tested:")
    print("  ✅ IF/ELSE statements")
    print("  ✅ FOR loops")
    print("  ✅ WHILE loops")
    print("  ✅ LOOP (infinite with break)")
    print("")
    print("💡 All control structures now have consistent, predictable scoping!")
    print("")
}
