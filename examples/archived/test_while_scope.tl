// Test WHILE loop scope rules

main {
    print("=== Testing WHILE Loop Scope Rules ===")
    print("")

    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    // Test: Loop-internal variables should be re-initialized each iteration
    print("[WHILE] Loop-internal variables should be re-initialized:")
    let i = 0

    while i < 3 {
        let loop_var = model.blk[i].attn_norm.weight
        print("  Iteration", i, ": loop_var declared")
        i := i + 1
    }

    print("  Expected: 3 iterations without error")
    print("  Actual: Check if 'already defined' error occurred")
    print("")
}
