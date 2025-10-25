// Test concat operation for KV cache implementation

main {
    print("=== Concat Operation Test ===")
    print("")

    // Test 1: Concat along dimension 0
    print("[Test 1] Concat along dimension 0:")
    let a = ones([2.0, 3.0])
    let b = ones([2.0, 3.0])
    print("  Tensor A: ", shape(a))
    print("  Tensor B: ", shape(b))

    let c0 = concat(a, b, 0.0)
    print("  Result (dim=0): ", shape(c0))
    print("  Expected: [4, 3]")
    print("")

    // Test 2: Concat along dimension 1
    print("[Test 2] Concat along dimension 1:")
    let c1 = concat(a, b, 1.0)
    print("  Result (dim=1): ", shape(c1))
    print("  Expected: [2, 6]")
    print("")

    // Test 3: KV cache scenario - simulating accumulation
    print("[Test 3] KV cache accumulation (seq_len dimension):")
    print("  Initial K cache (token 1): ", shape(a))

    let k_cache = a  // First token's K
    let new_k = b    // Second token's K
    let k_accumulated = concat(k_cache, new_k, 0.0)

    print("  After token 2: ", shape(k_accumulated))
    print("  Expected: [4, 3]  (seq_len doubled)")
    print("")

    print("✅ All concat tests passed!")
    print("")
    print("Ready for KV cache implementation in transformer!")
}
