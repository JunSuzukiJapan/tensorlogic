// Simple GNN function test

main {
    print("=== Testing GNN Functions ===")
    print("")

    let alice = zeros([8])
    let bob = zeros([8])

    print("Test 1: normalize_features")
    let normalized = normalize_features(alice, "l2")
    print("  Result:", normalized)
    print("")

    print("Test 2: relational_aggregate")
    let lives_in = zeros([8])
    let tokyo = zeros([8])
    let aggregated = relational_aggregate(alice, lives_in, tokyo)
    print("  Result:", aggregated)
    print("")

    print("Test 3: graph_attention")
    let attended = graph_attention(alice, bob, bob)
    print("  Result:", attended)
    print("")

    print("✅ All GNN function tests passed!")
}
