// Simple scatter test

main {
    print("=== Testing scatter with arange ===")
    print("")

    // Test 1: Create simple tensors
    print("[Test 1] Create tensor and arange:")
    let cache = zeros([10.0, 3.0])
    let s_cache = shape(cache)
    print("  Cache shape:", s_cache)

    let indices = arange(3.0)
    let s_ind = shape(indices)
    print("  Indices shape:", s_ind)

    let data = zeros([3.0, 3.0])
    let s_data = shape(data)
    print("  Data shape:", s_data)
    print("")

    // Test 2: Try scatter
    print("[Test 2] scatter(cache, 0, indices, data):")
    let result = scatter(cache, 0, indices, data)
    let s_result = shape(result)
    print("  Result shape:", s_result)
    print("")

    print("=== Tests complete ===")
}
