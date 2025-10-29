// Test Tensor::create_cache and .append() methods

main {
    print("=== Testing Tensor Cache Methods ===")
    print("")

    // Test 1: Tensor::create_cache (static method)
    print("[Test 1] Tensor::create_cache()")
    let cache = Tensor::create_cache(512.0, [4.0, 64.0])
    print("  Created cache with capacity 512")
    let cache_shape = cache.shape()
    print("  Shape:", cache_shape)
    print("")

    // Test 2: .append() instance method
    print("[Test 2] cache.append()")
    let new_data = f32::ones([1.0, 4.0, 64.0])
    cache = cache.append(new_data)
    print("  Appended 1 element")
    let new_shape = cache.shape()
    print("  New shape:", new_shape)
    print("")

    // Test 3: Multiple appends
    print("[Test 3] Multiple appends")
    let data2 = f32::ones([2.0, 4.0, 64.0])
    cache = cache.append(data2)
    print("  Appended 2 more elements")
    let final_shape = cache.shape()
    print("  Final shape:", final_shape)
    print("")

    print("=== All Tests Passed ===")
}
