// Scatter type debug test

main {
    print("=== Scatter Type Debug ===")
    print("")

    // Create all f32 tensors explicitly
    print("[Step 1] Create cache with zeros([10.0, 3.0]):")
    let cache = zeros([10.0, 3.0])
    print("  Cache created")
    print("")

    print("[Step 2] Create indices with arange(3.0):")
    let indices = arange(3.0)
    print("  Indices created")
    print("")

    print("[Step 3] Create data with ones([3.0, 3.0]):")
    let data = ones([3.0, 3.0])
    print("  Data created")
    print("")

    // Try to add to verify types match
    print("[Step 4] Type verification - cache + cache:")
    let verify1 = cache + cache
    print("  ✓ cache is consistent type")
    print("")

    print("[Step 5] Type verification - data + data:")
    let verify2 = data + data
    print("  ✓ data is consistent type")
    print("")

    print("[Step 6] Type verification - cache row + data row:")
    let cache_slice = slice(cache, 0, 0, 1)
    let data_slice = slice(data, 0, 0, 1)
    let cache_flat = reshape(cache_slice, [3.0])
    let data_flat = reshape(data_slice, [3.0])
    let verify3 = cache_flat + data_flat
    print("  ✓ cache and data are same type")
    print("")

    print("[Step 7] Now try scatter:")
    let result = scatter(cache, 0, indices, data)
    print("  ✓ scatter succeeded!")
    print("")

    print("=== Debug Complete ===")
}
