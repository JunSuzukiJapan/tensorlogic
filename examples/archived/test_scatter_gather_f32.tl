// Test f32 scatter() and gather() operations

main {
    print("=== Testing scatter() and gather() with f32 ===")
    print("")

    // Test 1: scatter with f32 indices
    print("[1] scatter() with f32 tensor and f32 indices:")
    let cache = f32::zeros([10, 4])
    let indices = f32::arange(3)
    let data = f32::ones([3, 4])
    let result = scatter(cache, 0, indices, data)
    print("  ✓ scatter() succeeded")
    print("  Result shape:", shape(result))
    print("")

    // Test 2: gather with f32 indices
    print("[2] gather() with f32 tensor and f32 indices:")
    let t = f32::ones([10, 4])
    let gather_indices = f32::arange(5)
    let gathered = t.gather(0, gather_indices)
    print("  ✓ gather() succeeded")
    print("  Gathered shape:", shape(gathered))
    print("")

    // Test 3: Combined scatter then gather
    print("[3] Combined: scatter then gather:")
    let cache2 = f32::zeros([20, 8])
    let pos1 = f32::ones([1]) * 5.0
    let pos2 = f32::ones([1]) * 10.0
    let data1 = f32::ones([1, 8]) * 2.0
    let data2 = f32::ones([1, 8]) * 3.0

    cache2 = scatter(cache2, 0, pos1, data1)
    cache2 = scatter(cache2, 0, pos2, data2)

    let extract_indices = f32::arange(15)
    let extracted = cache2.gather(0, extract_indices)
    print("  ✓ scatter + gather succeeded")
    print("  Extracted shape:", shape(extracted))
    print("")

    print("=== All scatter/gather f32 tests passed! ===")
}
