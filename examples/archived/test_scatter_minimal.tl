// Minimal scatter test without slice

main {
    print("=== Minimal Scatter Test ===")
    print("")

    print("[Step 1] Create f32 tensors:")
    let cache = zeros([10.0, 3.0])
    let indices = arange(3.0)
    let data = ones([3.0, 3.0])
    print("  ✓ Tensors created")
    print("")

    print("[Step 2] Execute scatter:")
    let result = scatter(cache, 0, indices, data)
    print("  ✓ scatter succeeded!")
    print("  Result shape:", shape(result))
    print("")

    print("=== Test Complete ===")
}
