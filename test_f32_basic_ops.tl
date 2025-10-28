// Test basic f32 operations to identify performance issue

main {
    print("=== Testing f32 Basic Operations ===")
    print("")

    // Test 1: Simple tensor creation
    print("[1] Creating tensors...")
    let a = f32::ones([10, 10])
    let b = f32::ones([10, 10])
    print("  ✓ Tensor creation OK")
    print("")

    // Test 2: Element-wise operations
    print("[2] Element-wise operations...")
    let c = a + b
    let d = a * b
    print("  ✓ Element-wise OK")
    print("")

    // Test 3: Small matmul
    print("[3] Small matmul (10x10 @ 10x10)...")
    let e = matmul(a, b)
    print("  ✓ Small matmul OK")
    print("")

    // Test 4: Medium matmul
    print("[4] Medium matmul (100x100 @ 100x100)...")
    let x = f32::ones([100, 100])
    let y = f32::ones([100, 100])
    let z = matmul(x, y)
    print("  ✓ Medium matmul OK")
    print("")

    // Test 5: Large matmul (similar to model size)
    print("[5] Large matmul (256x256 @ 256x256)...")
    let p = f32::ones([256, 256])
    let q = f32::ones([256, 256])
    let r = matmul(p, q)
    print("  ✓ Large matmul OK")
    print("")

    // Test 6: ReLU activation
    print("[6] ReLU activation...")
    let relu_out = relu(a)
    print("  ✓ ReLU OK")
    print("")

    // Test 7: Softmax
    print("[7] Softmax...")
    let softmax_out = softmax(a, 1)
    print("  ✓ Softmax OK")
    print("")

    print("=== All basic operations completed successfully ===")
}
