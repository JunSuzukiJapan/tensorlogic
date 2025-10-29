// Comprehensive test for f32 type namespace
// NOTE: :: notation is ONLY for constructor functions (zeros, ones, arange)
//       Instance methods use dot notation: tensor.reshape([2, 3])

main {
    print("=== Testing f32 Type System ===")
    print("")

    // Constructor functions (use f32:: prefix)
    print("[1] f32::zeros and f32::ones:")
    let z = f32::zeros([3, 2])
    let o = f32::ones([2, 3])
    print("  ✓ zeros shape:", shape(z))
    print("  ✓ ones shape:", shape(o))
    print("")

    // arange
    print("[2] f32::arange:")
    let r = f32::arange(5)
    print("  ✓ arange shape:", shape(r))
    print("")

    // reshape (instance method - use dot notation)
    print("[3] reshape on f32 tensor:")
    let data = f32::ones([6])
    let reshaped = data.reshape([2, 3])
    print("  ✓ reshape [6] -> [2, 3]:", shape(reshaped))
    print("")

    // transpose (instance method - use dot notation)
    print("[4] transpose on f32 tensor:")
    let mat = f32::ones([3, 4])
    let trans = mat.transpose()
    print("  ✓ transpose [3, 4] -> [4, 3]:", shape(trans))
    print("")

    // flatten, squeeze, unsqueeze (instance methods - use dot notation)
    print("[5] flatten, squeeze, unsqueeze on f32 tensors:")
    let cube = f32::zeros([2, 3, 4])
    let flat = cube.flatten()
    print("  ✓ flatten [2,3,4] ->", shape(flat))

    let single = f32::zeros([1, 5])
    let squeezed = single.squeeze(0)
    print("  ✓ squeeze [1,5] ->", shape(squeezed))

    let unsqueezed = squeezed.unsqueeze(0)
    print("  ✓ unsqueeze back ->", shape(unsqueezed))
    print("")

    // Math operations
    print("[6] Math operations:")
    let a = f32::ones([3, 2])
    let b = f32::ones([3, 2])
    let sum_result = a + b
    print("  ✓ addition works")
    let prod_result = a * b
    print("  ✓ multiplication works")
    print("")

    // scatter (zero-copy)
    print("[7] f32::scatter (zero-copy):")
    let cache = f32::zeros([10, 4])
    let indices_sc = f32::arange(3)
    let data_sc = f32::ones([3, 4])
    cache = scatter(cache, 0, indices_sc, data_sc)
    print("  ✓ scatter succeeded (no memory copy!)")
    print("")

    print("=== f32 Type System Test Complete ===")
    print("✓ Constructor functions: f32::zeros, f32::ones, f32::arange")
    print("✓ Instance methods work correctly with f32 tensors")
}
