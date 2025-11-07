// Test for loop scope management in TensorLogic DSL
// Tests that variables defined outside loops are not corrupted by operations inside loops

main {
    print("=== Testing For Loop Scope Management ===")
    print("")

    // Test 1: Basic tensor preservation across loop iterations
    print("Test 1: Basic tensor preservation")
    print("  Creating outer tensor [2, 3]...")
    let outer_tensor = zeros([2.0, 3.0])
    print("  Outer tensor created")
    print("  Outer tensor shape BEFORE loop: {}", shape(outer_tensor))

    for i in range(3) {
        print("  Iteration {}: Checking outer_tensor shape", i)
        let iter_shape = shape(outer_tensor)
        print("    Shape in iteration {}: {}", i, iter_shape)
    }

    print("  Outer tensor shape AFTER loop: {}", shape(outer_tensor))
    print("  ✅ Test 1 complete")
    print("")

    // Test 2: Multiple outer tensors with shape() calls inside loop
    print("Test 2: Multiple outer tensors with shape() inside loop")
    let tensor_a = zeros([4.0, 5.0])
    let tensor_b = zeros([6.0, 7.0])
    let tensor_c = zeros([8.0, 9.0])

    print("  BEFORE loop:")
    print("    tensor_a shape: {}", shape(tensor_a))
    print("    tensor_b shape: {}", shape(tensor_b))
    print("    tensor_c shape: {}", shape(tensor_c))

    for i in range(5) {
        print("  Iteration {}:", i)
        let shape_a = shape(tensor_a)
        let shape_b = shape(tensor_b)
        let shape_c = shape(tensor_c)
        print("    tensor_a: {}", shape_a)
        print("    tensor_b: {}", shape_b)
        print("    tensor_c: {}", shape_c)
    }

    print("  AFTER loop:")
    print("    tensor_a shape: {}", shape(tensor_a))
    print("    tensor_b shape: {}", shape(tensor_b))
    print("    tensor_c shape: {}", shape(tensor_c))
    print("  ✅ Test 2 complete")
    print("")

    // Test 3: Creating new tensors inside loop shouldn't affect outer tensors
    print("Test 3: Inner tensor creation doesn't corrupt outer tensors")
    let outer_data = zeros([10.0, 20.0])
    print("  BEFORE loop: outer_data shape = {}", shape(outer_data))

    for i in range(3) {
        print("  Iteration {}: Creating inner tensor", i)
        let inner_tensor = zeros([3.0, 4.0])
        let inner_shape = shape(inner_tensor)
        print("    Inner shape: {}", inner_shape)
        print("    Outer shape (should be unchanged): {}", shape(outer_data))
    }

    print("  AFTER loop: outer_data shape = {}", shape(outer_data))
    print("  ✅ Test 3 complete")
    print("")

    // Test 4: Nested loops with multiple outer scopes
    print("Test 4: Nested loops with multiple scopes")
    let level0_tensor = zeros([2.0, 2.0])
    print("  Level 0 tensor shape: {}", shape(level0_tensor))

    for i in range(2) {
        print("  Outer loop iteration {}", i)
        let level1_tensor = zeros([3.0, 3.0])
        print("    Level 1 tensor shape: {}", shape(level1_tensor))
        print("    Level 0 tensor shape (should be [2, 2]): {}", shape(level0_tensor))

        for j in range(2) {
            print("    Inner loop iteration {}", j)
            let level2_tensor = zeros([4.0, 4.0])
            print("      Level 2 tensor shape: {}", shape(level2_tensor))
            print("      Level 1 tensor shape (should be [3, 3]): {}", shape(level1_tensor))
            print("      Level 0 tensor shape (should be [2, 2]): {}", shape(level0_tensor))
        }

        print("    After inner loop - Level 1 shape: {}", shape(level1_tensor))
        print("    After inner loop - Level 0 shape: {}", shape(level0_tensor))
    }

    print("  After both loops - Level 0 shape: {}", shape(level0_tensor))
    print("  ✅ Test 4 complete")
    print("")

    // Test 5: Simulating the chat demo scenario
    print("Test 5: Simulating chat demo scenario (embedding + loop)")
    let tok_embd = zeros([32000.0, 2048.0])
    print("  tok_embd shape BEFORE loop: {}", shape(tok_embd))

    for i in range(3) {
        print("  DECODE iteration {}", i)
        print("    Checking tok_embd shape: {}", shape(tok_embd))

        // Simulate what happens in chat demo
        let token_id = 1.0
        let token_ids_single = zeros([1.0])
        print("    About to use tok_embd for embedding...")

        // This simulates: let new_token_emb = embedding(tok_embd, token_ids_single)
        // For testing, we just check shape
        let tok_embd_shape_check = shape(tok_embd)
        print("    tok_embd shape during embedding: {}", tok_embd_shape_check)

        // Create some intermediate tensors with shape() calls
        let intermediate = zeros([1.0, 2048.0])
        let inter_shape = shape(intermediate)
        print("    intermediate tensor shape: {}", inter_shape)

        print("    tok_embd shape at end of iteration: {}", shape(tok_embd))
    }

    print("  tok_embd shape AFTER loop: {}", shape(tok_embd))
    print("  ✅ Test 5 complete")
    print("")

    // Test 6: Heavy shape() usage inside loop
    print("Test 6: Heavy shape() usage inside loop")
    let tensor_x = zeros([100.0, 200.0])
    let tensor_y = zeros([300.0, 400.0])
    let tensor_z = zeros([500.0, 600.0])

    print("  BEFORE loop:")
    print("    X: {}", shape(tensor_x))
    print("    Y: {}", shape(tensor_y))
    print("    Z: {}", shape(tensor_z))

    for i in range(10) {
        // Call shape() multiple times per iteration
        let shape1 = shape(tensor_x)
        let shape2 = shape(tensor_y)
        let shape3 = shape(tensor_z)
        let shape4 = shape(tensor_x)
        let shape5 = shape(tensor_y)
        let shape6 = shape(tensor_z)

        if i == 0.0 {
            print("  First iteration shapes:")
            print("    X: {}, Y: {}, Z: {}", shape1, shape2, shape3)
        }

        if i == 9.0 {
            print("  Last iteration shapes:")
            print("    X: {}, Y: {}, Z: {}", shape4, shape5, shape6)
        }
    }

    print("  AFTER loop:")
    print("    X: {}", shape(tensor_x))
    print("    Y: {}", shape(tensor_y))
    print("    Z: {}", shape(tensor_z))
    print("  ✅ Test 6 complete")
    print("")

    // Test 7: Variable shadowing test
    print("Test 7: Variable shadowing behavior")
    let shadowed_var = zeros([7.0, 8.0])
    print("  Outer shadowed_var shape: {}", shape(shadowed_var))

    for i in range(2) {
        print("  Iteration {}", i)
        let shadowed_var = zeros([9.0, 10.0])
        print("    Inner shadowed_var shape: {}", shape(shadowed_var))
    }

    print("  After loop - outer shadowed_var shape: {}", shape(shadowed_var))
    print("  ✅ Test 7 complete")
    print("")

    print("=== All Scope Tests Complete ===")
    print("")
    print("EXPECTED BEHAVIOR:")
    print("  - All outer tensors should maintain their shapes")
    print("  - shape() calls inside loops should NOT corrupt outer variables")
    print("  - tok_embd should stay [32000, 2048] throughout Test 5")
    print("")
    print("FAILURE INDICATORS:")
    print("  - If any shape changes from expected (e.g., becomes [2])")
    print("  - If tok_embd becomes Tensor<f32>(shape=[2])")
    print("  - If outer tensors get overwritten by shape results")
}
