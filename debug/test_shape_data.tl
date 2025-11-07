// Test to verify shape() returns correct data values, not just display

main {
    print("=== Testing shape() Data Values ===")
    print("")

    // Test 1: Check shape tensor data for [2, 3] tensor
    print("Test 1: Shape of [2, 3] tensor")
    let t1 = zeros([2.0, 3.0])
    let s1 = shape(t1)
    print("  Tensor t1 created with zeros([2.0, 3.0])")
    print("  Shape tensor s1 = shape(t1)")
    print("  Displaying s1: {}", s1)
    print("  s1[0] = {}", s1[0.0])
    print("  s1[1.0] = {}", s1[1.0])
    print("")

    // Test 2: Check shape tensor for [32000, 2048] tensor
    print("Test 2: Shape of [32000, 2048] tensor (like tok_embd)")
    let t2 = zeros([32000.0, 2048.0])
    let s2 = shape(t2)
    print("  Tensor t2 created with zeros([32000.0, 2048.0])")
    print("  Shape tensor s2 = shape(t2)")
    print("  Displaying s2: {}", s2)
    print("  s2[0] = {}", s2[0.0])
    print("  s2[1.0] = {}", s2[1.0])
    print("")

    // Test 3: Verify tok_embd itself, not its shape
    print("Test 3: Direct tensor check (not shape)")
    let tok_embd = zeros([32000.0, 2048.0])
    print("  Created tok_embd = zeros([32000.0, 2048.0])")
    print("  Displaying tok_embd directly: {}", tok_embd)
    print("")

    // Test 4: Try to use shape tensor values
    print("Test 4: Using shape tensor values in calculations")
    let t4 = zeros([10.0, 20.0])
    let s4 = shape(t4)
    print("  t4 = zeros([10.0, 20.0])")
    print("  s4 = shape(t4)")
    print("  s4[0] = {}", s4[0.0])
    print("  s4[1.0] = {}", s4[1.0])
    let dim0 = s4[0.0]
    let dim1 = s4[1.0]
    print("  Extracted: dim0 = {}, dim1 = {}", dim0, dim1)
    print("")

    print("=== Tests Complete ===")
}
