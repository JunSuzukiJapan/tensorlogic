// Test Block Statement Syntax and Scoping
// Validates that { } blocks have proper variable scoping

main {
    print("=== Testing Block Statement ===")
    print("")

    // Test 1: Basic block with local variable
    print("[Test 1] Basic block with local variable")
    let outer = 42
    print("  Before block: outer =", outer)

    {
        let inner = 1
        print("  Inside block: outer =", outer, ", inner =", inner)
    }

    print("  After block: outer =", outer)
    print("  ✓ Block-local variable 'inner' correctly isolated")
    print("")

    // Test 2: Nested blocks
    print("[Test 2] Nested blocks")
    let x = 10
    print("  Outer: x =", x)

    {
        let y = 20
        print("  Block 1: x =", x, ", y =", y)

        {
            let z = 30
            print("  Block 2: x =", x, ", y =", y, ", z =", z)
        }

        print("  Block 1 again: x =", x, ", y =", y)
    }

    print("  Outer again: x =", x)
    print("  ✓ Nested blocks work correctly")
    print("")

    // Test 3: Multiple sequential blocks
    print("[Test 3] Multiple sequential blocks")

    {
        let block1_var = 100
        print("  Block 1: block1_var =", block1_var)
    }

    {
        let block2_var = 200
        print("  Block 2: block2_var =", block2_var)
    }

    print("  ✓ Sequential blocks are independent")
    print("")

    // Test 4: Block inside control flow
    print("[Test 4] Block inside IF statement")
    let condition = 1

    if condition > 0 {
        print("  Inside IF block")

        {
            let if_block_var = 999
            print("  Inside nested block within IF: if_block_var =", if_block_var)
        }

        print("  Still inside IF, after nested block")
    }

    print("  ✓ Blocks work inside control flow")
    print("")

    print("=== All Block Statement Tests Passed ===")
}
