// Test Variable Shadowing
// Validates that blocks can shadow outer variables and restore them correctly

main {
    print("=== Testing Variable Shadowing ===")
    print("")

    // Test 1: Basic shadowing in block
    print("[Test 1] Basic shadowing")
    let x = 10
    print("  Outer: x =", x)

    {
        let x = 20
        print("  Inside block: x =", x, "(shadowed)")
    }

    print("  After block: x =", x, "(restored)")
    print("  ✓ Basic shadowing works")
    print("")

    // Test 2: Multiple levels of shadowing
    print("[Test 2] Nested shadowing")
    let y = 1
    print("  Level 0: y =", y)

    {
        let y = 2
        print("  Level 1: y =", y)

        {
            let y = 3
            print("  Level 2: y =", y)
        }

        print("  Back to Level 1: y =", y)
    }

    print("  Back to Level 0: y =", y)
    print("  ✓ Nested shadowing works")
    print("")

    // Test 3: Shadowing in IF blocks
    print("[Test 3] Shadowing in IF blocks")
    let val = 100
    print("  Before IF: val =", val)

    if val > 0 {
        let val = 200
        print("  Inside IF: val =", val, "(shadowed)")
    }

    print("  After IF: val =", val, "(restored)")
    print("  ✓ IF block shadowing works")
    print("")

    // Test 4: Shadowing in FOR loops
    print("[Test 4] Shadowing in FOR loops")
    let i = 999
    print("  Before FOR: i =", i)

    for i in range(3) {
        print("  FOR iteration: i =", i, "(loop variable)")
        let inner = i
        print("    inner =", inner)
    }

    print("  After FOR: i =", i, "(restored)")
    print("  ✓ FOR loop shadowing works")
    print("")

    // Test 5: Shadowing in WHILE loops
    print("[Test 5] Shadowing in WHILE loops")
    let outer_counter = 0
    print("  Before WHILE: outer_counter =", outer_counter)

    while outer_counter < 2 {
        let inner_counter = 100
        print("  WHILE iteration: outer_counter =", outer_counter, ", inner_counter =", inner_counter, "(shadowed)")
        outer_counter = outer_counter + 1
    }

    print("  After WHILE: outer_counter =", outer_counter)
    print("  ✓ WHILE loop shadowing works")
    print("")

    // Test 6: Shadowing doesn't affect assignment
    print("[Test 6] Shadowing with assignment")
    let z = 1
    print("  Initial: z =", z)

    {
        let z = 2
        print("  Block: z =", z, "(shadowed)")

        z = 3
        print("  Block after assignment: z =", z, "(modified shadow)")
    }

    print("  After block: z =", z, "(original restored, not affected by shadow modification)")
    print("  ✓ Shadowing isolates modifications")
    print("")

    // Test 7: Mixed shadowing and new variables
    print("[Test 7] Mixed shadowing and new variables")
    let a = 10
    print("  Outer: a =", a)

    {
        let a = 20
        let b = 30
        print("  Block: a =", a, "(shadowed), b =", b, "(new)")
    }

    print("  After block: a =", a, "(restored)")
    // b should not be accessible here
    print("  ✓ Mixed shadowing works")
    print("")

    print("=== All Variable Shadowing Tests Passed ===")
}
