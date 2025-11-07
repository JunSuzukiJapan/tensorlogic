// ============================================================
// Scope Management Comprehensive Tests
// Tests block scope, variable shadowing, loops, functions, and early returns
// ============================================================

// Test 1: Basic Block Scope
// Block-local variables should not be visible outside the block
fn test_basic_block_scope() -> float {
    let outer = 10.0

    {
        let inner = 20.0
        let sum = outer + inner
        assert_eq(sum, 30.0)
    }

    // inner should not be accessible here
    // If scope is working, this would error, but we can't test that directly
    // Instead, verify outer still exists
    assert_eq(outer, 10.0)

    print("[PASS] Test 1: Basic block scope")
    return 0.0
}

// Test 2: Variable Shadowing
// Inner block can shadow outer variable, outer value restored after block
fn test_variable_shadowing() -> float {
    let x = 100.0
    assert_eq(x, 100.0)

    {
        let x = 200.0  // Shadows outer x
        assert_eq(x, 200.0)

        {
            let x = 300.0  // Shadows both
            assert_eq(x, 300.0)
        }

        // Back to middle scope
        assert_eq(x, 200.0)
    }

    // Back to outer scope
    assert_eq(x, 100.0)

    print("[PASS] Test 2: Variable shadowing")
    return 0.0
}

// Test 3: Nested Blocks (3+ levels)
// Multiple levels of nesting should maintain separate scopes
fn test_nested_blocks() -> float {
    let level1 = 1.0

    {
        let level2 = 2.0

        {
            let level3 = 3.0

            {
                let level4 = 4.0
                let sum = level1 + level2 + level3 + level4
                assert_eq(sum, 10.0)
            }

            // level4 not accessible, but others are
            let sum = level1 + level2 + level3
            assert_eq(sum, 6.0)
        }

        // level3, level4 not accessible
        let sum = level1 + level2
        assert_eq(sum, 3.0)
    }

    // Only level1 accessible
    assert_eq(level1, 1.0)

    print("[PASS] Test 3: Nested blocks")
    return 0.0
}

// Test 4: Loop Variable Scope
// Loop variables should be scoped to loop body
fn test_loop_variable_scope() -> float {
    let outer_count = 0.0

    for i in [1.0, 2.0, 3.0] {
        let loop_local = i * 10.0
        outer_count = outer_count + i
    }

    // Loop variable 'i' and 'loop_local' should not be accessible here
    assert_eq(outer_count, 6.0)

    print("[PASS] Test 4: Loop variable scope")
    return 0.0
}

// Test 5: Loop Shadowing
// Loop can shadow outer variables, restored after loop
fn test_loop_shadowing() -> float {
    let x = 0.0

    for x in [10.0, 20.0, 30.0] {
        // x shadows outer x in each iteration
        let dummy = x  // Use x to avoid unused warning
    }

    // Outer x should be restored
    assert_eq(x, 0.0)

    print("[PASS] Test 5: Loop shadowing")
    return 0.0
}

// Test 6: Function Local Scope
// Function parameters and locals should not leak outside
fn helper_function(param: Tensor) -> float {
    let local_var = param + 5.0
    return local_var
}

fn test_function_local_scope() -> float {
    let input = 10.0
    let result = helper_function(input)

    assert_eq(result, 15.0)
    // param and local_var from helper_function should not be accessible
    assert_eq(input, 10.0)

    print("[PASS] Test 6: Function local scope")
    return 0.0
}

// Test 7: Function with Block Scope
// Blocks inside functions should have proper scoping
fn function_with_blocks(x: Tensor) -> float {
    let result = 0.0

    {
        let temp = x * 2.0
        result = temp
    }

    // temp not accessible, but result is
    return result
}

fn test_function_with_blocks() -> float {
    let val = function_with_blocks(5.0)
    assert_eq(val, 10.0)

    print("[PASS] Test 7: Function with blocks")
    return 0.0
}

// Test 8: Early Return Cleanup
// Variables should be cleaned up when function returns early
fn early_return_function(flag: Tensor) -> float {
    let var1 = 10.0
    let var2 = 20.0

    if flag > 0.0 {
        let early_local = 100.0
        return early_local
    }

    return var1 + var2
}

fn test_early_return() -> float {
    let result1 = early_return_function(1.0)
    assert_eq(result1, 100.0)

    let result2 = early_return_function(-1.0)
    assert_eq(result2, 30.0)

    print("[PASS] Test 8: Early return cleanup")
    return 0.0
}

// Test 9: If/Else Branch Scope
// Each branch should have independent scope
fn test_if_else_scope() -> float {
    let cond = 1.0
    let result = 0.0

    if cond > 0.0 {
        let then_var = 100.0
        result = then_var
    } else {
        let else_var = 200.0
        result = else_var
    }

    // then_var and else_var should not be accessible
    assert_eq(result, 100.0)

    print("[PASS] Test 9: If/else branch scope")
    return 0.0
}

// Test 10: Memory Leak Prevention
// Reassigning same variable many times should not accumulate references
fn test_memory_leak_prevention() -> float {
    let x = 1.0

    // Reassign 100 times - should not leak memory
    for i in range(100) {
        x = x + 1.0
    }

    assert_eq(x, 101.0)

    print("[PASS] Test 10: Memory leak prevention")
    return 0.0
}

// Test 11: While Loop Scope
fn test_while_loop_scope() -> float {
    let counter = 0.0
    let sum = 0.0

    while counter < 5.0 {
        let iteration_value = counter * 2.0
        sum = sum + iteration_value
        counter = counter + 1.0
    }

    // iteration_value should not be accessible
    assert_eq(sum, 20.0)  // 0+2+4+6+8
    assert_eq(counter, 5.0)

    print("[PASS] Test 11: While loop scope")
    return 0.0
}

// Test 12: Complex Nested Scenario
// Combination of function, loops, blocks, and shadowing
fn complex_function(start: Tensor) -> float {
    let result = start

    for i in [1.0, 2.0, 3.0] {
        {
            let result = result + i  // Shadow result in block

            {
                let result = result * 2.0  // Shadow again
                // Innermost result = (start + i) * 2
            }

            // Middle result restored
        }

        // Outer result should be updated
        result = result + 10.0
    }

    return result
}

fn test_complex_nested_scenario() -> float {
    let val = complex_function(0.0)
    // Each iteration: result = result + 10
    // 3 iterations: 0 + 10 + 10 + 10 = 30
    assert_eq(val, 30.0)

    print("[PASS] Test 12: Complex nested scenario")
    return 0.0
}

// Main test runner
main {
    print("=== Scope Management Tests ===")
    print("")

    let _ = test_basic_block_scope()
    let _ = test_variable_shadowing()
    let _ = test_nested_blocks()
    let _ = test_loop_variable_scope()
    let _ = test_loop_shadowing()
    let _ = test_function_local_scope()
    let _ = test_function_with_blocks()
    let _ = test_early_return()
    let _ = test_if_else_scope()
    let _ = test_memory_leak_prevention()
    let _ = test_while_loop_scope()
    let _ = test_complex_nested_scenario()

    print("")
    print("=== All Scope Tests Passed ===")
}
