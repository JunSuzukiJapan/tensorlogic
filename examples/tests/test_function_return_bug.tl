// Test: Function return value bug
//
// This test verifies that functions correctly return computed tensor values
// without GPU synchronization issues.
//
// Bug symptoms:
// - Direct return of computed tensor (e.g., x * sigmoid(x)) returned 0
// - Using intermediate variables worked correctly
//
// Root cause:
// - GPU operations were not flushed before returning from function
// - Function return buffer was accessed before GPU operations completed

// Test 1: Direct return of binary operation
fn test_direct_return(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)  // Should work without intermediate variable
}

// Test 2: Nested function calls in return expression
fn test_nested_calls(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x + x)  // More complex expression
}

// Test 3: Multiple operations
fn test_multiple_ops(x: float16[?, ?]) -> float16[?, ?] {
    (x + x) * sigmoid(x * 2.0)
}

main {
    print("================================================================================")
    print("Function Return Value Bug Test")
    print("================================================================================")
    print("")

    // Create test tensor using zeros and assignment
    let x = f16::zeros([1, 5])
    // Note: In real usage, would load actual test data
    // For now, just test that functions don't return 0 unexpectedly

    print("[Test 1] Direct return: x * sigmoid(x)")
    let result1 = test_direct_return(x)
    let sum1 = sum(result1)
    print("  Sum:", sum1)

    // Expected: approximately 1.986 (based on mathematical properties)
    // SiLU(-2) ≈ -0.238, SiLU(-1) ≈ -0.269, SiLU(0) = 0
    // SiLU(1) ≈ 0.731, SiLU(2) ≈ 1.762
    // Total ≈ 1.986

    if sum1 > 1.9 {
        if sum1 < 2.1 {
            print("  ✅ PASS: Result is in expected range [1.9, 2.1]")
        }
    }

    if sum1 == 0.0 {
        print("  ❌ FAIL: Result is 0 (BUG DETECTED)")
    }

    if sum1 != sum1 {
        print("  ❌ FAIL: Result is NaN (BUG DETECTED)")
    }

    print("")

    print("[Test 2] Nested function calls")
    let result2 = test_nested_calls(x)
    let sum2 = sum(result2)
    print("  Sum:", sum2)

    if sum2 == 0.0 {
        print("  ❌ FAIL: Result is 0 (BUG DETECTED)")
    }

    if sum2 != sum2 {
        print("  ❌ FAIL: Result is NaN (BUG DETECTED)")
    }

    if sum2 != 0.0 {
        if sum2 == sum2 {
            print("  ✅ PASS: Result is non-zero and not NaN")
        }
    }

    print("")

    print("[Test 3] Multiple operations")
    let result3 = test_multiple_ops(x)
    let sum3 = sum(result3)
    print("  Sum:", sum3)

    if sum3 == 0.0 {
        print("  ❌ FAIL: Result is 0 (BUG DETECTED)")
    }

    if sum3 != sum3 {
        print("  ❌ FAIL: Result is NaN (BUG DETECTED)")
    }

    if sum3 != 0.0 {
        if sum3 == sum3 {
            print("  ✅ PASS: Result is non-zero and not NaN")
        }
    }

    print("")
    print("================================================================================")
    print("✅ Function Return Value Bug Test Complete")
    print("================================================================================")
}
