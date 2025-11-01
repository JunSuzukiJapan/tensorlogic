// Test all assertion functions

main {
    print("=== Testing assert() ===")

    // Test 1: assert with true condition (should pass)
    print("Test 1: assert(true) - should pass")
    assert(true)
    print("  ✓ Passed")

    // Test 2: assert with comparison (should pass)
    print("Test 2: assert(5 > 3) - should pass")
    assert(5 > 3)
    print("  ✓ Passed")

    // Test 3: assert with equality (should pass)
    print("Test 3: assert(10 == 10) - should pass")
    assert(10 == 10)
    print("  ✓ Passed")

    print("\n=== Testing assert_eq() ===")

    // Test 4: assert_eq with integers (should pass)
    print("Test 4: assert_eq(42, 42) - should pass")
    assert_eq(42, 42)
    print("  ✓ Passed")

    // Test 5: assert_eq with floats (should pass)
    print("Test 5: assert_eq(3.14, 3.14) - should pass")
    assert_eq(3.14, 3.14)
    print("  ✓ Passed")

    // Test 6: assert_eq with booleans (should pass)
    print("Test 6: assert_eq(true, true) - should pass")
    assert_eq(true, true)
    print("  ✓ Passed")

    // Test 7: assert_eq with strings (should pass)
    print("Test 7: assert_eq(\"hello\", \"hello\") - should pass")
    assert_eq("hello", "hello")
    print("  ✓ Passed")

    print("\n=== Testing assert_about_eq() ===")

    // Test 8: assert_about_eq with close floats (should pass)
    print("Test 8: assert_about_eq(3.14, 3.141, 0.01) - should pass")
    assert_about_eq(3.14, 3.141, 0.01)
    print("  ✓ Passed")

    // Test 9: assert_about_eq with close integers (should pass)
    print("Test 9: assert_about_eq(100, 101, 2.0) - should pass")
    assert_about_eq(100, 101, 2.0)
    print("  ✓ Passed")

    // Test 10: assert_about_eq with exact match (should pass)
    print("Test 10: assert_about_eq(5.0, 5.0, 0.001) - should pass")
    assert_about_eq(5.0, 5.0, 0.001)
    print("  ✓ Passed")

    print("\n=== Testing with tensors ===")

    // Test 11: assert_eq with identical tensors (should pass)
    print("Test 11: assert_eq with identical f32 tensors - should pass")
    let t1 = f32::ones([2.0, 3.0])
    let t2 = f32::ones([2.0, 3.0])
    assert_eq(t1, t2)
    print("  ✓ Passed")

    // Test 12: assert_about_eq with close tensors (should pass)
    print("Test 12: assert_about_eq with close tensors - should pass")
    let t3 = f32::ones([2.0])
    let t4 = f32::ones([2.0]) * 1.001
    assert_about_eq(t3, t4, 0.01)
    print("  ✓ Passed")

    print("\n✓ All assertion tests passed!")
}
