// Test that assertion functions work in main, test, and bench blocks

main {
    print("=== Testing assertions in main block ===")
    assert(true)
    assert_eq(42, 42)
    assert_about_eq(3.14, 3.141, 0.01)
    print("✓ All assertions in main block passed!")
}

test test_assert_in_test_block {
    print("\n=== Testing assertions in test block ===")

    // Test assert()
    print("Test: assert(5 > 3)")
    assert(5 > 3)
    print("  ✓ Passed")

    // Test assert_eq()
    print("Test: assert_eq(100, 100)")
    assert_eq(100, 100)
    print("  ✓ Passed")

    // Test assert_about_eq()
    print("Test: assert_about_eq(1.0, 1.001, 0.01)")
    assert_about_eq(1.0, 1.001, 0.01)
    print("  ✓ Passed")

    // Test with tensors
    print("Test: assert_eq with tensors")
    let t1 = f32::ones([2.0, 2.0])
    let t2 = f32::ones([2.0, 2.0])
    assert_eq(t1, t2)
    print("  ✓ Passed")

    print("✓ All assertions in test block passed!")
}

bench bench_assert_in_bench_block {
    print("\n=== Testing assertions in bench block ===")

    // Verify assertions work in bench blocks too
    print("Bench: assert(true)")
    assert(true)
    print("  ✓ Passed")

    print("Bench: assert_eq(\"hello\", \"hello\")")
    assert_eq("hello", "hello")
    print("  ✓ Passed")

    print("Bench: assert_about_eq(2.0, 2.0001, 0.001)")
    assert_about_eq(2.0, 2.0001, 0.001)
    print("  ✓ Passed")

    print("✓ All assertions in bench block passed!")
}
