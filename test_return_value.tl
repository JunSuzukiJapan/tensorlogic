// Test: Function should return value of last expression

fn test_expr_return() -> float32[1] {
    let x = f32::ones([1])
    let y = f32::ones([1])
    x + y
}

fn test_assignment_return() -> float32[1] {
    let x = f32::ones([1])
    let y = f32::ones([1])
    let result = x + y
    result
}

main {
    print("Test 1: Expression as last statement", "")
    let val1 = test_expr_return()
    print("  Result:", val1)

    print("Test 2: Assignment then variable as last statement", "")
    let val2 = test_assignment_return()
    print("  Result:", val2)

    print("All tests passed!", "")
}
