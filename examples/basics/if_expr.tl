// Test if-expression (value-returning if)

main {
    print("Testing if-expression:")
    print("")

    // Test 1: Basic if-else with comparison
    let x = 10
    let y = 5
    print("Test 1: Basic comparison")
    print("x =", x, ", y =", y)

    let max_val = if x > y {
        x
    } else {
        y
    }
    print("max(x, y) =", max_val)
    print("Expected: 10")
    print("")

    // Test 2: If-else with integer condition (0 = false, non-zero = true)
    print("Test 2: Integer as condition")
    let flag = 1
    let message = if flag {
        "flag is true"
    } else {
        "flag is false"
    }
    print("flag =", flag)
    print("message:", message)
    print("Expected: flag is true")
    print("")

    // Test 3: If-else with zero condition
    print("Test 3: Zero as false")
    let zero_flag = 0
    let zero_message = if zero_flag {
        "zero is true"
    } else {
        "zero is false"
    }
    print("zero_flag =", zero_flag)
    print("message:", zero_message)
    print("Expected: zero is false")
    print("")

    // Test 4: Nested if-expression
    print("Test 4: Nested if-expression")
    let a = 15
    let b = 10
    let c = 20

    let max_of_three = if a > b {
        if a > c {
            a
        } else {
            c
        }
    } else {
        if b > c {
            b
        } else {
            c
        }
    }
    print("a =", a, ", b =", b, ", c =", c)
    print("max(a, b, c) =", max_of_three)
    print("Expected: 20")
    print("")

    // Test 5: If-expression with tensor operations
    print("Test 5: If-expression with tensors")
    let seq_len = 1

    let result = if seq_len == 1 {
        [100.0]
    } else {
        [200.0, 300.0]
    }

    print("seq_len =", seq_len)
    print("result:", result)
    print("Expected: [100.0]")
    print("")

    // Test 6: If-expression with float condition
    print("Test 6: Float as condition")
    let f = 3.14
    let f_message = if f {
        "float is non-zero"
    } else {
        "float is zero"
    }
    print("f =", f)
    print("message:", f_message)
    print("Expected: float is non-zero")
    print("")

    print("All tests completed!")
}
