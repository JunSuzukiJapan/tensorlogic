// Test integer arithmetic and operator precedence

main {
    print("=== Integer Arithmetic Tests ===")
    print("")

    // Basic arithmetic
    print("[1] Basic Arithmetic:")
    let a = 10
    let b = 3
    print("  a = 10, b = 3")
    print("  a + b =", a + b)  // 13
    print("  a - b =", a - b)  // 7
    print("  a * b =", a * b)  // 30
    print("  a / b =", a / b)  // 3 (integer division)
    print("  a % b =", a % b)  // 1 (modulo)
    print("")

    // Operator precedence: multiplication before addition
    print("[2] Operator Precedence (multiplication before addition):")
    let result1 = 2 + 3 * 4
    print("  2 + 3 * 4 =", result1)  // Should be 14, not 20
    print("")

    // Operator precedence: division before subtraction
    print("[3] Operator Precedence (division before subtraction):")
    let result2 = 20 - 10 / 2
    print("  20 - 10 / 2 =", result2)  // Should be 15, not 5
    print("")

    // Parentheses override precedence
    print("[4] Parentheses Override:")
    let result3 = (2 + 3) * 4
    print("  (2 + 3) * 4 =", result3)  // Should be 20
    let result4 = (20 - 10) / 2
    print("  (20 - 10) / 2 =", result4)  // Should be 5
    print("")

    // Modulo with precedence
    print("[5] Modulo Operator:")
    let result5 = 17 % 5
    print("  17 % 5 =", result5)  // 2
    let result6 = 10 + 7 % 3
    print("  10 + 7 % 3 =", result6)  // Should be 11 (7%3=1, then 10+1)
    print("")

    // Complex expression
    print("[6] Complex Expression:")
    let result7 = 2 + 3 * 4 - 10 / 2
    print("  2 + 3 * 4 - 10 / 2 =", result7)  // 2 + 12 - 5 = 9
    print("")

    // Comparison operators
    print("[7] Comparison Operators:")
    let result8 = 5 + 3
    print("  5 + 3 == 8:", result8 == 8)  // true
    print("  5 + 3 > 7:", result8 > 7)    // true
    print("  5 + 3 < 10:", result8 < 10)  // true
    print("")

    // Negative numbers
    print("[8] Negative Numbers:")
    let neg = -5
    let result9 = neg + 10
    print("  -5 + 10 =", result9)  // 5
    let result10 = neg * 3
    print("  -5 * 3 =", result10)  // -15
    print("")

    // Token counting use case
    print("[9] Token Counting (like in chat demo):")
    let token_count = 1
    token_count = token_count + 1
    print("  Initial: 1")
    print("  After +1:", token_count)  // 2
    token_count = token_count + 1
    print("  After +1:", token_count)  // 3
    print("")

    print("=== All Tests Complete ===")
}
