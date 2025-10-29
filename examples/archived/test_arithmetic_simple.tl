// Simple integer arithmetic test

main {
    print("=== Integer Arithmetic ===")

    let a = 10
    let b = 3

    let add_result = a + b
    let sub_result = a - b
    let mul_result = a * b
    let div_result = a / b
    let mod_result = a % b

    print("a =", a)
    print("b =", b)
    print("a + b =", add_result)
    print("a - b =", sub_result)
    print("a * b =", mul_result)
    print("a / b =", div_result)
    print("a % b =", mod_result)

    print("")
    print("=== Operator Precedence ===")

    let precedence1 = 2 + 3 * 4
    print("2 + 3 * 4 =", precedence1)

    let precedence2 = (2 + 3) * 4
    print("(2 + 3) * 4 =", precedence2)

    print("")
    print("=== Done ===")
}
