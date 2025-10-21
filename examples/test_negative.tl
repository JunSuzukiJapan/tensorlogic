// Test negative number support

tensor neg_int: float16[1] = [-5.0]
tensor neg_float: float16[1] = [-3.14]
tensor mixed: float16[4] = [-1.0, 2.0, -3.5, 4.2]

main {
    print("Negative integer:", neg_int)
    print("Negative float:", neg_float)
    print("Mixed values:", mixed)

    result := neg_int + neg_float
    print("Sum:", result)
}
