// Test that interpreter uses Metal GPU

tensor x: float16[3] = [1.0, 2.0, 3.0]
tensor y: float16[3] = [4.0, 5.0, 6.0]

main {
    result := x + y
    print("x + y =", result)

    matrix_result := x @ y
    print("x @ y =", matrix_result)
}
