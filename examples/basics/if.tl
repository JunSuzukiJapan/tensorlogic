// Simple if test

main {
    print("Testing if statement:")

    tensor x: float16[1] = [5.0]
    tensor y: float16[1] = [3.0]

    print("x =", x[0])
    print("y =", y[0])

    if x[0] > y[0] {
        print("x > y: TRUE")
        tensor result: float16[1] = [1.0]
        print("result =", result[0])
    }

    print("Done!")
}
