// Test that variables can use names that start with or are similar to keywords

main {
    // Variables starting with keywords
    tensor input: float16[3] = [1.0, 2.0, 3.0]
    tensor index: float16[1] = [0.0]
    tensor information: float16[2] = [4.0, 5.0]

    // Variables that could be confused with keywords
    tensor query_param: float16[2] = [6.0, 7.0]
    tensor tensor_data: float16[2] = [8.0, 9.0]

    // Use them
    tensor result: float16[3] = input + [1.0, 1.0, 1.0]

    print("input:", input)
    print("index:", index)
    print("information:", information)
    print("query_param:", query_param)
    print("tensor_data:", tensor_data)
    print("result:", result)
    print("✓ All keyword-like variable names work!")
}
