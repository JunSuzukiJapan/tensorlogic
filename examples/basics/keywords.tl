// Test that variables can use names that start with or are similar to keywords

main {
    // Variables starting with keywords
    let input = [1.0, 2.0, 3.0]
    let index = [0.0]
    let information = [4.0, 5.0]

    // Variables that could be confused with keywords
    let query_param = [6.0, 7.0]
    let tensor_data = [8.0, 9.0]

    // Use them
    let result = input.add([1.0, 1.0, 1.0])

    print("input:", input)
    print("index:", index)
    print("information:", information)
    print("query_param:", query_param)
    print("tensor_data:", tensor_data)
    print("result:", result)
    print("✓ All keyword-like variable names work!")
}
