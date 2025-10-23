main {
    print("Testing array literals with variables")
    print("")

    let dim1 = 3
    let dim2 = 4

    print("dim1 =", dim1)
    print("dim2 =", dim2)
    print("")

    print("Creating array with variables: ones([dim1, dim2])")
    let arr = ones([dim1, dim2])
    print("Success!")
    print("")

    print("Creating nested array with variables: zeros([2, dim1, dim2])")
    let nested = zeros([2, dim1, dim2])
    print("Success!")
    print("")

    print("=== All tests passed! ===")
}
