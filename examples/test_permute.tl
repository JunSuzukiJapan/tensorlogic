main {
    print("Testing permute")
    let tensor = positional_encoding(6, 4)
    let p = permute(tensor, [1, 0])
    print("permute OK")
}
