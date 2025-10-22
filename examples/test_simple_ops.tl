// Simple test for new operations

main {
    print("Testing zeros")
    let z = zeros([2, 3])
    print("zeros OK")

    print("Testing ones")
    let o = ones([2, 3])
    print("ones OK")

    print("Testing reshape")
    let data = positional_encoding(6, 4)
    let r = reshape(data, [3, 8])
    print("reshape OK")

    print("All tests passed!")
}
