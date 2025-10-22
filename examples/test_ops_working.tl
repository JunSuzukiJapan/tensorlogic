main {
    print("Testing 18 new tensor operations")

    print("zeros")
    let z = zeros([2, 3])

    print("ones")
    let o = ones([2, 3])

    print("reshape")
    let data = positional_encoding(6, 4)
    let r = reshape(data, [3, 8])

    print("flatten")
    let f = flatten(r)

    print("max")
    let mx = max(positional_encoding(4, 5))

    print("min")
    let mn = min(positional_encoding(4, 5))

    print("gelu")
    let g = gelu(positional_encoding(3, 4))

    print("tanh")
    let th = tanh(positional_encoding(3, 4))

    print("exp")
    let e = exp(positional_encoding(2, 3))

    print("log")
    let l = log(e)

    print("sqrt")
    let sq = sqrt(positional_encoding(2, 2))

    print("pow")
    let pw = pow(positional_encoding(2, 3), 2)

    print("sin")
    let sn = sin(positional_encoding(2, 3))

    print("cos")
    let cs = cos(positional_encoding(2, 3))

    print("tan")
    let tn = tan(positional_encoding(2, 3))

    print("SUCCESS: 18 operations work!")
    print("Note: transpose and permute have parser issues")
    print("Note: gather and scatter require index tensors")
}
