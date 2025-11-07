// Test integer literal parsing

main {
    let x = 2
    let y = 0
    let z = 10

    print("x =", x)
    print("y =", y)
    print("z =", z)

    let a = x == 2
    let b = y == 0
    let c = z == 10

    print("x == 2:", a)
    print("y == 0:", b)
    print("z == 10:", c)

    if a {
        print("x is 2 - correct!")
    }

    if b {
        print("y is 0 - correct!")
    }

    if c {
        print("z is 10 - correct!")
    }
}
