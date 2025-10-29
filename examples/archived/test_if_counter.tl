// Test counter inside if statement

main {
    let count = 0
    let flag = true

    for i in range(5) {
        if flag {
            count = count + 1
            print(count)
        }
    }

    print("Final:")
    print(count)
}
