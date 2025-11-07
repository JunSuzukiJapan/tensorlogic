// Test boolean reassignment in loop

main {
    let flag = true
    let count = 0

    for i in range(5) {
        if flag {
            count = count + 1
            print("Count:", count)

            if count == 2 {
                flag = false
                print("Flag set to false")
            }
        }
    }

    print("Final count:", count)
}
