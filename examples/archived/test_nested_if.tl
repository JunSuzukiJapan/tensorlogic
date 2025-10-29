// Test nested if with variable reassignment

main {
    let continue_flag = true
    let count = 0

    for i in range(5) {
        print("Loop", i)

        if continue_flag {
            count = count + 1
            print("  Count:", count)

            if count == 3 {
                continue_flag = false
                print("  Stopping!")
            }
        }
    }

    print("Final count:", count)
}
