// Test counter increment inside if in loop

main {
    let EOS_TOKEN = 2
    let continue_generation = true
    let token_count = 1
    let token_id = 5

    for i in range(3) {
        if continue_generation {
            print(token_count)
            token_count = token_count + 1

            if token_id == EOS_TOKEN {
                continue_generation = false
            }
        }
    }

    print("Final:")
    print(token_count)
}
