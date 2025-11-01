// Minimal test to isolate the hanging issue

main {
    print("Test: Minimal token generation loop")

    let token_count = 0
    let continue_generation = true

    // Simulate token generation loop
    for i in range(5) {
        print("Iteration ", i, ", continue_generation = ", continue_generation)

        if continue_generation {
            print("  Generating token...")
            token_count = token_count + 1
            print("  Token count: ", token_count)

            // Simulate EOS check
            if token_count == 2 {
                print("  Stopping at token 2")
                continue_generation = false
            }
        } else {
            print("  Skipping (continue_generation = false)")
        }

        print("  After if, continue_generation = ", continue_generation)
    }

    print("Loop completed. Final token_count: ", token_count)
}
