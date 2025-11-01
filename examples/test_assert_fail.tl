// Test that assertions fail correctly

main {
    print("Testing assertion failure...")
    print("This should panic with 'Assertion failed!'")

    // This will panic because 5 < 3 is false
    assert(5 < 3)

    print("This should not be reached")
}
