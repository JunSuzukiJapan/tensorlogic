// Test string repetition feature

main {
    print("================================================================================")
    print("String Repetition Test")
    print("================================================================================")
    print("")

    // Test 1: Basic repetition
    print("Test 1: Basic repetition")
    line = "=" * 80
    print(line)
    print("Length:", line)  // Should print the repeated string
    print("")

    // Test 2: Different patterns
    print("Test 2: Different patterns")
    print("  Pattern 1:", "-" * 40)
    print("  Pattern 2:", "abc" * 5)
    print("  Pattern 3:", "xyz" * 3)
    print("")

    // Test 3: Reverse order (integer first)
    print("Test 3: Reverse order (integer * string)")
    print("  Result:", 10 * "*")
    print("  Result:", 5 * "Hello ")
    print("")

    // Test 4: Zero repetition
    print("Test 4: Zero repetition")
    empty = "test" * 0
    print("  Empty string: '", empty, "'")
    print("")

    // Test 5: Single repetition
    print("Test 5: Single repetition")
    single = "word" * 1
    print("  Single:", single)
    print("")

    // Test 6: Large repetition
    print("Test 6: Large repetition (10 times)")
    large = "#" * 10
    print("  Result:", large)
    print("")

    // Test 7: Unicode characters
    print("Test 7: Unicode/Japanese characters")
    japanese = "あ" * 5
    print("  Japanese:", japanese)
    print("")

    // Test 8: Practical use case - separators
    print("Test 8: Practical use - separators")
    separator = "=" * 80
    title_sep = "-" * 80

    print(separator)
    print("IMPORTANT MESSAGE")
    print(title_sep)
    print("This demonstrates practical use of string repetition")
    print(separator)
    print("")

    print("================================================================================")
    print("All string repetition tests completed!")
    print("================================================================================")
}
