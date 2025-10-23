// Test all new features

main {
    print("=== Test 1: let statement in main block ===")
    let x = 5
    let y = 10
    print("x =", x)
    print("y =", y)
    print("x + y =", x + y)
    
    print("\n=== Test 2: Comparison operators ===")
    print("5 == 5:", 5 == 5)
    print("5 != 3:", 5 != 3)
    print("5 < 10:", 5 < 10)
    print("5 > 3:", 5 > 3)
    
    print("\n=== Test 3: Logical operators ===")
    print("true && true:", true && true)
    print("true && false:", true && false)
    print("true || false:", true || false)
    print("false || false:", false || false)
    
    print("\n=== Test 4: String concatenation and comparison ===")
    let msg = "Hello" + " " + "World"
    print("Concatenated:", msg)
    print("Hello == Hello:", "Hello" == "Hello")
    print("Hello != World:", "Hello" != "World")
    
    print("\n=== Test 5: Escape sequences ===")
    print("Line 1\nLine 2\nLine 3")
    print("Tab\tseparated\tvalues")
    
    print("\n✅ All tests completed!")
}
