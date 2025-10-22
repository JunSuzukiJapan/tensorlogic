// Test variable redefinition error

main {
    print("=== Test: Variable cannot be redefined with let ===")
    let x = 5
    print("x =", x)
    
    print("\nTrying to redefine x with 'let'...")
    let x = 10  // This should error
    print("x =", x)
}
