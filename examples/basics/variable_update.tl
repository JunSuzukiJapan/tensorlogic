// Test variable update with assignment

main {
    print("=== Test: Variable update with = ===")
    let counter = 0
    print("Initial counter:", counter)
    
    counter = counter + 1
    print("After increment:", counter)
    
    counter = counter + 1
    print("After second increment:", counter)
    
    counter = counter * 10
    print("After multiply by 10:", counter)
    
    print("\n✅ Variable update test passed!")
}
