// Test destructors in function scope

struct ScopedResource {
    id: int32,
}

impl Drop for ScopedResource {
    fn drop(self) -> void {
        print("Dropping ScopedResource with id:")
        print(self.id)
    }
}

impl ScopedResource {
    fn new(id: int32) -> ScopedResource {
        print("Creating ScopedResource with id:")
        print(id)
        return ScopedResource { id: id }
    }
}

fn test_function(x: int32) -> int32 {
    print("--- Inside test_function ---")
    let r1 = ScopedResource::new(1)
    let r2 = ScopedResource::new(2)

    print("Function resources created")
    print("--- Exiting test_function ---")

    return x
    // Destructors should be called here before function returns
}

main {
    print("=== Function Scope Destructor Test ===")
    print("")

    print("Test 1: Main scope resource")
    let main_r = ScopedResource::new(100)
    print("")

    print("Test 2: Calling test_function")
    let result = test_function(42)
    print("Returned from test_function")
    print("")

    print("Test 3: Another main scope resource")
    let main_r2 = ScopedResource::new(200)
    print("")

    print("=== End of main ===")
    // Expected destructor call order:
    // Inside function: r2 (id=2), then r1 (id=1)
    // At end of main: main_r2 (id=200), then main_r (id=100)
}
