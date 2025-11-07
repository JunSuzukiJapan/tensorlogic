// Test Drop trait and destructor functionality

struct Resource {
    name: string,
    value: int32,
}

impl Drop for Resource {
    fn drop(self) -> void {
        print("Destroying resource:")
        print(self.name)
    }
}

impl Resource {
    fn new(name: string, value: int32) -> Resource {
        print("Creating resource:")
        print(name)
        return Resource {
            name: name,
            value: value,
        }
    }
}

struct Counter {
    count: int32,
}

impl Drop for Counter {
    fn drop(self) -> void {
        print("Counter dropped with value:")
        print(self.count)
    }
}

impl Counter {
    fn new(initial: int32) -> Counter {
        return Counter { count: initial }
    }
}

main {
    print("=== Destructor Tests ===")
    print("")

    // Test 1: Basic destructor call
    print("Test 1: Basic destructor")
    let r1 = Resource::new("Resource1", 100)
    print("Resource created")
    print("")

    // Test 2: Multiple resources
    print("Test 2: Multiple resources")
    let r2 = Resource::new("Resource2", 200)
    let r3 = Resource::new("Resource3", 300)
    print("Multiple resources created")
    print("")

    // Test 3: Different struct type
    print("Test 3: Different struct type (Counter)")
    let c1 = Counter::new(42)
    let c2 = Counter::new(99)
    print("Counters created")
    print("")

    print("=== End of main - destructors should be called in reverse order ===")
    // Expected destructor call order (LIFO):
    // 1. Counter dropped with value: 99
    // 2. Counter dropped with value: 42
    // 3. Destroying resource: Resource3
    // 4. Destroying resource: Resource2
    // 5. Destroying resource: Resource1
}
