// Test basic struct definition and instantiation

struct Point {
    x: float32,
    y: float32,
}

struct Rectangle {
    width: float32,
    height: float32,
    x: float32,
    y: float32,
}

main {
    print("=== Basic Struct Tests ===")
    print("")

    // Test 1: Struct literal
    print("Test 1: Create Point with struct literal")
    let p1 = Point { x: 3.0, y: 4.0 }
    print("Created point")

    // Test 2: Field access
    print("Test 2: Field access")
    print("p1.x should be 3.0")
    print("p1.y should be 4.0")

    // Test 3: Multiple struct instances
    print("")
    print("Test 3: Multiple instances")
    let p2 = Point { x: 0.0, y: 0.0 }
    let p3 = Point { x: 10.0, y: 10.0 }
    print("Created 3 points total")

    // Test 4: Struct with more fields
    print("")
    print("Test 4: Rectangle struct")
    let rect = Rectangle {
        width: 100.0,
        height: 50.0,
        x: 10.0,
        y: 20.0,
    }
    print("Created rectangle")

    print("")
    print("All basic struct tests passed!")
}
