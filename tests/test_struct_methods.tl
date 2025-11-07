// Test impl blocks and methods

struct Point {
    x: float32,
    y: float32,
}

impl Point {
    fn new(x: float32, y: float32) -> Point {
        return Point { x: x, y: y }
    }

    fn origin() -> Point {
        return Point { x: 0.0, y: 0.0 }
    }
}

struct Counter {
    value: int32,
}

impl Counter {
    fn new(initial: int32) -> Counter {
        return Counter { value: initial }
    }

    fn zero() -> Counter {
        return Counter { value: 0 }
    }
}

main {
    print("=== Impl Block and Method Tests ===")
    print("")

    // Test 1: Associated function (constructor)
    print("Test 1: Point::new()")
    let p1 = Point::new(10.0, 20.0)
    print("Created point with new()")

    // Test 2: Another associated function
    print("")
    print("Test 2: Point::origin()")
    let p2 = Point::origin()
    print("Created origin point")

    // Test 3: Different struct's methods
    print("")
    print("Test 3: Counter::new()")
    let c1 = Counter::new(42)
    print("Created counter with initial value")

    // Test 4: Zero constructor
    print("")
    print("Test 4: Counter::zero()")
    let c2 = Counter::zero()
    print("Created zero counter")

    // Test 5: Mix of literal and constructor
    print("")
    print("Test 5: Mix literal and constructor")
    let p3 = Point { x: 5.0, y: 5.0 }
    let p4 = Point::new(15.0, 15.0)
    print("Created points both ways")

    print("")
    print("All method tests passed!")
}
