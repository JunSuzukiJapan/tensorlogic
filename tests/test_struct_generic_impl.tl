// Test generic impl blocks

struct Box<T> {
    value: T,
}

impl<T> Box<T> {
    fn new(value: T) -> Box<T> {
        return Box<T> { value: value }
    }
}

struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Pair<T, U> {
        return Pair<T, U> {
            first: first,
            second: second,
        }
    }

    fn make_pair(a: T, b: U) -> Pair<T, U> {
        return Pair<T, U> { first: a, second: b }
    }
}

main {
    print("=== Generic Impl Block Tests ===")
    print("")

    // Test 1: Generic constructor with float
    print("Test 1: Box<float32>::new()")
    let box1 = Box<float32>::new(3.14)
    print("Created Box<float32>")

    // Test 2: Generic constructor with int
    print("")
    print("Test 2: Box<int32>::new()")
    let box2 = Box<int32>::new(42)
    print("Created Box<int32>")

    // Test 3: Pair with two type parameters
    print("")
    print("Test 3: Pair<float32, int32>::new()")
    let pair1 = Pair<float32, int32>::new(1.5, 10)
    print("Created Pair<float32, int32>")

    // Test 4: Alternative constructor
    print("")
    print("Test 4: Pair::make_pair()")
    let pair2 = Pair<int32, bool>::make_pair(100, true)
    print("Created Pair<int32, bool>")

    // Test 5: Mix of different instantiations
    print("")
    print("Test 5: Multiple generic instantiations")
    let b1 = Box<float32>::new(1.0)
    let b2 = Box<int32>::new(2)
    let b3 = Box<bool>::new(false)
    print("Created 3 different Box types")

    print("")
    print("All generic impl tests passed!")
}
