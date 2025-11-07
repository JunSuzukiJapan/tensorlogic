// Test generic struct definitions

struct Container<T> {
    value: T,
    count: int32,
}

struct Pair<T, U> {
    first: T,
    second: U,
}

struct Triple<A, B, C> {
    a: A,
    b: B,
    c: C,
}

main {
    print("=== Generic Struct Tests ===")
    print("")

    // Test 1: Generic with scalar type
    print("Test 1: Container<float32>")
    let c1 = Container<float32> {
        value: 42.0,
        count: 1,
    }
    print("Created container with float32")

    // Test 2: Generic with integer
    print("")
    print("Test 2: Container<int32>")
    let c2 = Container<int32> {
        value: 100,
        count: 2,
    }
    print("Created container with int32")

    // Test 3: Pair with two different types
    print("")
    print("Test 3: Pair<float32, int32>")
    let pair = Pair<float32, int32> {
        first: 3.14,
        second: 42,
    }
    print("Created pair with mixed types")

    // Test 4: Triple with three types
    print("")
    print("Test 4: Triple<int32, float32, bool>")
    let triple = Triple<int32, float32, bool> {
        a: 10,
        b: 2.5,
        c: true,
    }
    print("Created triple")

    print("")
    print("All generic struct tests passed!")
}
