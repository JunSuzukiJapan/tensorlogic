// Test: Vec type operations with method call syntax

main {
    print("===============================================================================")
    print("Vec Type Test - Method Call Syntax")
    print("===============================================================================")
    print("")

    // Test 1: IntVec
    print("Test 1: IntVec operations")
    print("============================================================")
    let int_vec = IntVec::new()
    print("Created empty IntVec, len:", len(int_vec))

    int_vec.push(10)
    int_vec.push(20)
    int_vec.push(30)
    print("After pushing 10, 20, 30, len:", len(int_vec))
    print("IntVec:", int_vec)

    print("get(int_vec, 0):", get(int_vec, 0))
    print("get(int_vec, 1):", get(int_vec, 1))
    print("get(int_vec, 2):", get(int_vec, 2))

    int_vec.set(1, 99)
    print("After int_vec.set(1, 99):", int_vec)

    let popped = int_vec.pop()
    print("Popped value:", popped)
    print("After pop, len:", len(int_vec))
    print("IntVec after pop:", int_vec)
    print("")

    // Test 2: FloatVec
    print("Test 2: FloatVec operations")
    print("============================================================")
    let float_vec = FloatVec::new()
    float_vec.push(1.5)
    float_vec.push(2.5)
    float_vec.push(3.5)
    print("FloatVec after pushes:", float_vec)
    print("get(float_vec, 1):", get(float_vec, 1))
    print("")

    // Test 3: StringVec
    print("Test 3: StringVec operations")
    print("============================================================")
    let str_vec = StringVec::new()
    str_vec.push("hello")
    str_vec.push("world")
    str_vec.push("test")
    print("StringVec:", str_vec)
    print("get(str_vec, 0):", get(str_vec, 0))
    print("get(str_vec, 2):", get(str_vec, 2))
    print("")

    // Test 4: BoolVec
    print("Test 4: BoolVec operations")
    print("============================================================")
    let bool_vec = BoolVec::new()
    bool_vec.push(true)
    bool_vec.push(false)
    bool_vec.push(true)
    print("BoolVec:", bool_vec)
    print("is_empty:", bool_vec.is_empty())
    bool_vec.clear()
    print("After clear, is_empty:", bool_vec.is_empty())
    print("After clear, len:", len(bool_vec))
    print("")

    // Test 5: Array vs Vec comparison
    print("Test 5: Array vs Vec comparison")
    print("============================================================")
    let arr = [1, 2, 3]
    print("Array:", arr)
    print("Array len:", len(arr))
    print("Array get(0):", get(arr, 0))
    print("")

    let vec = IntVec::new()
    vec.push(1)
    vec.push(2)
    vec.push(3)
    print("Vec:", vec)
    print("Vec len:", len(vec))
    print("Vec get(0):", get(vec, 0))
    print("")

    print("===============================================================================")
    print("All Vec tests completed successfully!")
    print("===============================================================================")
}
