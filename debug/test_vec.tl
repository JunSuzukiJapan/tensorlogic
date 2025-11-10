// Test: Vec type operations

main {
    print("===============================================================================")
    print("Vec Type Test")
    print("===============================================================================")
    print("")

    // Test 1: IntVec
    print("Test 1: IntVec operations")
    print("============================================================")
    let int_vec = new_int_vec()
    print("Created empty IntVec, len:", len(int_vec))

    vec_push(int_vec, 10)
    vec_push(int_vec, 20)
    vec_push(int_vec, 30)
    print("After pushing 10, 20, 30, len:", len(int_vec))
    print("IntVec:", int_vec)

    print("get(int_vec, 0):", get(int_vec, 0))
    print("get(int_vec, 1):", get(int_vec, 1))
    print("get(int_vec, 2):", get(int_vec, 2))

    vec_set(int_vec, 1, 99)
    print("After vec_set(int_vec, 1, 99):", int_vec)

    let popped = vec_pop(int_vec)
    print("Popped value:", popped)
    print("After pop, len:", len(int_vec))
    print("IntVec after pop:", int_vec)
    print("")

    // Test 2: FloatVec
    print("Test 2: FloatVec operations")
    print("============================================================")
    let float_vec = new_float_vec()
    vec_push(float_vec, 1.5)
    vec_push(float_vec, 2.5)
    vec_push(float_vec, 3.5)
    print("FloatVec after pushes:", float_vec)
    print("get(float_vec, 1):", get(float_vec, 1))
    print("")

    // Test 3: StringVec
    print("Test 3: StringVec operations")
    print("============================================================")
    let str_vec = new_string_vec()
    vec_push(str_vec, "hello")
    vec_push(str_vec, "world")
    vec_push(str_vec, "test")
    print("StringVec:", str_vec)
    print("get(str_vec, 0):", get(str_vec, 0))
    print("get(str_vec, 2):", get(str_vec, 2))
    print("")

    // Test 4: BoolVec
    print("Test 4: BoolVec operations")
    print("============================================================")
    let bool_vec = new_bool_vec()
    vec_push(bool_vec, true)
    vec_push(bool_vec, false)
    vec_push(bool_vec, true)
    print("BoolVec:", bool_vec)
    print("is_empty:", vec_is_empty(bool_vec))
    vec_clear(bool_vec)
    print("After clear, is_empty:", vec_is_empty(bool_vec))
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

    let vec = new_int_vec()
    vec_push(vec, 1)
    vec_push(vec, 2)
    vec_push(vec, 3)
    print("Vec:", vec)
    print("Vec len:", len(vec))
    print("Vec get(0):", get(vec, 0))
    print("")

    print("===============================================================================")
    print("All Vec tests completed successfully!")
    print("===============================================================================")
}
