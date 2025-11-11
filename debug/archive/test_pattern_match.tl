// Test pattern matching in TensorLogic

main {
    // Test 1: Simple integer matching
    let x = 1
    let result1 = match x {
        1 => 10,
        2 => 20,
        _ => 30
    }
    print("Test 1 - Integer matching: ", result1)

    // Test 2: String matching
    let s = "hello"
    let result2 = match s {
        "hello" => 1,
        "world" => 2,
        _ => 0
    }
    print("Test 2 - String matching: ", result2)

    // Test 3: Boolean matching
    let b = true
    let result3 = match b {
        true => 100,
        false => 200
    }
    print("Test 3 - Boolean matching: ", result3)

    // Test 4: Variable binding
    let y = 42
    let result4 = match y {
        0 => 0,
        n => n * 2
    }
    print("Test 4 - Variable binding: ", result4)

    // Test 5: Or patterns
    let z = 3
    let result5 = match z {
        1 | 2 | 3 => 999,
        _ => 0
    }
    print("Test 5 - Or pattern: ", result5)

    // Test 6: Guard条件
    let val = 15
    let result6 = match val {
        n if n > 10 => 1000,
        n if n > 5 => 500,
        _ => 100
    }
    print("Test 6 - Guard condition: ", result6)
}
