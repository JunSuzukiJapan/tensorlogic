// Test example with test blocks

main {
    print("This is the main block")
    let x = f32::ones([2.0, 3.0])
    print("Main execution complete")
}

test simple_test {
    print("Running simple_test")
    let a = f32::ones([2.0])
    print("Test passed")
}

test another_test {
    print("Running another_test")
    let b = f32::zeros([3.0, 3.0])
    print("Another test passed")
}

bench simple_bench {
    print("Running benchmark")
    let x = f32::ones([100.0, 100.0])
    let y = f32::ones([100.0, 100.0])
    let z = linear(x, y)
}
