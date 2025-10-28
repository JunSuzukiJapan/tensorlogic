// Step 3: テンソル操作

main {
    print("Step 3: テンソル操作")

    for i in range(3) {
        let val = 20.0 + i
        let t = f32::ones([1]) * val
        print("t shape:", shape(t))
    }

    print("成功")
}
