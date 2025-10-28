// Step 4: 括弧内での計算

main {
    print("Step 4: 括弧内での計算")

    for i in range(3) {
        let t = f32::ones([1]) * (20.0 + i)
        print("t shape:", shape(t))
    }

    print("成功")
}
