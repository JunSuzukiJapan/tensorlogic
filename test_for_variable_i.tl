// 徹底的にforループ内でiが使えるかテスト

main {
    print("=== forループ内での変数i使用テスト ===")
    print("")

    // Test 1: iを直接print
    print("[Test 1] iを直接print:")
    for i in range(3) {
        print("  i =", i)
    }
    print("  ✓ 成功")
    print("")

    // Test 2: iを計算に使う
    print("[Test 2] iを計算に使う:")
    for i in range(3) {
        let result = i + 10
        print("  i + 10 =", result)
    }
    print("  ✓ 成功")
    print("")

    // Test 3: iを浮動小数点計算に使う
    print("[Test 3] iを浮動小数点計算に使う:")
    for i in range(3) {
        let result = 20.0 + i
        print("  20.0 + i =", result)
    }
    print("  ✓ 成功")
    print("")

    // Test 4: iをテンソル作成に使う
    print("[Test 4] iをテンソル作成に使う:")
    for i in range(3) {
        let t = f32::ones([1]) * i
        print("  f32::ones([1]) * i, shape:", shape(t))
    }
    print("  ✓ 成功")
    print("")

    // Test 5: iを計算してからテンソルに使う
    print("[Test 5] iを計算してからテンソルに使う:")
    for i in range(3) {
        let val = 20.0 + i
        let t = f32::ones([1]) * val
        print("  val =", val, ", shape:", shape(t))
    }
    print("  ✓ 成功")
    print("")

    // Test 6: 括弧内でiを計算
    print("[Test 6] 括弧内でiを計算:")
    for i in range(3) {
        let t = f32::ones([1]) * (20.0 + i)
        print("  shape:", shape(t))
    }
    print("  ✓ 成功")
    print("")

    // Test 7: iをscatterのインデックスに使う
    print("[Test 7] iをscatterのインデックスに使う:")
    let cache = f32::zeros([10, 4])
    for i in range(3) {
        let pos = f32::ones([1]) * i
        let data = f32::ones([1, 4])
        cache = scatter(cache, 0, pos, data)
    }
    print("  ✓ 成功")
    print("")

    print("=== すべてのテスト完了 ===")
}
