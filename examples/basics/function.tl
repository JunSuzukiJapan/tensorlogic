// 最小限の関数テスト

fn simple_add(a: float16[?, ?], b: float16[?, ?]) -> float16[?, ?] {
    let result = a + b
    return result
}

main {
    print("=== 関数テスト ===")
    
    let x = ones([2, 3])
    let y = ones([2, 3])
    
    print("入力1:", shape(x))
    print("入力2:", shape(y))
    
    let z = simple_add(x, y)
    
    print("出力:", shape(z))
    print("✅ 関数呼び出し成功!")
}
