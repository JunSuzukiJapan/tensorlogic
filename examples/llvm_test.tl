// LLVM Compiler Test Program
// このプログラムはLLVMコンパイラの基本機能をテストします

fn add(a: int, b: int) -> int {
    return a + b;
}

fn multiply(a: int, b: int) -> int {
    return a * b;
}

fn factorial(n: int) -> int {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fn sum_range(n: int) -> int {
    let total = 0;
    for i in range(n) {
        total := total + i;
    }
    return total;
}

main {
    // 基本的な算術演算
    let x = 10;
    let y = 20;
    let sum = add(x, y);
    let product = multiply(x, y);

    // 制御フロー
    if sum > 25 {
        let doubled = sum * 2;
    }

    // ループ
    let total = 0;
    for i in range(10) {
        total := total + i;
    }

    // 再帰関数
    let fact5 = factorial(5);
    let fib7 = fibonacci(7);

    // より複雑な計算
    let range_sum = sum_range(100);
}
