# LLVM Compiler for TensorLogic

TensorLogicプログラムをLLVM IRにコンパイルし、高速実行やネイティブコード生成を可能にします。

## 概要

LLVMコンパイラ機能は、TensorLogicプログラムを以下の形式で出力・実行できます：

1. **JIT実行**: プログラムをLLVM IRにコンパイルし、JIT(Just-In-Time)で実行することで、インタープリターよりも高速に動作
2. **LLVM IRアセンブリ出力**: LLVM中間表現(.llファイル)をテキスト形式で出力
3. **ネイティブアセンブリ出力**: プラットフォーム固有のアセンブリコード(.sファイル)を出力
4. **オブジェクトファイル出力**: リンク可能なオブジェクトファイル(.o / .obj)を生成
5. **静的ライブラリ出力**: 静的リンクライブラリ(.a / .lib)を生成
6. **共有ライブラリ出力**: 動的リンクライブラリ(.so / .dll / .dylib)を生成
7. **実行ファイル出力**: スタンドアロンの実行可能バイナリを生成

## ビルド方法

LLVMフィーチャーを有効にしてビルドします：

```bash
cargo build --features llvm
```

## 使用方法

### 1. JIT実行（インタープリター高速化）

```bash
tl run program.tl --jit
```

プログラムをLLVM JITでコンパイル・実行します。インタープリターよりも高速に動作します。

### 2. LLVM IRアセンブリ出力

```bash
tl run program.tl --emit-llvm output.ll
```

LLVM中間表現をテキスト形式で`output.ll`に出力します。

### 3. ネイティブアセンブリ出力

```bash
tl run program.tl --emit-asm output.s
```

プラットフォーム固有のアセンブリコードを`output.s`に出力します。

**注意**: ネイティブアセンブリ出力は以下のプラットフォームでサポートされています：
- x86_64 (Linux, macOS, Windows)
- aarch64 / arm64 (Linux, Apple Silicon)
- i686

サポートされていないプラットフォームではエラーメッセージが表示されます。

### 4. オブジェクトファイル出力

```bash
tl run program.tl --emit-obj output.o
```

リンク可能なオブジェクトファイルを生成します（.o on Unix、.obj on Windows）。

### 5. 静的ライブラリ出力

```bash
tl run program.tl --emit-lib libmylib.a
```

静的リンクライブラリを生成します（.a on Unix、.lib on Windows）。

**必要なツール**:
- Linux/macOS: `ar` (binutilsに含まれる)
- Windows: `lib.exe` (MSVC) または `ar` (MinGW)

### 6. 共有ライブラリ出力

```bash
tl run program.tl --emit-shared libmylib.so
```

動的リンクライブラリを生成します（.so on Linux、.dll on Windows、.dylib on macOS）。

**必要なツール**:
- Linux: `gcc` または `clang`
- macOS: `clang` (Xcode Command Line Tools)
- Windows: `link.exe` (MSVC) または `gcc` (MinGW)

### 7. 実行ファイル出力

```bash
tl run program.tl --emit-bin myprogram
```

スタンドアロンの実行可能バイナリを生成します。プログラムには`main`ブロックが必要です。

**必要なツール**:
- Linux: `gcc` または `clang`
- macOS: `clang` (Xcode Command Line Tools)
- Windows: `link.exe` (MSVC) または `gcc` (MinGW)

### 8. 最適化レベルの指定

```bash
tl run program.tl --jit --opt-level 3
```

最適化レベルを0〜3で指定できます（デフォルト: 2）：
- 0: 最適化なし
- 1: 少しの最適化
- 2: 標準的な最適化（デフォルト）
- 3: 積極的な最適化

## アーキテクチャ

### モジュール構成

```
src/compiler/
├── mod.rs         # モジュールエントリポイント、オプション定義
├── codegen.rs     # LLVM IR生成
├── jit.rs         # JIT実行エンジン
├── output.rs      # ファイル出力（.ll, .s, .o, .a, .so, バイナリ）
└── linker.rs      # クロスプラットフォームリンカーインフラ
```

### コンパイルフロー

```
TensorLogic AST
    ↓
LLVMCodeGen
    ↓
LLVM IR Module
    ↓
    ├─→ JITCompiler → 実行
    ├─→ OutputWriter → .llファイル
    ├─→ OutputWriter → .sファイル
    ├─→ OutputWriter → .oファイル
    └─→ OutputWriter + Linker → .a / .so / バイナリ
```

## 現在の制限事項

現在のLLVMコンパイラは以下の機能をサポートしています：

### サポート済み
- スカラー型（int, float, bool）
- 基本的な算術演算（+, -, *, /, %）
- 比較演算（==, !=, <, <=, >, >=）
- 制御フロー（if, while, for, loop）
- 関数定義と呼び出し
- 変数の宣言と代入

### 未サポート（インタープリターにフォールバック）
- テンソル演算
- 構造体とメソッド
- 論理プログラミング機能
- 埋め込み（Embedding）
- Pythonインテグレーション

## 将来の拡張

以下の機能が将来的に追加される予定です：

1. **テンソル演算のサポート**: LLVM IRでテンソル演算を表現し、最適化
2. **GPU統合**: LLVM PTXバックエンドを使用したGPUコード生成
3. **より高度なJIT最適化**: ランタイムプロファイリングに基づく動的最適化
4. **インクリメンタルコンパイル**: REPLでの使用を改善
5. **モジュールシステム**: 複数のTensorLogicモジュールをリンクして単一のライブラリ・実行ファイルを生成

## 例

### 簡単な計算プログラム

```tensorlogic
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

main {
    let result = factorial(10);
    print("Factorial of 10 is:", result);
}
```

実行方法：

```bash
# インタープリターで実行
tl run factorial.tl

# JITで実行（高速）
tl run factorial.tl --jit

# LLVM IRを確認
tl run factorial.tl --emit-llvm factorial.ll
cat factorial.ll

# ネイティブアセンブリを生成
tl run factorial.tl --emit-asm factorial.s
cat factorial.s

# オブジェクトファイルを生成
tl run factorial.tl --emit-obj factorial.o

# 静的ライブラリを生成
tl run factorial.tl --emit-lib libfactorial.a

# 共有ライブラリを生成
tl run factorial.tl --emit-shared libfactorial.so

# 実行可能バイナリを生成
tl run factorial.tl --emit-bin factorial
./factorial  # 実行
```

## トラブルシューティング

### LLVMフィーチャーが有効にならない

LLVMフィーチャーを有効にしてビルドしてください：

```bash
cargo build --features llvm
```

### JITコンパイルに失敗する

プログラムに未サポートの機能が含まれている場合、自動的にインタープリターにフォールバックします。
デバッグモードで詳細を確認できます：

```bash
tl run program.tl --jit --debug
```

### ネイティブアセンブリ出力がサポートされていない

現在のプラットフォームが対応していません。LLVM IR出力を代わりに使用してください：

```bash
tl run program.tl --emit-llvm output.ll
```

## 技術詳細

### 使用ライブラリ

- **inkwell**: LLVMのRustバインディング
- **LLVM 18.0**: バックエンドコンパイラ

### 型マッピング

| TensorLogic型 | LLVM型 |
|--------------|--------|
| int          | i64    |
| float        | f64    |
| bool         | i1     |

### 最適化

LLVMの標準的な最適化パスを使用：
- 定数畳み込み
- デッドコード削除
- インライン化
- ループ最適化
- など

## 参考資料

- [LLVM Documentation](https://llvm.org/docs/)
- [Inkwell Documentation](https://thedan64.github.io/inkwell/)
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
