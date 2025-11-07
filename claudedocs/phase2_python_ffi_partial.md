# Phase 2: Python FFI Integration - Partial Complete ⏳

**Date**: 2025-10-21
**Status**: ⏳ Partial (AST & Parser Complete, Interpreter TODO)
**Build**: ✅ Successful
**Tests**: ✅ All parser tests passing (5/5)

---

## Summary

Phase 2の初期実装が完了しました：

- ✅ AST拡張（PythonImport、PythonCall）
- ✅ Parser更新（python import、python.call()）
- ✅ Lexer更新（python、import、as、.）
- ✅ Visitor実装
- ✅ Typecheck対応
- ✅ インタープリタのスタブ実装
- ✅ パーサーテスト（5つ全てパス）
- ⏳ 実際のPython実行機能（Phase 3で実装予定）

---

## 実装した機能

### 1. AST拡張

**[src/ast/mod.rs](../src/ast/mod.rs)**に以下のノードを追加：

```rust
// Statement列挙型に追加
Statement::PythonImport {
    module: String,      // "numpy"
    alias: Option<String>, // Some("np")
}

// TensorExpr列挙型に追加
TensorExpr::PythonCall {
    function: String,     // "np.sum"
    args: Vec<TensorExpr>, // [x, y]
}
```

### 2. Parser更新

**文法ルール追加** ([src/parser/grammar.pest](../src/parser/grammar.pest)):

```pest
// Python import: python import module [as alias]
python_import = { "python" ~ "import" ~ python_module ~ ("as" ~ identifier)? }
python_module = @{ identifier ~ ("." ~ identifier)* }

// Python function call: python.call("function_name", args)
python_call = { "python" ~ "." ~ "call" ~ "(" ~ string_literal ~ ("," ~ tensor_list)? ~ ")" }
```

**パーサー実装追加** ([src/parser/mod.rs](../src/parser/mod.rs)):

```rust
fn parse_python_import(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
    let mut inner = pair.into_inner();
    let module = inner.next().ok_or_else(...)?.as_str().to_string();
    let alias = inner.next().map(|p| p.as_str().to_string());
    Ok(Statement::PythonImport { module, alias })
}

fn parse_python_call(pair: pest::iterators::Pair<Rule>) -> Result<TensorExpr, ParseError> {
    let mut inner = pair.into_inner();
    let function = Self::parse_string_literal(inner.next().ok_or_else(...)?)?;
    let args = if let Some(tensor_list) = inner.next() {
        Self::parse_tensor_list(tensor_list)?
    } else {
        Vec::new()
    };
    Ok(TensorExpr::PythonCall { function, args })
}
```

### 3. Lexer更新

**新しいトークン** ([src/lexer/mod.rs](../src/lexer/mod.rs)):

```rust
pub enum TokenType {
    // ... 既存のトークン
    Python,   // "python"
    Import,   // "import"
    As,       // "as"
    Dot,      // "."
    // ...
}
```

**キーワード認識**:
- `python`, `import`, `as` を予約語リストに追加
- `.` (ドット) トークンのサポート

### 4. Visitor実装

**[src/ast/visitor.rs](../src/ast/visitor.rs)** に追加：

```rust
// TensorExpr::PythonCall の処理
TensorExpr::PythonCall { args, .. } => {
    for arg in args {
        visitor.visit_tensor_expr(arg)?;
    }
    Ok(())
}

// Statement::PythonImport の処理
Statement::PythonImport { .. } => {
    // No sub-expressions to visit
    Ok(())
}
```

### 5. インタープリタ対応

**[src/interpreter/mod.rs](../src/interpreter/mod.rs)** にスタブ実装：

```rust
// python import文の処理
Statement::PythonImport { module, alias } => {
    let name = alias.as_ref().unwrap_or(module);
    println!("Python import: {} (as {})", module, name);
    // Phase 3: Store imported modules in environment
    Ok(())
}

// python.call()の処理
TensorExpr::PythonCall { function, args } => {
    println!("Python call: {}({} args)", function, args.len());
    // Phase 3: Actually call Python function via PyO3
    Err(RuntimeError::NotImplemented(
        format!("Python function calls not yet implemented: {}", function)
    ))
}
```

### 6. Typecheck対応

**[src/typecheck/mod.rs](../src/typecheck/mod.rs)** に追加：

```rust
TensorExpr::PythonCall { .. } => {
    // TODO: Python function type inference
    // Phase 3: Infer return type from Python function signature
    Ok(TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Dynamic],
    ))
}
```

---

## テスト結果

### パーサーテスト ([tests/python_parser_test.rs](../tests/python_parser_test.rs))

✅ **5つ全てパス**:

1. ✅ `test_parse_python_import` - `python import numpy as np`
2. ✅ `test_parse_python_import_without_alias` - `python import torch`
3. ✅ `test_parse_python_call` - `python.call("np.sum", x)`
4. ✅ `test_parse_python_call_multiple_args` - `python.call("np.add", x, y)`
5. ✅ `test_parse_combined_python_integration` - 複合テスト

### サンプルコード

**[examples/python_integration_test.tl](../examples/python_integration_test.tl)**:

```tensorlogic
main {
    // Python import statement
    python import numpy as np
    python import torch

    // Create a tensor
    tensor x: float16[3] = [1.0, 2.0, 3.0]

    // Python function call (Phase 3で実装予定)
    // tensor y = python.call("np.sum", x)

    print("Python integration test loaded successfully")
}
```

---

## 構文例

### Python Import

```tensorlogic
// エイリアス付きインポート
python import numpy as np
python import torch as th

// エイリアスなしインポート
python import scipy
python import matplotlib.pyplot
```

### Python Call

```tensorlogic
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[3] = [4.0, 5.0, 6.0]

    // NumPy関数の呼び出し
    tensor sum_x: float16[1] = python.call("np.sum", x)
    tensor mean_x: float16[1] = python.call("np.mean", x)

    // 複数引数
    tensor added: float16[3] = python.call("np.add", x, y)
    tensor multiplied: float16[3] = python.call("np.multiply", x, 2.0)

    // PyTorch関数
    tensor softmax: float16[3] = python.call("torch.softmax", x, 0)
}
```

---

## ファイル変更まとめ

### 変更したファイル (6個)

1. **[src/ast/mod.rs](../src/ast/mod.rs)** - AST拡張
   - `Statement::PythonImport` 追加
   - `TensorExpr::PythonCall` 追加

2. **[src/parser/grammar.pest](../src/parser/grammar.pest)** - 文法ルール
   - `python_import` ルール追加
   - `python_module` ルール追加
   - `python_call` ルール追加
   - `statement` に `python_import` 追加
   - `tensor_term` に `python_call` 追加
   - `reserved_keyword` に `python`, `import`, `as` 追加

3. **[src/parser/mod.rs](../src/parser/mod.rs)** - パーサー実装
   - `parse_python_import()` 関数追加 (13行)
   - `parse_python_call()` 関数追加 (16行)
   - `parse_statement()` に `python_import` ケース追加
   - `parse_tensor_term()` に `python_call` ケース追加

4. **[src/ast/visitor.rs](../src/ast/visitor.rs)** - Visitor実装
   - `walk_tensor_expr()` に `PythonCall` ケース追加
   - `walk_statement()` に `PythonImport` ケース追加

5. **[src/lexer/mod.rs](../src/lexer/mod.rs)** - Lexer更新
   - `TokenType::Python`, `Import`, `As`, `Dot` 追加
   - キーワード認識に `python`, `import`, `as` 追加
   - `.` (ドット) トークン処理追加

6. **[src/interpreter/mod.rs](../src/interpreter/mod.rs)** - インタープリタスタブ
   - `execute_statement()` に `PythonImport` ケース追加
   - `eval_expr()` に `PythonCall` ケース追加

7. **[src/typecheck/mod.rs](../src/typecheck/mod.rs)** - 型チェック対応
   - `infer_expr_type()` に `PythonCall` ケース追加

### 新規作成ファイル (2個)

1. **[tests/python_parser_test.rs](../tests/python_parser_test.rs)** - パーサーテスト (124行)
2. **[examples/python_integration_test.tl](../examples/python_integration_test.tl)** - サンプルコード

---

## ビルド結果

```bash
$ cargo build
   Compiling tensorlogic v0.1.0
warning: use of deprecated method `rand::Rng::gen`
warning: unused variable: `size`
warning: `tensorlogic` (lib) generated 2 warnings
    Finished `dev` profile [unoptimized + debuginfo]
```

✅ **ビルド成功** (警告のみ、エラーなし)

```bash
$ cargo test --test python_parser_test
running 5 tests
test test_parse_python_import ... ok
test test_parse_python_import_without_alias ... ok
test test_parse_python_call ... ok
test test_parse_python_call_multiple_args ... ok
test test_parse_combined_python_integration ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured
```

✅ **全テストパス**

---

## Phase 2 残りのタスク

### まだ実装していない機能

❌ **Python実行エンジン統合**:
- PyO3を使った実際のPython関数呼び出し
- インポートされたモジュールの管理
- Tensor ↔ NumPy自動変換
- エラーハンドリングとPythonエラーの伝播

❌ **変数共有**:
- TensorLogic変数をPythonに渡す
- Python変数をTensorLogicで使う
- スコープ管理

❌ **型推論の改善**:
- Python関数のシグネチャからの型推論
- NumPy配列の型とshapeの取得

---

## Phase 3の計画

Phase 3では実際にPythonコードを実行できるようにします：

### 1. Python実行環境の埋め込み

```rust
#[cfg(feature = "python")]
use pyo3::prelude::*;

struct PythonEnvironment {
    gil: GILGuard,
    globals: Py<PyDict>,
    modules: HashMap<String, Py<PyModule>>,
}

impl PythonEnvironment {
    fn new() -> Self {
        Python::with_gil(|py| {
            let globals = PyDict::new(py).into();
            PythonEnvironment {
                gil: py.allow_threads(),
                globals,
                modules: HashMap::new(),
            }
        })
    }

    fn import_module(&mut self, module: &str, alias: Option<&str>) -> Result<()> {
        Python::with_gil(|py| {
            let module_obj = py.import(module)?;
            let name = alias.unwrap_or(module);
            self.modules.insert(name.to_string(), module_obj.into());
            self.globals.as_ref(py).set_item(name, module_obj)?;
            Ok(())
        })
    }

    fn call_function(&self, function: &str, args: Vec<Tensor>) -> Result<Tensor> {
        Python::with_gil(|py| {
            // 関数名をパース ("np.sum" -> module="np", func="sum")
            let parts: Vec<&str> = function.split('.').collect();

            // モジュールから関数を取得
            let module = self.globals.as_ref(py).get_item(parts[0])?;
            let func = module.getattr(parts[1])?;

            // TensorをNumPy配列に変換
            let py_args: Vec<PyObject> = args.iter()
                .map(|t| tensor_to_numpy(py, t))
                .collect::<Result<Vec<_>>>()?;

            // Python関数を呼び出し
            let result = func.call1(PyTuple::new(py, py_args))?;

            // NumPy配列をTensorに変換
            numpy_to_tensor(py, result)
        })
    }
}
```

### 2. Interpreterへの統合

```rust
pub struct Interpreter {
    env: Environment,
    logic_engine: LogicEngine,
    neural_engine: NeuralEngine,
    #[cfg(feature = "python")]
    python_env: Option<PythonEnvironment>, // Phase 3で追加
}

impl Interpreter {
    fn execute_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
        match stmt {
            Statement::PythonImport { module, alias } => {
                #[cfg(feature = "python")]
                {
                    self.python_env
                        .get_or_insert_with(PythonEnvironment::new)
                        .import_module(module, alias.as_deref())
                        .map_err(|e| RuntimeError::PythonError(e.to_string()))?;
                }
                Ok(())
            }
            // ...
        }
    }

    fn eval_expr(&mut self, expr: &TensorExpr) -> RuntimeResult<Value> {
        match expr {
            TensorExpr::PythonCall { function, args } => {
                #[cfg(feature = "python")]
                {
                    let tensor_args: Vec<Tensor> = args.iter()
                        .map(|arg| self.eval_expr(arg)?.as_tensor_f16())
                        .collect::<Result<Vec<_>, _>>()?;

                    let result = self.python_env
                        .as_ref()
                        .ok_or_else(|| RuntimeError::PythonError(
                            "Python environment not initialized".to_string()
                        ))?
                        .call_function(function, tensor_args)
                        .map_err(|e| RuntimeError::PythonError(e.to_string()))?;

                    Ok(Value::Tensor(result))
                }
                #[cfg(not(feature = "python"))]
                Err(RuntimeError::NotImplemented(
                    "Python integration not enabled (compile with --features python)".to_string()
                ))
            }
            // ...
        }
    }
}
```

### 3. テスト

```rust
#[test]
#[cfg(feature = "python")]
fn test_python_numpy_sum() {
    let source = r#"
main {
    python import numpy as np

    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor sum_x: float16[1] = python.call("np.sum", x)

    print("Sum:", sum_x)
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();
}
```

---

## 進捗サマリー

### Phase 1 (完了 ✅)
- PyO3基盤
- Tensor ↔ NumPy変換
- Python bindings
- maturinビルド

### Phase 2 (部分完了 ⏳)
- ✅ AST拡張
- ✅ Parser実装
- ✅ Lexer更新
- ✅ Visitorサポート
- ✅ テスト
- ⏳ Python実行エンジン統合（Phase 3へ）

### Phase 3 (未着手 ⏳)
- Python実行環境の埋め込み
- モジュール管理
- 関数呼び出し実装
- エラーハンドリング
- 変数共有
- 統合テスト

---

**Status**: Phase 2 AST/Parser層完了、Phase 3 実行層へ進む準備完了
**Next**: Python実行エンジンの統合実装
