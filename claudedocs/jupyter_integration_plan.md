# TensorLogic Jupyter Integration - 包括的実装計画

**作成日**: 2025年10月21日
**目標**: Jupyter Lab上でTensorLogicインタープリターを動かし、Pythonライブラリを呼び出せるようにする
**期間**: 5-7週間
**バージョン**: v0.2.0

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [技術スタック](#技術スタック)
4. [実装フェーズ](#実装フェーズ)
5. [ファイル構成](#ファイル構成)
6. [使用例](#使用例)
7. [課題とリスク](#課題とリスク)
8. [マイルストーン](#マイルストーン)

---

## 概要

### ゴール

TensorLogicをJupyter Labで実行可能にし、以下を実現：

1. ✅ **Jupyterカーネルとして動作**: TensorLogicコードをノートブックで実行
2. ✅ **Pythonライブラリ呼び出し**: NumPy, PyTorch, scikit-learnなどを使用
3. ✅ **双方向データ交換**: Tensor ↔ NumPy配列の自動変換
4. ✅ **Metal GPU最適化維持**: f16精度でのApple Silicon最適化を保持

### 価値提案

- **TensorLogicの型安全性** + **Pythonの豊富なエコシステム**
- **Metal GPU加速** + **NumPy/PyTorchの機能性**
- **f16ネイティブサポート** + **既存ライブラリとの互換性**

---

## アーキテクチャ

### システム構成図

```
┌─────────────────────────────────────────────────────────┐
│                    Jupyter Lab                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │           TensorLogic Notebook                    │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  %%tensorlogic                              │  │  │
│  │  │  python import numpy as np                  │  │  │
│  │  │                                             │  │  │
│  │  │  main {                                     │  │  │
│  │  │    tensor x: float16[3,3] = [...]          │  │  │
│  │  │    result := python.call("np.sum", x)      │  │  │
│  │  │  }                                          │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              TensorLogic Jupyter Kernel (Python)        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  kernel.py (IPyKernel based)                      │  │
│  │    - ZeroMQ message handling                      │  │
│  │    - Code execution orchestration                 │  │
│  │    - Output formatting                            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ (PyO3)
┌─────────────────────────────────────────────────────────┐
│         TensorLogic Core (Rust + PyO3 Bindings)         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Interpreter                                      │  │
│  │    - AST parsing                                  │  │
│  │    - Code execution                               │  │
│  │    - Python FFI calls                             │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  PyO3 Bindings                                    │  │
│  │    - Tensor ↔ NumPy conversion                   │  │
│  │    - Python function calls                        │  │
│  │    - Error handling                               │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Metal GPU Backend                                │  │
│  │    - f16 tensor operations                        │  │
│  │    - Metal acceleration                           │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ (Data conversion)
┌─────────────────────────────────────────────────────────┐
│                Python Ecosystem                         │
│    NumPy | PyTorch | scikit-learn | Pandas | ...       │
└─────────────────────────────────────────────────────────┘
```

### データフロー

```
TensorLogic Code
    │
    ├─> Parser → AST
    │
    ├─> Python Import Statement
    │       └─> Load Python module (via PyO3)
    │
    └─> Python Function Call
            │
            ├─> Tensor → NumPy (f16 array)
            │       └─> Call Python function
            │               └─> Get result (NumPy array)
            │                       └─> NumPy → Tensor (f16)
            │
            └─> Return to TensorLogic
```

---

## 技術スタック

### Rust側

| 依存関係 | バージョン | 用途 |
|---------|----------|------|
| `pyo3` | 0.21+ | PythonバインディングとFFI |
| `numpy` | 0.21+ | NumPy配列との連携 |
| `half` | 2.4+ | f16サポート |
| `metal` | 0.29+ | GPU加速 |
| `pest` | 2.7+ | パーサー |

### Python側

| パッケージ | バージョン | 用途 |
|-----------|----------|------|
| `ipykernel` | 6.0+ | Jupyterカーネル基盤 |
| `jupyter_client` | 8.0+ | カーネルプロトコル |
| `numpy` | 1.20+ | 数値計算（float16サポート） |
| `maturin` | 1.0+ | Rustパッケージビルド |

### 必須環境

- **OS**: macOS 12+ (Apple Silicon)
- **Python**: 3.8+
- **Rust**: 1.70+
- **Jupyter Lab**: 3.0+

---

## 実装フェーズ

### Phase 1: PyO3基盤 (Week 1-2)

**目標**: RustとPythonの基本的な連携を確立

#### タスク

1. **依存関係の追加**
   ```toml
   [dependencies]
   pyo3 = { version = "0.21", features = ["extension-module", "abi3-py38"] }
   numpy = "0.21"
   ```

2. **Pythonモジュール作成**
   - `src/python/mod.rs`: PyO3エントリーポイント
   - `pyproject.toml`: Pythonパッケージ設定
   - `maturin`ビルド設定

3. **Tensor ↔ NumPy変換**
   ```rust
   // src/python/tensor.rs

   use pyo3::prelude::*;
   use numpy::{PyArray, PyArrayMethods};
   use half::f16;

   impl ToPyObject for Tensor {
       fn to_object(&self, py: Python) -> PyObject {
           // 1. CPU上のデータを取得
           let data = self.to_cpu().unwrap().to_vec();
           let shape: Vec<usize> = self.dims().to_vec();

           // 2. NumPy float16配列を作成
           let array = PyArray::from_vec(py, data);
           array.reshape(shape).unwrap().to_object(py)
       }
   }

   impl FromPyObject<'_> for Tensor {
       fn extract(ob: &PyAny) -> PyResult<Self> {
           // NumPy配列からTensorを作成
           let array: &PyArray<f16, _> = ob.extract()?;
           let data = array.to_vec()?;
           let shape = array.shape().to_vec();

           Tensor::from_vec(data, shape)
               .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
       }
   }
   ```

4. **基本的なテスト**
   ```python
   import tensorlogic as tl
   import numpy as np

   # Tensor作成
   tensor = tl.Tensor.from_vec([1.0, 2.0, 3.0], [3])

   # NumPyに変換
   array = np.array(tensor)
   assert array.dtype == np.float16
   ```

**成果物**:
- ✅ `tensorlogic` Pythonパッケージ
- ✅ Tensor ↔ NumPy変換機能
- ✅ 基本的なテストスイート

---

### Phase 2: Python FFI統合 (Week 2-3)

**目標**: TensorLogic言語からPython関数を呼び出せるようにする

#### タスク

1. **AST拡張**
   ```rust
   // src/ast.rs

   #[derive(Debug, Clone, PartialEq)]
   pub enum Statement {
       // ... 既存のステートメント

       /// Python import statement
       /// Example: python import numpy as np
       PythonImport {
           module: String,
           alias: Option<String>,
       },
   }

   #[derive(Debug, Clone, PartialEq)]
   pub enum Expression {
       // ... 既存の式

       /// Python function call
       /// Example: python.call("np.sum", x)
       PythonCall {
           function: String,
           args: Vec<Expression>,
       },
   }
   ```

2. **パーサー拡張**
   ```pest
   // src/parser/grammar.pest

   statement = {
       // ... 既存のステートメント
       | python_import
   }

   python_import = {
       "python" ~ "import" ~ module_path ~ ("as" ~ identifier)?
   }

   module_path = @{
       identifier ~ ("." ~ identifier)*
   }

   primary = {
       // ... 既存の式
       | python_call
   }

   python_call = {
       "python" ~ "." ~ "call" ~ "(" ~ string_literal ~ ("," ~ expression)* ~ ")"
   }
   ```

3. **インタープリター実装**
   ```rust
   // src/interpreter/python_call.rs

   use pyo3::prelude::*;
   use pyo3::types::{PyModule, PyDict};

   pub struct PythonContext {
       py: Python<'static>,
       globals: Py<PyDict>,
   }

   impl PythonContext {
       pub fn new() -> PyResult<Self> {
           Python::with_gil(|py| {
               let globals = PyDict::new(py);
               Ok(Self {
                   py,
                   globals: globals.into(),
               })
           })
       }

       pub fn import_module(&mut self, module: &str, alias: Option<&str>) -> RuntimeResult<()> {
           Python::with_gil(|py| {
               let module_obj = PyModule::import(py, module)?;
               let name = alias.unwrap_or(module);
               self.globals.as_ref(py).set_item(name, module_obj)?;
               Ok(())
           })
       }

       pub fn call_function(&self, func_path: &str, args: Vec<Value>) -> RuntimeResult<Value> {
           Python::with_gil(|py| {
               // 1. 関数を取得
               let func = self.get_function(py, func_path)?;

               // 2. 引数を変換（Tensor → NumPy）
               let py_args = args.iter()
                   .map(|v| v.to_python(py))
                   .collect::<PyResult<Vec<_>>>()?;

               // 3. 関数を呼び出す
               let result = func.call1(PyTuple::new(py, py_args))?;

               // 4. 結果を変換（NumPy → Tensor）
               Value::from_python(result)
           })
       }
   }
   ```

4. **インタープリター統合**
   ```rust
   // src/interpreter/mod.rs

   pub struct Interpreter {
       env: Environment,
       python_ctx: Option<PythonContext>,
       // ...
   }

   impl Interpreter {
       pub fn new() -> Self {
           Self {
               env: Environment::new(),
               python_ctx: PythonContext::new().ok(),
               // ...
           }
       }

       fn execute_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
           match stmt {
               Statement::PythonImport { module, alias } => {
                   if let Some(ctx) = &mut self.python_ctx {
                       ctx.import_module(module, alias.as_deref())?;
                   }
                   Ok(())
               }
               // ... 他のステートメント
           }
       }

       fn eval_expr(&mut self, expr: &Expression) -> RuntimeResult<Value> {
           match expr {
               Expression::PythonCall { function, args } => {
                   if let Some(ctx) = &self.python_ctx {
                       let arg_values: Vec<Value> = args.iter()
                           .map(|e| self.eval_expr(e))
                           .collect::<RuntimeResult<_>>()?;

                       ctx.call_function(function, arg_values)
                   } else {
                       Err(RuntimeError::InvalidOperation(
                           "Python context not initialized".to_string()
                       ))
                   }
               }
               // ... 他の式
           }
       }
   }
   ```

**成果物**:
- ✅ Python import構文サポート
- ✅ Python関数呼び出し機能
- ✅ エラーハンドリング
- ✅ テストケース

---

### Phase 3: Jupyterカーネル (Week 3-4)

**目標**: TensorLogicをJupyter Labで実行可能にする

#### タスク

1. **Pythonカーネルラッパー**
   ```python
   # python/tensorlogic/kernel.py

   from ipykernel.kernelbase import Kernel
   import tensorlogic

   class TensorLogicKernel(Kernel):
       implementation = 'TensorLogic'
       implementation_version = '0.2.0'
       language = 'tensorlogic'
       language_version = '0.2.0'
       language_info = {
           'name': 'tensorlogic',
           'mimetype': 'text/x-tensorlogic',
           'file_extension': '.tl',
       }
       banner = "TensorLogic - Tensor algebra meets logic programming"

       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.interpreter = tensorlogic.Interpreter()
           self._execution_count = 0

       def do_execute(self, code, silent, store_history=True,
                      user_expressions=None, allow_stdin=False):
           """コードを実行"""
           if not silent:
               self._execution_count += 1

           try:
               # TensorLogicコードを実行
               result = self.interpreter.execute(code)

               # 結果を出力
               if result is not None and not silent:
                   stream_content = {
                       'name': 'stdout',
                       'text': str(result)
                   }
                   self.send_response(self.iopub_socket, 'stream', stream_content)

               return {
                   'status': 'ok',
                   'execution_count': self._execution_count,
                   'payload': [],
                   'user_expressions': {},
               }

           except Exception as e:
               # エラーハンドリング
               if not silent:
                   error_content = {
                       'ename': type(e).__name__,
                       'evalue': str(e),
                       'traceback': self._format_traceback(e)
                   }
                   self.send_response(self.iopub_socket, 'error', error_content)

               return {
                   'status': 'error',
                   'execution_count': self._execution_count,
                   'ename': type(e).__name__,
                   'evalue': str(e),
                   'traceback': self._format_traceback(e)
               }

       def do_complete(self, code, cursor_pos):
           """コード補完"""
           # TODO: 実装
           return {
               'matches': [],
               'cursor_start': cursor_pos,
               'cursor_end': cursor_pos,
               'metadata': {},
               'status': 'ok'
           }

       def _format_traceback(self, exception):
           """トレースバックのフォーマット"""
           import traceback
           return traceback.format_exception(
               type(exception),
               exception,
               exception.__traceback__
           )
   ```

2. **kernel.json作成**
   ```json
   {
     "argv": [
       "python",
       "-m",
       "tensorlogic.kernel",
       "-f",
       "{connection_file}"
     ],
     "display_name": "TensorLogic",
     "language": "tensorlogic",
     "interrupt_mode": "signal",
     "env": {}
   }
   ```

3. **インストールスクリプト**
   ```python
   # python/tensorlogic/install.py

   import json
   import os
   import sys
   from jupyter_client.kernelspec import KernelSpecManager

   def install_kernel():
       """TensorLogicカーネルをインストール"""
       kernel_json = {
           "argv": [
               sys.executable,
               "-m",
               "tensorlogic.kernel",
               "-f",
               "{connection_file}"
           ],
           "display_name": "TensorLogic",
           "language": "tensorlogic",
           "interrupt_mode": "signal"
       }

       # カーネルディレクトリを作成
       kernel_dir = os.path.join(
           os.path.dirname(__file__),
           'kernelspec'
       )
       os.makedirs(kernel_dir, exist_ok=True)

       # kernel.jsonを書き込む
       with open(os.path.join(kernel_dir, 'kernel.json'), 'w') as f:
           json.dump(kernel_json, f, indent=2)

       # カーネルをインストール
       ksm = KernelSpecManager()
       ksm.install_kernel_spec(
           kernel_dir,
           'tensorlogic',
           user=True,
           replace=True
       )

       print("TensorLogic kernel installed successfully!")
       print("Launch Jupyter Lab and select 'TensorLogic' kernel.")

   if __name__ == '__main__':
       install_kernel()
   ```

4. **setup.py整備**
   ```python
   # python/setup.py

   from setuptools import setup
   from setuptools_rust import Binding, RustExtension

   setup(
       name="tensorlogic",
       version="0.2.0",
       rust_extensions=[
           RustExtension(
               "tensorlogic.tensorlogic",
               binding=Binding.PyO3,
               path="../Cargo.toml"
           )
       ],
       packages=["tensorlogic"],
       install_requires=[
           "ipykernel>=6.0.0",
           "jupyter-client>=8.0.0",
           "numpy>=1.20.0",
       ],
       entry_points={
           'console_scripts': [
               'tensorlogic-install-kernel=tensorlogic.install:install_kernel',
           ],
       },
       zip_safe=False,
   )
   ```

**成果物**:
- ✅ Jupyterカーネル実装
- ✅ インストールスクリプト
- ✅ kernel.json定義
- ✅ 基本的なREPL機能

---

### Phase 4: 統合テスト (Week 5)

**目標**: 実際のユースケースで動作確認

#### タスク

1. **NumPy統合テスト**
   ```python
   # tests/test_numpy_integration.py

   import tensorlogic as tl
   import numpy as np

   def test_tensor_to_numpy():
       tensor = tl.Tensor.from_vec([1.0, 2.0, 3.0], [3])
       array = np.array(tensor)
       assert array.dtype == np.float16
       assert np.allclose(array, [1.0, 2.0, 3.0])

   def test_numpy_to_tensor():
       array = np.array([1.0, 2.0, 3.0], dtype=np.float16)
       tensor = tl.Tensor.from_numpy(array)
       assert tensor.shape == [3]

   def test_numpy_function_call():
       code = """
       python import numpy as np

       main {
           tensor x: float16[3, 3] = [[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]]

           result := python.call("np.sum", x)
           print("Sum:", result)
       }
       """

       interp = tl.Interpreter()
       result = interp.execute(code)
       assert result is not None
   ```

2. **PyTorch統合テスト**
   ```python
   # tests/test_pytorch_integration.py

   def test_pytorch_matmul():
       code = """
       python import torch
       python import numpy as np

       main {
           tensor a: float16[2, 3] = [[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]]
           tensor b: float16[3, 2] = [[1.0, 2.0],
                                      [3.0, 4.0],
                                      [5.0, 6.0]]

           # PyTorchでmatmul
           a_torch := python.call("torch.from_numpy", a)
           b_torch := python.call("torch.from_numpy", b)
           result := python.call("torch.matmul", a_torch, b_torch)

           print("Result:", result)
       }
       """

       interp = tl.Interpreter()
       result = interp.execute(code)
   ```

3. **サンプルノートブック作成**
   - `examples/jupyter_numpy_demo.ipynb`
   - `examples/jupyter_pytorch_demo.ipynb`
   - `examples/jupyter_sklearn_demo.ipynb`

**成果物**:
- ✅ 統合テストスイート
- ✅ サンプルノートブック
- ✅ ドキュメント

---

### Phase 5: 最適化と公開 (Week 6-7)

**目標**: パフォーマンス最適化とリリース準備

#### タスク

1. **パフォーマンス最適化**
   - GIL管理の最適化
   - 変換キャッシング
   - ゼロコピー最適化（可能な場合）

2. **ドキュメント整備**
   - `README.md`更新
   - API リファレンス
   - チュートリアル
   - トラブルシューティング

3. **PyPIパッケージング**
   - `pyproject.toml`整備
   - `maturin publish`設定
   - CI/CD（GitHub Actions）

4. **リリース**
   - v0.2.0タグ作成
   - PyPIへの公開
   - リリースノート

**成果物**:
- ✅ 最適化されたコード
- ✅ 完全なドキュメント
- ✅ PyPIパッケージ
- ✅ v0.2.0リリース

---

## ファイル構成

```
tensorlogic/
├── Cargo.toml                    # Rust依存関係
├── pyproject.toml                # Pythonパッケージ設定
├── README.md
│
├── src/
│   ├── lib.rs                    # Rustライブラリエントリーポイント
│   ├── main.rs                   # CLIツール
│   │
│   ├── python/                   # PyO3バインディング
│   │   ├── mod.rs                # Pythonモジュールエントリーポイント
│   │   ├── tensor.rs             # Tensor ↔ NumPy変換
│   │   ├── interpreter.rs        # Interpreterバインディング
│   │   └── ffi.rs                # Python関数呼び出し
│   │
│   ├── interpreter/
│   │   ├── mod.rs
│   │   └── python_call.rs        # Python呼び出しロジック
│   │
│   ├── ast.rs                    # Python関連AST追加
│   ├── parser/
│   │   ├── mod.rs
│   │   └── grammar.pest          # Python構文追加
│   └── ...
│
├── python/                       # Pythonパッケージ
│   ├── tensorlogic/
│   │   ├── __init__.py
│   │   ├── kernel.py             # Jupyterカーネル
│   │   ├── install.py            # インストールスクリプト
│   │   ├── magic.py              # Jupyter magic commands（将来）
│   │   └── kernelspec/
│   │       └── kernel.json       # カーネル定義
│   │
│   └── setup.py                  # Pythonパッケージビルド
│
├── examples/
│   ├── jupyter_numpy_demo.ipynb
│   ├── jupyter_pytorch_demo.ipynb
│   └── jupyter_sklearn_demo.ipynb
│
├── tests/
│   ├── test_numpy_integration.py
│   ├── test_pytorch_integration.py
│   └── test_kernel.py
│
└── claudedocs/
    ├── jupyter_integration_plan.md  # このドキュメント
    ├── pyo3_bindings_guide.md       # PyO3実装ガイド
    └── kernel_development_guide.md  # カーネル開発ガイド
```

---

## 使用例

### 基本的な使用例

```python
# Jupyter Notebookのセル

%%tensorlogic
python import numpy as np

main {
    # TensorLogicでテンソルを作成
    tensor x: float16[3, 3] = [[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]]

    # NumPyで固有値計算
    eigenvalues := python.call("np.linalg.eigvals", x)
    print("Eigenvalues:", eigenvalues)
}
```

### PyTorch統合例

```python
%%tensorlogic
python import torch
python import numpy as np

main {
    tensor weights: float16[10, 5] learnable
    tensor input: float16[5] = [1.0, 2.0, 3.0, 4.0, 5.0]

    # TensorLogicで計算
    output := weights @ input

    # PyTorchで後処理
    output_torch := python.call("torch.from_numpy", output)
    normalized := python.call("torch.nn.functional.softmax", output_torch)

    print("Normalized output:", normalized)
}
```

### scikit-learn統合例

```python
%%tensorlogic
python import numpy as np
python from sklearn.decomposition import PCA

main {
    # データセット
    tensor data: float16[100, 10] = auto

    # PCAで次元削減
    pca_result := python.call("PCA(n_components=2).fit_transform", data)

    print("PCA result shape:", python.call("pca_result.shape"))
}
```

---

## 課題とリスク

### 技術的課題

| 課題 | 影響度 | 対策 |
|-----|-------|------|
| **GIL管理の複雑さ** | 高 | PyO3のGIL APIを適切に使用、明示的な`Python::with_gil()` |
| **Metal ↔ CPU ↔ NumPy変換オーバーヘッド** | 中 | キャッシング戦略、必要な場合のみ変換 |
| **f16互換性** | 中 | 自動的にfloat32/64に変換するオプション提供 |
| **エラーハンドリング** | 中 | 統一的なエラー型、詳細なエラーメッセージ |

### リスク管理

1. **Apple Silicon専用の制約**
   - リスク: ユーザーベースが限定される
   - 対策: READMEで明確に記載、将来的にCPU対応を検討

2. **f16サポートの限界**
   - リスク: 一部のPythonライブラリがfloat16未対応
   - 対策: 自動変換機能、警告メッセージ

3. **パフォーマンスオーバーヘッド**
   - リスク: Python FFIの呼び出しコスト
   - 対策: ベンチマーク測定、キャッシング、最適化

---

## マイルストーン

### M1: PyO3基盤完成 (Week 2終了時)
- ✅ PyO3ビルド設定
- ✅ Tensor ↔ NumPy変換
- ✅ 基本的なテスト

### M2: Python FFI完成 (Week 3終了時)
- ✅ Python import構文
- ✅ Python関数呼び出し
- ✅ エラーハンドリング

### M3: Jupyterカーネル完成 (Week 4終了時)
- ✅ カーネル実装
- ✅ インストール機能
- ✅ 基本的な実行環境

### M4: 統合テスト完了 (Week 5終了時)
- ✅ NumPy/PyTorch/scikit-learn統合
- ✅ サンプルノートブック
- ✅ ドキュメント

### M5: v0.2.0リリース (Week 7終了時)
- ✅ パフォーマンス最適化
- ✅ 完全なドキュメント
- ✅ PyPI公開

---

## 次のステップ

1. **Phase 1開始**: PyO3依存関係の追加とビルド設定
2. **プロトタイプ作成**: 最小限のTensor ↔ NumPy変換
3. **動作確認**: 基本的なPython関数呼び出し

このプランに基づいて実装を開始しますか？

---

**関連ドキュメント**:
- [PyO3 Documentation](https://pyo3.rs/)
- [Jupyter Kernel Documentation](https://jupyter-client.readthedocs.io/)
- [maturin Guide](https://www.maturin.rs/)
