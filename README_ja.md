# TensorLogic

Apple Silicon向けの本格的なf16テンソルライブラリ。自動微分と最適化アルゴリズムを搭載し、Metal GPUとNeural Engineアクセラレーションに対応。

> **⚠️ 実験的プロジェクトについて**
>
> これは以下の制約がある**実験的研究プロジェクト**です:
> - **Apple Siliconのみ**: M-seriesチップ（M1/M2/M3/M4）搭載のmacOSが必要
> - **f16（半精度）のみ**: すべての浮動小数点演算は16ビット浮動小数点のみを使用
> - **Metalフレームワーク必須**: GPUアクセラレーションはAppleのMetalフレームワークに依存
>
> Apple製品以外のハードウェアや汎用コンピューティングでの本番利用を想定していません。

## 概要

TensorLogicは、Apple Silicon（M-seriesチップ）専用に設計された統合テンソル代数ライブラリで、CoreMLを介してMetal GPUとNeural Engineのシームレスな統合を提供します。すべての演算は最適なNeural Engine互換性とパフォーマンスのためにf16（半精度）を維持します。

**📚 [入門ガイド](claudedocs/getting_started.md)** | **📖 [オプティマイザチュートリアル](claudedocs/optimizer_tutorial.md)** | **🔧 [完全仕様](claudedocs/f16_neural_engine_metal_spec.md)**

**📚 ドキュメント**: [English](README.md) | [日本語](README_ja.md)

## 主な機能

### ✅ 本番環境対応

- **テンソル演算** (Phase 1-3)
  - 要素ごとの演算: add, sub, mul, div
  - GPU最適化された行列乗算
  - 活性化関数: ReLU, GELU, Softmax
  - ブロードキャスト（NumPy互換）
  - リダクション: sum, mean, max, min（全体および次元指定）
  - Einsteinの規約による最適化
  - メモリ効率のためのインプレース演算

- **自動微分** (Phase 5-6) ⚡
  - 動的計算グラフ（PyTorchスタイル）
  - 完全な逆伝播実装
  - すべての演算の勾配計算
  - 二階微分（ヘッセ行列の基礎）
  - 数値検証による勾配チェック
  - 高階微分のためのグラフ作成モード

- **最適化アルゴリズム** (Phase 9.1) 🚀 NEW
  - **SGD**: 基本的な勾配降下法、モメンタム、Nesterov
  - **Adam**: AMSGradサポート付き適応学習率
  - **AdamW**: より良い正則化のための分離重み減衰
  - 学習率スケジューリング
  - チェックポイントのための状態保存/読み込み
  - 複数パラメータグループサポート

- **デバイスアクセラレーション** (Phase 2, 4, 7)
  - すべての演算でMetal GPUアクセラレーション
  - Neural Engine統合（基盤完成）
  - ゼロコピーMetal ↔ Neural Engine変換
  - メモリ最適化のためのバッファプーリング
  - 融合演算（add+relu, mul+relu, affine）
  - Metal GPU勾配カーネル
  - 自動デバイス選択のためのExecutionPlanner

- **Python統合** (Phase 1-3) 🐍 NEW
  - Pythonモジュールインポート: `python import numpy as np`
  - Python関数呼び出し: `python.call("np.sum", x)`
  - シームレスなTensor ↔ NumPy変換（f16 ↔ f32）
  - NumPy、PyTorch、SciKit-Learn統合
  - PyO3によるPythonバインディング
  - Jupyter対応アーキテクチャ

### 🚧 高度な機能（オプション）

- **Neural Engine推論**: 完全なCoreMLモデル統合（基盤完成、延期）
- **学習率スケジューラ**: コサイン、ステップ減衰、ウォームアップ（近日公開）
- **追加の最適化アルゴリズム**: RMSprop、Adagrad（将来の作業）

## クイックスタート

包括的なチュートリアルについては、[**入門ガイド**](claudedocs/getting_started.md)をご覧ください。

### インストール

**オプション1: CLIバイナリ**

TensorLogicインタープリタをバイナリとしてインストール:

```bash
cargo install --git https://github.com/JunSuzukiJapan/tensorlogic.git
```

これにより`tl`コマンドがインストールされます。

またはソースからビルド:

```bash
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
cargo build --release
# バイナリは: target/release/tl
```

**オプション2: Pythonモジュール** 🐍 NEW

TensorLogicをPythonパッケージとしてインストール:

```bash
# maturinをインストール
pip install maturin

# wheelをビルドしてインストール
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
maturin build --features python-extension --release
pip install target/wheels/tensorlogic-*.whl
```

Pythonで使用:

```python
import tensorlogic as tl
import numpy as np

# インタープリタを作成
interp = tl.Interpreter()

# TensorLogicコードを実行
interp.execute("""
    main {
        python import numpy as np

        tensor x: float16[3] = [1.0, 2.0, 3.0]
        tensor sum_x: float16[1] = python.call("np.sum", x)

        print("Sum:", sum_x)
    }
""")
# 出力: ✓ Python import: numpy (as np)
#       Sum: [6.0000]
```

### 基本的な使い方

TensorLogicは、テンソル演算とニューラルネットワークトレーニングのためのインタープリタ型言語です。

#### 1. テンソル宣言

`float16`型でテンソルを宣言（TensorLogicはGPU効率のために16ビット浮動小数点を使用）:

```tensorlogic
// スカラーテンソル
tensor x: float16[1] = [2.0]

// ベクトルテンソル
tensor v: float16[3] = [1.0, 2.0, 3.0]

// 行列テンソル
tensor W: float16[2, 3] = [[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]

// 学習可能パラメータ（トレーニング用）
tensor w: float16[1] learnable = [0.5]
```

#### 2. テンソル演算

要素ごとの演算と行列演算:

```tensorlogic
main {
    // 要素ごとの演算
    result := a + b      // 加算
    result := a - b      // 減算
    result := a * b      // 乗算
    result := a / b      // 除算

    // 行列演算
    result := A @ B      // 行列乗算
    result := a ** 2     // べき乗

    // 活性化関数
    result := relu(x)
    result := gelu(x)

    // リダクション
    sum := sum(tensor)
    mean := mean(tensor)
}
```

#### 3. 外部ファイルのインポート

他のTensorLogicファイルから宣言をインポート:

```tensorlogic
// 別のファイルからテンソルと関数定義をインポート
import "path/to/module.tl"

main {
    // インポートしたテンソルと関数を使用
    result := imported_tensor * 2
}
```

**機能**:
- 相対パス解決（インポートするファイルからの相対パス）
- 重複インポート防止（同じファイルは2回インポートされない）
- メインブロック実行なし（宣言のみがインポートされる）
- 循環依存検出

**例**: [examples/import_test/](examples/import_test/)を参照

#### 4. 制御フロー

条件分岐とループ:

```tensorlogic
main {
    // if文
    if x > 0 {
        y := x * 2
    }

    // if-else文
    if x > 0 {
        y := x * 2
    } else {
        y := x * 0.5
    }

    // forループ
    for i in 0..10 {
        x := x + i
    }

    // whileループ
    while x < 100 {
        x := x * 1.1
    }
}
```

#### 5. 勾配降下法によるトレーニング

`learn`文を使用してモデルをトレーニング（ローカル変数サポート）:

```tensorlogic
// 学習可能パラメータを宣言
tensor W: float16[1] learnable = [0.5]

main {
    // トレーニングデータ
    tensor x1: float16[1] = [1.0]
    tensor y1: float16[1] = [3.0]
    tensor x2: float16[1] = [-2.0]  // 負の数がサポートされています
    tensor y2: float16[1] = [-6.0]

    // learnブロックでローカル変数を使ってモデルをトレーニング
    learn {
        // 中間計算のためのローカル変数
        pred1 := x1 * W
        pred2 := x2 * W

        // 損失を計算
        err1 := pred1 - y1
        err2 := pred2 - y2
        loss := err1 * err1 + err2 * err2

        objective: loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }

    print("学習したW:", W)  // 3.0に近い値になるはず
}
```

**機能**:
- **ローカル変数**: `learn`ブロック内で`:=`を使用して中間計算を行う
- **負の数**: 負の数値リテラルを完全サポート
- **複数の最適化アルゴリズム**: SGD、Adam、AdamWとカスタマイズ可能なハイパーパラメータ

利用可能な最適化アルゴリズム:
- `sgd(lr: 0.1)` - 確率的勾配降下法
- `adam(lr: 0.001)` - Adamオプティマイザ
- `adamw(lr: 0.001, weight_decay: 0.01)` - 重み減衰付きAdamW

#### 6. プログラムの実行

```bash
# TensorLogicスクリプトを実行
tl run your_script.tl

# デバッグモード
tl run your_script.tl --debug

# REPL（対話モード）を起動
tl repl

# バージョンを表示
tl --version

# ヘルプを表示
tl --help
```

### 完全な例

完全な線形回帰の例（[examples/tutorial_01_linear_regression.tl](examples/tutorial_01_linear_regression.tl)）:

```tensorlogic
// 学習可能パラメータを宣言
tensor w: float16[1] learnable = [0.5]
tensor b: float16[1] learnable = [0.5]

main {
    // トレーニング: 損失関数を最小化
    // Loss = w^2 + b^2 （w=0, b=0に収束）

    learn {
        objective: w * w + b * b,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

実行:
```bash
tl run examples/tutorial_01_linear_regression.tl
```

### その他の例

- [シンプルな線形モデル](examples/simple_linear_model.tl) - ローカル変数を使ったトレーニングと推論の例
- [チュートリアル01: 線形回帰](examples/tutorial_01_linear_regression.tl) - 基本的な最適化
- [チュートリアル02: 複数パラメータ最適化](examples/tutorial_02_logistic_regression.tl) - 複数のパラメータ
- [チュートリアル03: ニューラルネットワークの重み](examples/tutorial_03_neural_network.tl) - 重み正則化
- [チュートリアル04: 論理プログラミング](examples/tutorial_04_logic_programming.tl) - ニューラル・シンボリック統合
- [インポートテスト](examples/import_test/) - 循環依存検出付き外部ファイルインポート
- [入門ガイド](claudedocs/getting_started.md) - 包括的なチュートリアル
- [言語リファレンス](docs/ja/language_reference.md) - 完全な構文リファレンス

## Python統合 🐍

TensorLogicは、NumPy、PyTorch、SciKit-Learnなどのライブラリとシームレスに統合できます。

### Pythonモジュールのインポート

```tensorlogic
main {
    // Pythonモジュールをオプションのエイリアス付きでインポート
    python import numpy as np
    python import torch
    python import sklearn.preprocessing as preprocessing
}
```

### Python関数の呼び出し

```tensorlogic
main {
    python import numpy as np

    // TensorLogicテンソルを作成
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[3] = [4.0, 5.0, 6.0]

    // NumPy関数を呼び出し
    tensor sum_result: float16[3] = python.call("np.add", x, y)
    tensor mean_x: float16[1] = python.call("np.mean", x)
    tensor max_y: float16[1] = python.call("np.max", y)

    print("加算:", sum_result)   // [5.0, 7.0, 9.0]
    print("平均:", mean_x)        // [2.0]
    print("最大:", max_y)          // [6.0]
}
```

### Tensor ↔ NumPy変換

TensorLogicはf16テンソルとNumPy配列を自動変換します:

- **TensorLogic → NumPy**: f16 → f32（わずかな精度損失）
- **NumPy → TensorLogic**: f32/f64 → f16
- GPUテンソルは変換のため自動的にCPUに移動

### Pythonから使用

```python
import tensorlogic as tl
import numpy as np

# インタープリタを作成
interp = tl.Interpreter()

# Python統合を使ってTensorLogicを実行
code = """
main {
    python import numpy as np

    tensor data: float16[5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor normalized: float16[5] = python.call("preprocessing.normalize", data)

    print("正規化済み:", normalized)
}
"""

interp.execute(code)
```

詳細な例については、[examples/python_integration_test.tl](examples/python_integration_test.tl)を参照してください。

## Jupyter Notebookサポート 📊

TensorLogicには、Jupyterノートブックでの対話的開発のためのJupyterカーネルが含まれています。

### インストール

```bash
# TensorLogic Jupyterカーネルをインストール
jupyter kernelspec install --user jupyter/tensorlogic

# インストールを確認
jupyter kernelspec list
```

### 使い方

1. **Jupyterを起動**:
```bash
jupyter notebook
# または
jupyter lab
```

2. **新しいノートブックを作成**し、カーネルとして「TensorLogic」を選択

3. **セルにTensorLogicコードを記述**:

```tensorlogic
// セル1: テンソルを宣言
tensor W: float16[1] learnable = [0.5]
```

```tensorlogic
// セル2: モデルをトレーニング
tensor x: float16[1] = [2.0]
tensor y: float16[1] = [6.0]

learn {
    pred := x * W
    loss := (pred - y) * (pred - y)

    objective: loss,
    optimizer: sgd(lr: 0.1),
    epochs: 50
}

print("トレーニング済みW:", W)  // ~3.0になるはず
```

4. **`Shift+Enter`でセルを実行**

### 機能

- **対話的実行**: TensorLogicコードをセルごとに実行
- **変数の永続性**: セッション内でセル間で変数が永続化
- **リアルタイム出力**: トレーニングの進行状況と結果を即座に確認
- **混合ワークフロー**: データ前処理/可視化のためにPythonセルと組み合わせ

### サンプルノートブック

完全なチュートリアルについては、[examples/jupyter_tutorial.ipynb](examples/jupyter_tutorial.ipynb)を参照してください。

## アーキテクチャ

### デバイス階層

```
Device
├── Metal GPU (MTLDevice)
│   ├── MetalBuffer (f16共有メモリ)
│   ├── KernelExecutor (コンピュートパイプラインキャッシュ)
│   └── Compute Shaders (.metalファイル)
├── Neural Engine (CoreML)
│   ├── NeuralEngineBuffer (MLMultiArrayラッパー)
│   ├── NeuralEngineOps (matmul, relu)
│   └── モデル統合（将来）
└── CPU (フォールバック)
    └── f16演算
```

### 演算フロー

1. **作成**: テンソルはMetal GPUまたはCPU上で作成
2. **計算**: 演算はコンピュートシェーダーを介してMetal GPU上で実行
3. **変換**: シームレスなMetal ↔ Neural Engineバッファ変換
4. **フォールバック**: サポートされていない演算の自動CPUフォールバック

## テストとパフォーマンス

### テスト結果

**286/286 テスト合格** ✅ （`--test-threads=1`使用時）

- パーサー: 18テスト（すべての6タイプ: float16, int16, int32, int64, bool, complex16）
- 型チェッカー: 20テスト
- インタープリタ: 45テスト
- テンソル演算: 95テスト
- 自動微分: 32テスト
- 最適化アルゴリズム: 27テスト（SGD、Adam、AdamW + スケジューラー）
- CoreML統合: 16テスト
- パフォーマンステスト: 10テスト
- GPU演算: 23テスト

**重要**: Metal GPUテストは競合状態を避けるためにシングルスレッド実行が必要です:

```bash
# すべてのテストを実行（GPUテストに必要）
cargo test --lib -- --test-threads=1

# 特定のテストを実行
cargo test --lib test_name -- --test-threads=1
```

**注意**: `--test-threads=1`なしでは、一部のGPUテストが同時Metalリソースアクセスにより失敗する可能性があります。これは予期される動作であり、ライブラリ自体のバグではありません。

### ハードウェアサポート

- **Apple M4 Pro** （主要開発環境）
- **Apple M1/M2/M3** （互換性あり）
- macOS 13+（Neural Engine機能用）
- すべての演算は最適なパフォーマンスのためにf16を使用

## ライセンス

デュアルライセンス:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) または http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) または http://opensource.org/licenses/MIT)

お好みのライセンスを選択できます。

## 謝辞

このプロジェクトは以下にインスパイアされました:
- **[Andrej Karpathyの「Intro to Large Language Models」](https://www.youtube.com/watch?v=rkBLPYqPkP4)** - 効率的なニューラル計算と最適化に関する洞察
- **[Tensor-Logic論文（arXiv:2510.12269）](https://arxiv.org/abs/2510.12269)** - テンソル代数と論理プログラミングを統合するための理論的基礎

[Claude Code](https://claude.com/claude-code)を使用して構築:
- GPU加速のためのAppleのMetalフレームワーク
- Neural Engine統合のためのAppleのCoreML
- Rustの優れた型システムと安全性保証

## 連絡先

プロジェクトメンテナ: Jun Suzuki
- GitHub: [@JunSuzukiJapan](https://github.com/JunSuzukiJapan)

---

**注意**: このライブラリはApple Silicon向けに最適化されており、macOSが必要です。Neural Engine機能にはmacOS 13+またはiOS 16+が必要です。
