# モデルローディングガイド

このドキュメントは、PyTorchやHuggingFaceのモデルをTensorLogicにロードして使用する方法を説明します。SafeTensors形式（PyTorch互換）およびGGUF形式（量子化LLM）に対応しています。

## 基本的な使用法

### 1. SafeTensors形式のモデルをロード（PyTorchから保存したモデル）

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. GGUF形式のモデルをロード（量子化されたLLM）

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. モデルからテンソルを取得

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## 実用例：線形レイヤーの推論

モデルのウェイトとバイアスを使って推論を実行：

```tensorlogic
function forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // 線形変換: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## PyTorchモデルの準備手順

Python側で以下のコードを実行してSafeTensors形式で保存：

```python
import torch
from safetensors.torch import save_file

# PyTorchモデルを作成
model = MyModel()

# モデルの重みを辞書形式で取得
tensors = {name: param for name, param in model.named_parameters()}

# SafeTensors形式で保存
save_file(tensors, "model.safetensors")
```

その後、TensorLogicで読み込み：

```tensorlogic
model = load_model("model.safetensors")
```

## サポートされるフォーマット

### 1. SafeTensors (.safetensors)

- PyTorch, HuggingFace互換
- F32, F64, F16, BF16データ型をサポート
- 全てのデータは自動的にf16に変換
- Metal GPUに直接ロード

### 2. GGUF (.gguf)

- llama.cpp形式の量子化モデル
- Q4_0, Q8_0, F32, F16サポート
- Metal GPUに直接ロード

### 3. CoreML (.mlmodel, .mlpackage)

- Apple Neural Engine最適化モデル
- iOS/macOS専用

## 実際の線形モデルの例

```tensorlogic
// 入力データ（バッチサイズ4、特徴量次元3）
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// ウェイト行列（3 x 2）
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// バイアス（2次元）
let b = tensor<float16>([0.01, 0.02], device: gpu)

// 推論実行
let output = forward(X, W, b)

// 結果を出力
print("Output shape:", output.shape)
print("Output:", output)
```

## モデルの保存

TensorLogicで作成したモデルもSafeTensors形式で保存可能：

```tensorlogic
save_model(model, "output.safetensors")
```

これにより、PyTorchやHuggingFaceとの相互運用が可能になります。

## 注意事項

- TensorLogicは全ての演算をf16で実行（Metal GPU最適化）
- 読み込み時に他のデータ型は自動的にf16に変換
- 整数型（i8, i32など）はサポート外（浮動小数点のみ）
- 大規模モデルは自動的にMetal GPUメモリにロード

## 関連ドキュメント

- [GGUF量子化モデル](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Getting Started Guide](../claudedocs/getting_started.md)
