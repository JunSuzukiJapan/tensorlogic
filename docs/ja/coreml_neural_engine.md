# CoreML & Neural Engine 統合ガイド

このガイドは、CoreMLモデルをTensorLogicで使用し、Apple Neural Engineで高速推論を実行する方法を説明します。

## CoreML と Neural Engine について

### CoreML

- Apple独自の機械学習フレームワーク
- iOS/macOS専用に最適化
- Neural Engine、GPU、CPUを自動的に活用
- .mlmodel / .mlmodelc形式

### Neural Engine

- Apple Silicon専用のAI専用チップ
- 最大15.8 TOPS (M1 Pro/Max)
- 超低消費電力（GPU比で1/10以下）
- f16演算に最適化

### TensorLogicとの統合

- 全てf16演算（Neural Engine最適化）
- Metal GPUとシームレスに連携
- モデルの自動フォーマット検出

## CoreMLモデルの作成

CoreMLモデルは通常、PythonのcoreMLtoolsで作成します：

```python
import coremltools as ct
import torch

# PyTorchモデルを作成
model = MyModel()
model.eval()

# トレースモデルを作成
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# CoreMLに変換
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Neural Engine最適化
    compute_precision=ct.precision.FLOAT16  # f16精度
)

# 保存
mlmodel.save("model.mlpackage")
```

## TensorLogicでの使用

### 1. CoreMLモデルをロード（macOSのみ）

```tensorlogic
model = load_model("model.mlpackage")
// または
model = load_model("model.mlmodelc")
```

### 2. メタデータ確認

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Neural Engine最適化のベストプラクティス

### 1. データ型: f16を使用

✅ 推奨: `compute_precision=ct.precision.FLOAT16`
❌ 非推奨: FLOAT32（GPUで実行される）

### 2. モデル形式: mlprogram形式を使用

✅ 推奨: `convert_to="mlprogram"`
❌ 非推奨: `convert_to="neuralnetwork"`（旧形式）

### 3. バッチサイズ: 1が最適

✅ 推奨: `batch_size=1`
⚠️ 注意: `batch_size>1`はGPUで実行される可能性

### 4. 入力サイズ: 固定サイズが最適

✅ 推奨: `shape=[1, 3, 224, 224]`
⚠️ 注意: 可変サイズは最適化が制限される

## サポートされる演算

### Neural Engineで高速実行される演算

- ✅ 畳み込み (conv2d, depthwise_conv)
- ✅ 全結合層 (linear, matmul)
- ✅ プーリング (max_pool, avg_pool)
- ✅ 正規化 (batch_norm, layer_norm)
- ✅ 活性化関数 (relu, gelu, sigmoid, tanh)
- ✅ エレメントワイズ演算 (add, mul, sub, div)

### GPUで実行される演算

- ⚠️ カスタム演算
- ⚠️ 複雑な制御フロー
- ⚠️ 非標準の活性化関数

## 性能比較

ResNet-50推論 (224x224画像):

| デバイス            | レイテンシ | 消費電力 | 効率   |
|-------------------|---------|--------|--------|
| Neural Engine     | ~3ms    | ~0.5W  | 最高   |
| Metal GPU (M1)    | ~8ms    | ~5W    | 中     |
| CPU (M1)          | ~50ms   | ~2W    | 低     |

※ Neural Engineは特に連続推論で電力効率が圧倒的

## 実用例：画像分類モデル

```tensorlogic
// CoreMLモデルをロード
let model = load_model("resnet50.mlpackage")

// 画像データを準備（224x224x3、f16形式）
let image = load_image("cat.jpg")
let preprocessed = preprocess(image)  // 正規化、リサイズ

// Neural Engineで推論実行
let output = model.predict(preprocessed)
let class_id = argmax(output)
print("Predicted class:", class_id)
```

## TensorLogic → CoreML エクスポート

TensorLogicで学習したモデルをCoreMLにエクスポート：

### 1. TensorLogicモデルをSafeTensors形式で保存

```tensorlogic
save_model(tl_model, "model.safetensors")
```

### 2. Pythonでロード＆CoreMLに変換

```python
import torch
from safetensors.torch import load_file
import coremltools as ct

# SafeTensorsをロード
weights = load_file("model.safetensors")

# PyTorchモデルに重みをロード
model = MyModel()
model.load_state_dict(weights)
model.eval()

# CoreMLに変換
example_input = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16
)
mlmodel.save("model.mlpackage")
```

### 3. TensorLogicでCoreMLモデルをロード

```tensorlogic
model = load_model("model.mlpackage")
```

## Neural Engine制約事項

### 1. macOS/iOS専用

- Apple Siliconが必要
- M1/M2/M3/M4シリーズ

### 2. 推論専用

- 学習はMetal GPUを使用
- 推論のみNeural Engine

### 3. モデルサイズ制限

- 推奨: < 1GB
- 大規模モデルは分割推論を検討

### 4. 演算制約

- f16のみ（f32は自動的にGPUへ）
- 標準的な畳み込みニューラルネットワークに最適
- Transformerも対応（最近のモデル）

## デバッグ＆最適化

### 1. Neural Engineが使われているか確認

- Instruments.appの「Core ML」テンプレートを使用
- "Compute Unit"がANE（Apple Neural Engine）になっているか確認

### 2. 性能プロファイリング

```python
# Pythonでプロファイル
import time
start = time.time()
predictions = model.predict({"input": image})
latency = time.time() - start
print(f"Latency: {latency*1000:.2f}ms")
```

### 3. 最適化のヒント

- バッチサイズ=1を維持
- f16精度を使用
- mlprogram形式を使用
- 不要な演算を削除

## よくある質問

### Q: Neural EngineとMetal GPUの違いは？

A: Neural Engineは推論専用で超低消費電力。Metal GPUは汎用的で学習にも使用可能。

### Q: どちらを選ぶべき？

A: 推論のみ → Neural Engine、学習 → Metal GPU、複雑な演算 → Metal GPU

### Q: CoreMLモデルの重みを取得できる？

A: CoreMLモデルはコンパイル済みで、重みに直接アクセスできない。学習にはSafeTensors形式を使用。

### Q: カスタムレイヤーは対応している？

A: 一部対応。coreMLtoolsでカスタム演算を定義可能だが、Neural Engineではなく、GPUで実行される可能性が高い。

### Q: 量子化モデルは対応している？

A: CoreMLは独自の量子化をサポート（W8A8など）。TensorLogicのGGUF量子化とは別のアプローチ。

## モデル形式の選び方

### 用途別推奨フォーマット

#### 学習: SafeTensors

- PyTorch互換
- 重みの保存/ロード
- Metal GPUで学習

#### 推論（iOS/macOS）: CoreML

- Neural Engine最適化
- 超低消費電力
- アプリ組み込み

#### 推論（汎用）: GGUF

- 量子化対応
- クロスプラットフォーム
- メモリ効率

#### 開発/デバッグ: SafeTensors

- 重みを直接確認可能
- 柔軟な変更が可能

## Neural Engine活用の実例

### 1. リアルタイム画像分類（iOS/macOS）

- レイテンシ: 3-5ms
- バッテリー消費: 最小

### 2. オブジェクト検出（YOLO, SSD）

- 30-60 FPS
- リアルタイムビデオ処理

### 3. 自然言語処理（BERT, GPT-2）

- トークン生成: 10-20ms/token
- オンデバイス処理

### 4. 画像生成（Stable Diffusion）

- 512x512画像: ~2秒
- CoreML Stable Diffusion使用

## 参考リンク

- [CoreML公式ドキュメント](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Neural Engineガイド](https://machinelearning.apple.com/research/neural-engine-transformers)
- [モデルローディングガイド](model_loading.md)
