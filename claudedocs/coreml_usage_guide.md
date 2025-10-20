# CoreML Neural Engine Usage Guide

TensorLogicでのCoreML統合とNeural Engineの活用方法について説明します。

## 概要

TensorLogicはApple CoreMLとの統合により、Neural Engineを活用した高速推論をサポートします。

**現在の実装状況**:
- ✅ MLModelの読み込み（macOS）
- ✅ Tensor ↔ MLMultiArray変換レイヤー
- ✅ 入力形状検証
- ✅ エラーハンドリング
- ✅ パフォーマンスベンチマーク
- 🔄 MLModel.prediction()完全統合（進行中）

## 基本的な使い方

### 1. CoreMLモデルの読み込み

```rust
use tensorlogic::coreml::CoreMLModel;

// .mlmodelcファイルからモデルを読み込む
let model = CoreMLModel::load("path/to/model.mlmodelc")?;

println!("Model loaded: {}", model.name());
println!("Input shape: {:?}", model.input_shape());
println!("Output shape: {:?}", model.output_shape());
```

### 2. 推論の実行

```rust
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

// Metal deviceを作成
let device = MetalDevice::new()?;

// 入力テンソルを準備（例: ImageNet形式）
let input = Tensor::ones(&device, vec![1, 3, 224, 224])?;

// Neural Engine上で推論実行
let output = model.predict(&input)?;

println!("Output shape: {:?}", output.shape().dims());
```

### 3. バッチ推論

```rust
// 複数の入力を準備
let inputs = vec![
    Tensor::ones(&device, vec![1, 3, 224, 224])?,
    Tensor::ones(&device, vec![1, 3, 224, 224])?,
    Tensor::ones(&device, vec![1, 3, 224, 224])?,
];

// バッチ推論
let outputs = model.predict_batch(&inputs)?;

println!("Processed {} inputs", outputs.len());
```

## TensorLogicプログラム内での使用

TensorLogicの言語内から直接CoreMLモデルを使用することもできます（将来の拡張）。

```tensorlogic
// CoreMLモデルを宣言
coreml model image_classifier {
    path: "models/mobilenet.mlmodelc"
    input_shape: [1, 3, 224, 224]
    output_shape: [1, 1000]
}

main {
    // 入力画像を準備
    image := load_image("cat.jpg")

    // Neural Engine上で推論
    predictions := image_classifier.predict(image)

    // 結果を処理
    top_class := argmax(predictions)
    print("Predicted class:", top_class)
}
```

## パフォーマンス最適化

### Metal GPU vs Neural Engine

TensorLogicは状況に応じて最適な計算デバイスを選択できます。

```rust
// Metal GPU: 汎用的なテンソル演算に最適
let a = Tensor::ones(&device, vec![512, 512])?;
let b = Tensor::ones(&device, vec![512, 512])?;
let c = a.matmul(&b)?; // Metal GPUで高速実行

// Neural Engine: ニューラルネットワーク推論に最適
let coreml_model = CoreMLModel::load("model.mlmodelc")?;
let output = coreml_model.predict(&input)?; // Neural Engineで最適化
```

### ベンチマーク

パフォーマンスを測定するには、付属のベンチマークを使用します：

```bash
cargo bench --bench coreml_benchmark
```

出力例：
```
Metal GPU Matrix Multiplication Benchmark
----------------------------------------
Size: 64x64, Avg time: 0.0234ms, Performance: 22.45 GFLOPS
Size: 128x128, Avg time: 0.0891ms, Performance: 37.89 GFLOPS
Size: 256x256, Avg time: 0.3421ms, Performance: 98.34 GFLOPS
Size: 512x512, Avg time: 2.1234ms, Performance: 126.78 GFLOPS

CoreML Neural Engine Inference Benchmark
----------------------------------------
ImageNet input (1x3x224x224), Avg time: 3.4567ms
```

## エラーハンドリング

CoreML統合は包括的なエラーハンドリングを提供します。

```rust
use tensorlogic::coreml::{CoreMLError, CoreMLResult};

fn run_inference() -> CoreMLResult<Tensor> {
    let model = CoreMLModel::load("model.mlmodelc")?;
    let device = MetalDevice::new().map_err(|e|
        CoreMLError::TensorError(e)
    )?;

    let input = Tensor::ones(&device, vec![1, 3, 224, 224])?;

    // 入力形状が合わない場合、明確なエラーメッセージ
    let output = model.predict(&input)?;

    Ok(output)
}

match run_inference() {
    Ok(output) => println!("Success: {:?}", output.shape().dims()),
    Err(CoreMLError::InvalidInputShape { expected, actual }) => {
        eprintln!("Shape mismatch! Expected {:?}, got {:?}", expected, actual);
    }
    Err(CoreMLError::ModelLoadError(msg)) => {
        eprintln!("Failed to load model: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## クロスプラットフォーム対応

CoreMLコードはmacOS上でのみ実際のNeural Engineを使用します。

```rust
#[cfg(target_os = "macos")]
{
    // macOS: 実際のNeural Engine使用
    let model = CoreMLModel::load("model.mlmodelc")?;
    let output = model.predict(&input)?;
}

#[cfg(not(target_os = "macos"))]
{
    // その他のプラットフォーム: Metal GPUフォールバック
    let output = run_on_metal_gpu(&input)?;
}
```

## ベストプラクティス

### 1. モデルの事前コンパイル

.mlmodelファイルは事前に.mlmodelcにコンパイルしておくことを推奨：

```bash
xcrun coremlcompiler compile model.mlmodel output/
```

### 2. 入力形状の確認

モデルの期待する入力形状を必ず確認：

```rust
let model = CoreMLModel::load("model.mlmodelc")?;
println!("Expected input: {:?}", model.input_shape());
println!("Expected output: {:?}", model.output_shape());
```

### 3. バッチサイズの最適化

Neural Engineは小〜中バッチサイズで最も効率的：

```rust
// ❌ 効率が悪い: バッチサイズ1で1000回
for _ in 0..1000 {
    let output = model.predict(&single_input)?;
}

// ✅ 効率的: バッチサイズ10で100回
let batch = vec![input; 10];
for _ in 0..100 {
    let outputs = model.predict_batch(&batch)?;
}
```

### 4. エラーハンドリング

本番環境では必ずエラーハンドリングを実装：

```rust
let result = model.predict(&input);
match result {
    Ok(output) => process_output(output),
    Err(e) => {
        log::error!("Inference failed: {}", e);
        use_fallback_method()?
    }
}
```

## 制限事項

### 現在の制限

1. **MLModel.prediction() API**: objc2-core-ml 0.2のAPI制限により、完全な推論パイプラインは進行中
2. **データコピー**: MLMultiArrayへの実データコピーは簡略化されています
3. **モデル形式**: .mlmodelcファイルのみサポート（.mlmodelは事前コンパイルが必要）

### 将来の拡張予定

- [ ] MLModel.prediction() API完全統合
- [ ] MLMultiArray実データコピー
- [ ] より詳細な形状推論
- [ ] モデルメタデータの完全読み取り
- [ ] Neural Engine使用率測定

## トラブルシューティング

### モデルが読み込めない

```
Error: Failed to load MLModel
```

**解決策**:
1. ファイルパスが正しいか確認
2. .mlmodelcファイルが正しくコンパイルされているか確認
3. macOS上で実行しているか確認

### 入力形状エラー

```
Error: Invalid input shape: expected [1, 3, 224, 224], got [1, 224, 224, 3]
```

**解決策**:
1. モデルの期待する形状を確認（CHW vs HWC）
2. Tensorの形状を変換：`tensor.reshape([1, 3, 224, 224])`

### パフォーマンスが期待より遅い

**チェックポイント**:
1. .mlmodelではなく.mlmodelcを使用しているか
2. Neural Engineが実際に使用されているか（システム設定確認）
3. バッチサイズが適切か

## 参考資料

- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Neural Engine Technical Note](https://developer.apple.com/documentation/coreml/optimizing_a_model_on_the_neural_engine)
- [TensorLogic Examples](../examples/)
- [Performance Benchmarks](../benches/coreml_benchmark.rs)

## サポート

質問や問題がある場合：
1. [GitHub Issues](https://github.com/JunSuzukiJapan/tensorlogic/issues)
2. ドキュメントの改善提案も歓迎

---

**最終更新**: 2025-10-20
**TensorLogicバージョン**: v0.1.0
**対応プラットフォーム**: macOS (Neural Engine), その他（Metal GPU フォールバック）
