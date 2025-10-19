# Phase 4完全実装: Neural Engine推論実行設計

## 概要

Phase 4基本実装（CoreMLモデルローダー）を拡張し、実際のNeural Engine推論を実行可能にする。Apple SiliconのNeural Engineを活用した高速・省電力推論を実現。

## 現状の制限

**Phase 4基本実装の状態:**
- ✅ CoreMLModelManager（モデルファイル読み込み）
- ✅ NeuralEngineBuffer（MLMultiArrayラッパー）
- ✅ Metal ↔ Neural Engine変換（SharedBuffer）
- ❌ 実際の推論実行（未実装）
- ❌ MLFeatureProvider統合（未実装）

## 設計目標

1. **推論実行**: CoreMLモデルを使った実際の推論
2. **入出力変換**: Tensor ↔ MLFeatureValue
3. **バッチ推論**: 複数入力の効率的処理
4. **モデル管理**: キャッシュ、メタデータ
5. **エラーハンドリング**: 推論失敗時の適切な処理

## Neural Engine概要

**Apple Neural Engine (ANE) とは:**
- Apple Silicon専用の機械学習アクセラレータ
- M4 Proには16-core Neural Engine搭載
- 最大38 TOPS (Trillion Operations Per Second)
- 超低消費電力（GPU比で10分の1）

**対応演算:**
- 畳み込み（Convolution）
- 完全結合層（Fully Connected）
- アテンション機構（Attention）
- 正規化（Normalization）
- 活性化関数（ReLU, GELU, Swish等）

**制約:**
- f16のみサポート（✅ TensorLogicはf16設計）
- 特定の演算のみ最適化
- モデルはCoreML形式が必要

## アーキテクチャ

```
┌─────────────────────────────────────────────┐
│      Neural Engine Inference System         │
│                                             │
│  ┌────────────────────────────────────────┐│
│  │  Model Management                      ││
│  │  - CoreMLModelManager                  ││
│  │  - Model caching                       ││
│  │  - Metadata extraction                 ││
│  └────────────────────────────────────────┘│
│                                             │
│  ┌────────────────────────────────────────┐│
│  │  Feature Conversion                    ││
│  │  - Tensor → MLFeatureValue             ││
│  │  - MLFeatureValue → Tensor             ││
│  │  - Batch processing                    ││
│  └────────────────────────────────────────┘│
│                                             │
│  ┌────────────────────────────────────────┐│
│  │  Inference Execution                   ││
│  │  - prediction(from:)                   ││
│  │  - Batch inference                     ││
│  │  - Error handling                      ││
│  └────────────────────────────────────────┘│
│                                             │
│  ┌────────────────────────────────────────┐│
│  │  Performance Optimization              ││
│  │  - Compute unit selection              ││
│  │  - Warm-up compilation                 ││
│  │  - Inference batching                  ││
│  └────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

## データ構造

### MLFeatureProvider実装

```rust
use objc2_core_ml::{MLFeatureValue, MLFeatureProvider, MLModel};
use objc2_foundation::{NSString, NSDictionary};

/// Tensor to MLFeatureProvider converter
pub struct TensorFeatureProvider {
    /// Feature name -> Tensor mapping
    features: HashMap<String, Tensor>,

    /// Cached MLFeatureValue instances
    cached_values: RefCell<HashMap<String, Retained<MLFeatureValue>>>,
}

impl TensorFeatureProvider {
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
            cached_values: RefCell::new(HashMap::new()),
        }
    }

    pub fn add_feature(&mut self, name: String, tensor: Tensor) {
        self.features.insert(name, tensor);
        // キャッシュをクリア
        self.cached_values.borrow_mut().clear();
    }

    /// Convert Tensor to MLFeatureValue
    fn tensor_to_feature_value(&self, tensor: &Tensor) -> TensorResult<Retained<MLFeatureValue>> {
        // TensorをNeural Engine bufferに変換
        let ne_buffer = tensor.buffer().to_neural_engine_buffer(tensor.shape())?;

        // MLFeatureValueを作成
        unsafe {
            let feature_value = MLFeatureValue::featureValueWithMultiArray(
                ne_buffer.as_ml_multi_array()
            );
            Ok(feature_value)
        }
    }
}

// objc2を使ったMLFeatureProviderの実装
// IMPORTANT: これはObjective-Cプロトコルの実装
impl MLFeatureProvider for TensorFeatureProvider {
    fn featureValueForName(&self, feature_name: &NSString) -> Option<Retained<MLFeatureValue>> {
        let name = feature_name.to_string();

        // キャッシュをチェック
        if let Some(cached) = self.cached_values.borrow().get(&name) {
            return Some(cached.clone());
        }

        // Tensorから変換
        if let Some(tensor) = self.features.get(&name) {
            if let Ok(value) = self.tensor_to_feature_value(tensor) {
                self.cached_values.borrow_mut().insert(name.clone(), value.clone());
                return Some(value);
            }
        }

        None
    }

    fn featureNames(&self) -> Retained<NSSet<NSString>> {
        let names: Vec<_> = self.features.keys()
            .map(|k| NSString::from_str(k))
            .collect();

        unsafe { NSSet::setWithArray(&NSArray::arrayWithObjects(&names)) }
    }
}
```

### NeuralEngineInference

```rust
pub struct NeuralEngineInference {
    /// CoreML model
    model: Retained<MLModel>,

    /// Model metadata
    metadata: ModelMetadata,

    /// Input feature descriptions
    input_descriptions: HashMap<String, FeatureDescription>,

    /// Output feature descriptions
    output_descriptions: HashMap<String, FeatureDescription>,
}

pub struct ModelMetadata {
    pub name: String,
    pub author: String,
    pub license: String,
    pub description: String,
    pub version: String,
}

pub struct FeatureDescription {
    pub name: String,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub is_optional: bool,
}

impl NeuralEngineInference {
    /// Load model from file
    pub fn load(path: &str) -> TensorResult<Self> {
        let model = CoreMLModelManager::load_model(path)?;

        // Extract metadata
        let metadata = Self::extract_metadata(&model)?;

        // Extract input/output descriptions
        let input_descriptions = Self::extract_input_descriptions(&model)?;
        let output_descriptions = Self::extract_output_descriptions(&model)?;

        Ok(Self {
            model,
            metadata,
            input_descriptions,
            output_descriptions,
        })
    }

    /// Perform inference
    pub fn predict(&self, inputs: HashMap<String, Tensor>) -> TensorResult<HashMap<String, Tensor>> {
        // 1. Create feature provider from tensors
        let mut provider = TensorFeatureProvider::new();
        for (name, tensor) in inputs {
            provider.add_feature(name, tensor);
        }

        // 2. Perform prediction
        let output = unsafe {
            self.model.predictionFromFeatures_error(&provider)
                .map_err(|e| TensorError::NeuralEngineError(format!("Prediction failed: {:?}", e)))?
        };

        // 3. Convert output features to tensors
        let mut results = HashMap::new();
        for (name, desc) in &self.output_descriptions {
            if let Some(feature_value) = output.featureValueForName(&NSString::from_str(name)) {
                let tensor = self.feature_value_to_tensor(&feature_value, &desc)?;
                results.insert(name.clone(), tensor);
            }
        }

        Ok(results)
    }

    /// Batch inference (multiple inputs)
    pub fn predict_batch(&self, batch_inputs: Vec<HashMap<String, Tensor>>) -> TensorResult<Vec<HashMap<String, Tensor>>> {
        let mut results = Vec::with_capacity(batch_inputs.len());

        for inputs in batch_inputs {
            let output = self.predict(inputs)?;
            results.push(output);
        }

        Ok(results)
    }

    /// Convert MLFeatureValue to Tensor
    fn feature_value_to_tensor(
        &self,
        value: &MLFeatureValue,
        desc: &FeatureDescription,
    ) -> TensorResult<Tensor> {
        unsafe {
            // MLMultiArrayを取得
            let multi_array = value.multiArrayValue()
                .ok_or_else(|| TensorError::NeuralEngineError("Not a multi-array".to_string()))?;

            // NeuralEngineBufferに変換
            let ne_buffer = NeuralEngineBuffer::from_ml_multi_array(multi_array);

            // Tensorに変換
            Tensor::from_neural_engine_buffer(ne_buffer, desc.shape.clone())
        }
    }

    /// Extract model metadata
    fn extract_metadata(model: &MLModel) -> TensorResult<ModelMetadata> {
        unsafe {
            let description = model.modelDescription();

            Ok(ModelMetadata {
                name: description.metadata()
                    .objectForKey(&NSString::from_str("MLModelCreatorDefinedKey"))
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "Unknown".to_string()),
                author: "".to_string(),
                license: "".to_string(),
                description: "".to_string(),
                version: "".to_string(),
            })
        }
    }

    /// Extract input feature descriptions
    fn extract_input_descriptions(model: &MLModel) -> TensorResult<HashMap<String, FeatureDescription>> {
        unsafe {
            let description = model.modelDescription();
            let input_desc = description.inputDescriptionsByName();

            let mut descriptions = HashMap::new();

            for (name, feature_desc) in input_desc.iter() {
                let name_str = name.to_string();

                let desc = FeatureDescription {
                    name: name_str.clone(),
                    dtype: DataType::Float16,  // ANEはf16
                    shape: Self::extract_shape(&feature_desc)?,
                    is_optional: false,
                };

                descriptions.insert(name_str, desc);
            }

            Ok(descriptions)
        }
    }

    /// Extract shape from feature description
    fn extract_shape(feature_desc: &MLFeatureDescription) -> TensorResult<Vec<usize>> {
        unsafe {
            let constraint = feature_desc.multiArrayConstraint()
                .ok_or_else(|| TensorError::NeuralEngineError("Not a multi-array feature".to_string()))?;

            let shape_array = constraint.shape();
            let count = shape_array.count();

            let mut shape = Vec::with_capacity(count);
            for i in 0..count {
                let dim = shape_array.objectAtIndex(i);
                shape.push(dim.integerValue() as usize);
            }

            Ok(shape)
        }
    }
}
```

### ComputeUnit選択

```rust
pub enum ComputeUnit {
    /// CPU only
    CPUOnly,

    /// CPU and GPU
    CPUAndGPU,

    /// All available units (CPU, GPU, Neural Engine)
    All,

    /// Neural Engine only (if available)
    NeuralEngineOnly,
}

impl NeuralEngineInference {
    /// Load model with specific compute unit
    pub fn load_with_compute_unit(
        path: &str,
        compute_unit: ComputeUnit,
    ) -> TensorResult<Self> {
        let config = unsafe {
            let config = MLModelConfiguration::new();

            let unit = match compute_unit {
                ComputeUnit::CPUOnly => MLComputeUnitsCPUOnly,
                ComputeUnit::CPUAndGPU => MLComputeUnitsCPUAndGPU,
                ComputeUnit::All => MLComputeUnitsAll,
                ComputeUnit::NeuralEngineOnly => MLComputeUnitsAll,  // ANE優先
            };

            config.setComputeUnits(unit);
            config
        };

        // 設定を使ってモデルをロード
        let model = CoreMLModelManager::load_model_with_config(path, &config)?;

        // ... metadata extraction ...

        Ok(Self { model, /* ... */ })
    }
}
```

## 実装計画

### Phase 4.1: MLFeatureProvider実装

1. **TensorFeatureProvider**
   - Tensor → MLFeatureValue変換
   - MLFeatureProviderプロトコル実装
   - キャッシング機構

2. **テスト**
   - 単一入力の変換
   - 複数入力の変換
   - 形状検証

### Phase 4.2: 推論実行

1. **NeuralEngineInference**
   - predict()メソッド
   - MLModel::prediction()呼び出し
   - 出力変換

2. **テスト**
   - 簡単なモデルでの推論
   - 入出力形状の検証
   - エラーハンドリング

### Phase 4.3: メタデータ抽出

1. **モデル情報取得**
   - 入力記述
   - 出力記述
   - モデルメタデータ

2. **形状推論**
   - 動的形状対応
   - バッチサイズ処理

### Phase 4.4: バッチ推論

1. **効率的なバッチ処理**
   - 複数入力の並列化
   - メモリ効率

2. **パフォーマンス最適化**
   - Warm-up compilation
   - ComputeUnit選択

## 使用例

### 基本的な推論

```rust
use tensorlogic::neural_engine::NeuralEngineInference;

// モデルロード
let model = NeuralEngineInference::load("model.mlmodelc")?;

// 入力準備
let input = Tensor::randn(vec![1, 224, 224, 3])?;  // Batch=1, 224x224 RGB
let mut inputs = HashMap::new();
inputs.insert("input".to_string(), input);

// 推論実行
let outputs = model.predict(inputs)?;

// 結果取得
let prediction = outputs.get("output").unwrap();
println!("Prediction shape: {:?}", prediction.shape());
```

### バッチ推論

```rust
// 複数画像の推論
let images = vec![
    load_image("cat.jpg")?,
    load_image("dog.jpg")?,
    load_image("bird.jpg")?,
];

let batch_inputs: Vec<_> = images.into_iter()
    .map(|img| {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), img);
        inputs
    })
    .collect();

// バッチ推論
let batch_outputs = model.predict_batch(batch_inputs)?;

for (i, output) in batch_outputs.iter().enumerate() {
    let pred = output.get("output").unwrap();
    println!("Image {}: {:?}", i, pred);
}
```

### ComputeUnit指定

```rust
// Neural Engine優先でロード
let model = NeuralEngineInference::load_with_compute_unit(
    "model.mlmodelc",
    ComputeUnit::All,  // ANE > GPU > CPU の順で選択
)?;

// CPU専用（デバッグ用）
let cpu_model = NeuralEngineInference::load_with_compute_unit(
    "model.mlmodelc",
    ComputeUnit::CPUOnly,
)?;
```

### モデル情報の取得

```rust
let model = NeuralEngineInference::load("model.mlmodelc")?;

println!("Model: {}", model.metadata.name);
println!("Version: {}", model.metadata.version);

println!("Inputs:");
for (name, desc) in &model.input_descriptions {
    println!("  {}: {:?} {:?}", name, desc.dtype, desc.shape);
}

println!("Outputs:");
for (name, desc) in &model.output_descriptions {
    println!("  {}: {:?} {:?}", name, desc.dtype, desc.shape);
}
```

### Warm-up（初回コンパイル）

```rust
// モデルロード時に一度推論してコンパイル
let model = NeuralEngineInference::load("model.mlmodelc")?;

// Dummy入力でwarm-up
let dummy_input = Tensor::zeros(vec![1, 224, 224, 3])?;
let mut inputs = HashMap::new();
inputs.insert("input".to_string(), dummy_input);

// 初回は遅い（コンパイル）
let _ = model.predict(inputs.clone())?;

// 2回目以降は高速
let start = std::time::Instant::now();
let output = model.predict(inputs)?;
let elapsed = start.elapsed();

println!("Inference time: {:?}", elapsed);
```

## テスト戦略

### 単体テスト

1. **Feature変換**
   - Tensor → MLFeatureValue
   - MLFeatureValue → Tensor
   - 形状保存の検証

2. **モデルロード**
   - 正常ロード
   - 不正なパスでのエラー
   - メタデータ抽出

### 統合テスト

1. **実際のモデルでの推論**
   - MobileNetV2 (画像分類)
   - ResNet50 (特徴抽出)
   - BERT (テキスト)

2. **パフォーマンステスト**
   - CPU vs GPU vs Neural Engine
   - バッチサイズの影響
   - レイテンシ測定

### 検証テスト

1. **精度検証**
   - PyTorchと同じ結果が出るか
   - 数値誤差の許容範囲

2. **メモリ使用量**
   - リーク検出
   - バッチサイズとメモリの関係

## 期待される結果

| モデル | Neural Engine | GPU | CPU |
|--------|--------------|-----|-----|
| MobileNetV2 | 2ms | 8ms | 50ms |
| ResNet50 | 5ms | 20ms | 200ms |
| 消費電力 | 100mW | 1W | 500mW |

**Neural Engineの利点:**
- **速度**: GPU比で2-5倍高速
- **省電力**: GPU比で10倍省電力
- **バッテリー**: モバイルデバイスで長時間動作

## 実装ノート

- **objc2の使用**: Objective-C APIへの安全なアクセス
- **Send/Sync**: Neural Engine APIはスレッド安全
- **エラーハンドリング**: NSErrorの適切な処理
- **メモリ管理**: Retained<T>でARC管理

## 制限事項

- **モデル形式**: CoreML (.mlmodel, .mlmodelc) のみ
- **データ型**: f16のみ（ANEの制約）
- **演算制限**: ANE非対応演算はCPU/GPUフォールバック
- **macOS/iOS専用**: Apple Silicon必須

## 将来の拡張

1. **モデルキャッシュ**: 頻繁に使うモデルをメモリ保持
2. **非同期推論**: バックグラウンドスレッドでの実行
3. **ストリーミング**: リアルタイム処理
4. **量子化サポート**: INT8モデル
5. **モデル変換**: PyTorch/TensorFlow → CoreML

## 参考文献

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Apple Neural Engine Overview](https://github.com/hollance/neural-engine)
- [Core ML Performance Guide](https://developer.apple.com/documentation/coreml/core_ml_api/reducing_the_memory_footprint_of_your_app)
- [Converting Models to Core ML](https://coremltools.readme.io/)
