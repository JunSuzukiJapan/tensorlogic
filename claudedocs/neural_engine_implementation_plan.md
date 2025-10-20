# Neural Engine完全統合 実装計画

**作成日**: 2025-10-20
**推定工数**: 14-20時間
**優先度**: 低（MVPは動作中、完全統合はオプション）

---

## 📋 概要

TensorLogicのCoreML/Neural Engine統合を完成させ、実際のMLモデルで推論を実行できるようにする。

### 現在の状態
- ✅ MLModel読み込み（objc2-core-ml 0.2）
- ✅ Tensor → MLMultiArray変換
- ✅ MLMultiArray → Tensor変換
- ❌ 実際の推論実行（モックのゼロテンソルを返す）

### 目標
- ✅ 実際のMLModel.predictionFromFeatures実行
- ✅ モデルのinput/output名の自動取得
- ✅ 完全なデータフロー: Tensor → MLMultiArray → MLFeatureValue → 推論 → Tensor

---

## 🎯 Phase 1: 事前調査（完了）✅

**工数**: 1時間 ✅
**状態**: 完了

### 実施内容
- [x] objc2-core-ml 0.2のAPI調査
- [x] 利用可能なクラスとメソッドの確認
- [x] 実装パターンの決定

### 確認したAPI
```rust
// MLFeatureValue
MLFeatureValue::featureValueWithMultiArray(&MLMultiArray) -> Retained<MLFeatureValue>
feature_value.multiArrayValue() -> Option<Retained<MLMultiArray>>

// MLDictionaryFeatureProvider
MLDictionaryFeatureProvider::initWithDictionary_error(
    &NSDictionary<NSString, AnyObject>
) -> Result<Retained<Self>, NSError>

// MLModelDescription
ml_model.modelDescription() -> Retained<MLModelDescription>
description.inputDescriptionsByName() -> NSDictionary<NSString, MLFeatureDescription>
description.outputDescriptionsByName() -> NSDictionary<NSString, MLFeatureDescription>

// MLModel
ml_model.predictionFromFeatures_error(
    &ProtocolObject<dyn MLFeatureProvider>
) -> Result<ProtocolObject<dyn MLFeatureProvider>, NSError>
```

---

## 🔧 Phase 2: Cargo.toml feature flags設定

**工数**: 1時間
**ファイル**: `Cargo.toml`
**状態**: 未実装

### 現在の設定
```toml
objc2-core-ml = { version = "0.2", features = ["MLMultiArray", "MLModel"] }
objc2-foundation = { version = "0.2", features = ["NSArray", "NSValue", "NSError", "NSString"] }
```

### 追加するfeatures

#### objc2-core-ml
```toml
objc2-core-ml = { version = "0.2", features = [
    "MLMultiArray",           # 既存
    "MLModel",                # 既存
    "MLFeatureValue",         # 🆕 FeatureValue作成
    "MLFeatureProvider",      # 🆕 プロトコル
    "MLDictionaryFeatureProvider",  # 🆕 入力データ提供
    "MLModelDescription",     # 🆕 モデルメタデータ
    "MLFeatureDescription",   # 🆕 入力/出力情報
] }
```

#### objc2-foundation（追加要否確認）
```toml
objc2-foundation = { version = "0.2", features = [
    "NSArray", "NSValue", "NSError", "NSString",  # 既存
    "NSDictionary",  # 🆕 入力辞書作成（既に使用中なら不要）
] }
```

### 実装手順
1. Cargo.tomlを編集
2. `cargo check`でコンパイル確認
3. feature依存関係のエラーがあれば追加調整

### 検証
```bash
cargo build --lib
cargo test --lib coreml --quiet
```

---

## 📝 Phase 3: MLModelDescription統合

**工数**: 2-3時間
**ファイル**: `src/coreml/model.rs`
**状態**: 未実装

### 目的
モデルのinput/output名と形状を自動取得し、ハードコードされた値を削除する。

### 実装箇所
`CoreMLModel::load()` メソッド（src/coreml/model.rs:56-118）

### 現在のコード（抜粋）
```rust
// TODO: Extract input/output shapes from model description
// For now, use default ImageNet shapes
Ok(CoreMLModel {
    name,
    path: path_str,
    input_shape: vec![1, 3, 224, 224],  // ハードコード
    output_shape: vec![1, 1000],         // ハードコード
    ml_model: Some(ml_model),
})
```

### 新しい実装

#### 1. 構造体の拡張
```rust
pub struct CoreMLModel {
    name: String,
    path: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_name: String,   // 🆕 入力名
    output_name: String,  // 🆕 出力名
    #[cfg(target_os = "macos")]
    ml_model: Option<Retained<MLModel>>,
}
```

#### 2. load()メソッドの実装
```rust
#[cfg(target_os = "macos")]
{
    use objc2_core_ml::{MLModel, MLModelDescription};
    use objc2_foundation::{NSString, NSURL};

    let ml_model = unsafe { MLModel::modelWithContentsOfURL_error(&url)? };

    // MLModelDescriptionを取得
    let description = unsafe { ml_model.modelDescription() };

    // 入力情報を取得
    let input_dict = unsafe { description.inputDescriptionsByName() };
    let input_keys = unsafe { input_dict.allKeys() };

    if input_keys.count() == 0 {
        return Err(CoreMLError::ModelLoadError(
            "No input descriptions found".to_string()
        ));
    }

    // 最初の入力を使用
    let input_name = unsafe { input_keys.objectAtIndex(0) };
    let input_name_str = unsafe { input_name.as_ref().to_string() };

    let input_desc = unsafe { input_dict.objectForKey(input_name).unwrap() };
    let input_shape = extract_shape_from_description(&input_desc)?;

    // 出力情報を取得（同様の処理）
    let output_dict = unsafe { description.outputDescriptionsByName() };
    let output_keys = unsafe { output_dict.allKeys() };

    let output_name = unsafe { output_keys.objectAtIndex(0) };
    let output_name_str = unsafe { output_name.as_ref().to_string() };

    let output_desc = unsafe { output_dict.objectForKey(output_name).unwrap() };
    let output_shape = extract_shape_from_description(&output_desc)?;

    Ok(CoreMLModel {
        name,
        path: path_str,
        input_shape,
        output_shape,
        input_name: input_name_str,
        output_name: output_name_str,
        ml_model: Some(ml_model),
    })
}
```

#### 3. ヘルパー関数
```rust
#[cfg(target_os = "macos")]
fn extract_shape_from_description(
    desc: &objc2_core_ml::MLFeatureDescription
) -> CoreMLResult<Vec<usize>> {
    // MLFeatureDescriptionから形状を抽出
    // multiArrayConstraint().shape() などから取得
    // 実装詳細はMLFeatureDescriptionのAPIに依存

    // 簡易版: とりあえずデフォルト形状を返す
    // 完全版では実際にAPIを呼び出す
    Ok(vec![1, 3, 224, 224])
}
```

### エラーハンドリング
- 入力/出力が存在しない場合
- 複数の入力/出力がある場合（最初のものを使用）
- 形状情報が取得できない場合

### テスト
```rust
#[test]
#[cfg(target_os = "macos")]
fn test_model_description_extraction() {
    // モデルファイルが必要なので、実際のテストは統合テスト
    // ユニットテストでは構造体のアクセサのみテスト
}
```

---

## 🔄 Phase 4: MLFeatureValue統合

**工数**: 4-6時間
**ファイル**: `src/coreml/conversion.rs`
**状態**: 未実装

### 目的
MLMultiArrayからMLFeatureValueへの変換関数を追加する。

### 新規関数

#### 1. mlmultiarray_to_feature_value()
```rust
#[cfg(target_os = "macos")]
pub fn mlmultiarray_to_feature_value(
    ml_array: &objc2_core_ml::MLMultiArray
) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLFeatureValue>> {
    use objc2_core_ml::MLFeatureValue;

    println!("Converting MLMultiArray to MLFeatureValue");

    // MLFeatureValue::featureValueWithMultiArray を呼び出し
    let feature_value = unsafe {
        MLFeatureValue::featureValueWithMultiArray(ml_array)
    };

    println!("  MLFeatureValue created successfully");
    Ok(feature_value)
}
```

#### 2. feature_value_to_mlmultiarray()
```rust
#[cfg(target_os = "macos")]
pub fn feature_value_to_mlmultiarray(
    feature_value: &objc2_core_ml::MLFeatureValue
) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLMultiArray>> {
    println!("Extracting MLMultiArray from MLFeatureValue");

    let ml_array = unsafe {
        feature_value.multiArrayValue()
            .ok_or_else(|| CoreMLError::ConversionError(
                "FeatureValue does not contain MLMultiArray".to_string()
            ))?
    };

    println!("  MLMultiArray extracted successfully");
    Ok(ml_array)
}
```

#### 3. 非macOS版のplaceholder
```rust
#[cfg(not(target_os = "macos"))]
pub fn mlmultiarray_to_feature_value() -> CoreMLResult<()> {
    println!("MLFeatureValue conversion (non-macOS placeholder)");
    Ok(())
}
```

### tensor_to_mlmultiarray()の拡張
現在の関数は `-> CoreMLResult<()>` を返しますが、実際のMLMultiArrayを返すように変更:

```rust
#[cfg(target_os = "macos")]
pub fn tensor_to_mlmultiarray(
    tensor: &Tensor
) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLMultiArray>> {
    // 既存の実装 + 最後に multi_array を返す
    Ok(multi_array)
}
```

### テスト
```rust
#[test]
#[cfg(target_os = "macos")]
fn test_mlmultiarray_to_feature_value() {
    let device = MetalDevice::new().unwrap();
    let tensor = Tensor::ones(&device, vec![1, 10]).unwrap();

    let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
    let feature_value = mlmultiarray_to_feature_value(&ml_array).unwrap();

    // FeatureValueからMLMultiArrayを取り戻せることを確認
    let recovered = feature_value_to_mlmultiarray(&feature_value).unwrap();
    assert!(recovered.is_some());
}
```

---

## 📦 Phase 5: MLDictionaryFeatureProvider統合

**工数**: 3-4時間
**ファイル**: `src/coreml/model.rs`
**状態**: 未実装

### 目的
入力データをMLDictionaryFeatureProviderとして構築し、モデルに渡す。

### 実装箇所
`CoreMLModel::predict()` メソッド内

### 実装コード

#### 1. NSDictionary作成
```rust
use objc2_foundation::{NSDictionary, NSString};
use objc2_core_ml::{MLFeatureValue, MLDictionaryFeatureProvider};
use objc2::ProtocolObject;

// 1. Tensor → MLMultiArray
let ml_array = super::conversion::tensor_to_mlmultiarray(input)?;

// 2. MLMultiArray → MLFeatureValue
let feature_value = super::conversion::mlmultiarray_to_feature_value(&ml_array)?;

// 3. NSDictionaryを作成
let input_name_ns = NSString::from_str(&self.input_name);

// AnyObjectにキャスト
let feature_value_obj = unsafe {
    std::mem::transmute::<
        objc2::rc::Retained<MLFeatureValue>,
        objc2::rc::Retained<objc2::runtime::AnyObject>
    >(feature_value)
};

// NSDictionary作成
let dict = NSDictionary::from_keys_and_objects(
    &[&*input_name_ns],
    &[&*feature_value_obj],
);
```

#### 2. MLDictionaryFeatureProvider作成
```rust
// 4. MLDictionaryFeatureProvider作成
let input_provider = unsafe {
    let allocated = MLDictionaryFeatureProvider::alloc();
    MLDictionaryFeatureProvider::initWithDictionary_error(allocated, &dict)
        .map_err(|e| CoreMLError::ConversionError(
            format!("Failed to create feature provider: {:?}", e)
        ))?
};
```

#### 3. ProtocolObjectへのキャスト
```rust
use objc2_core_ml::MLFeatureProvider;

// 5. ProtocolObject<dyn MLFeatureProvider>にキャスト
let provider_protocol: &ProtocolObject<dyn MLFeatureProvider> =
    ProtocolObject::from_ref(&*input_provider);
```

### エラーハンドリング
- NSDictionary作成失敗
- MLDictionaryFeatureProvider初期化失敗
- 型キャストエラー

### デバッグログ
```rust
println!("Creating MLDictionaryFeatureProvider:");
println!("  Input name: {}", self.input_name);
println!("  Feature count: 1");
```

---

## 🚀 Phase 6: 完全prediction()実装

**工数**: 4-6時間
**ファイル**: `src/coreml/model.rs`
**状態**: 未実装

### 目的
すべてのコンポーネントを統合し、実際のNeural Engine推論を実行する。

### 完全なpredict()実装

```rust
pub fn predict(&self, input: &Tensor) -> CoreMLResult<Tensor> {
    // 入力形状検証
    let input_dims = input.shape().dims();
    if input_dims != self.input_shape {
        return Err(CoreMLError::InvalidInputShape {
            expected: self.input_shape.clone(),
            actual: input_dims.to_vec(),
        });
    }

    use crate::device::MetalDevice;
    let device = MetalDevice::new().map_err(|e| CoreMLError::TensorError(e))?;

    #[cfg(target_os = "macos")]
    {
        if let Some(ref ml_model) = self.ml_model {
            use super::conversion::{tensor_to_mlmultiarray, mlmultiarray_to_feature_value};
            use objc2_foundation::{NSDictionary, NSString};
            use objc2_core_ml::{MLFeatureValue, MLDictionaryFeatureProvider, MLFeatureProvider};
            use objc2::ProtocolObject;

            println!("Running CoreML inference on Neural Engine...");
            println!("  Model: {}", self.name);
            println!("  Input: {} → Output: {}", self.input_name, self.output_name);

            // Step 1: Tensor → MLMultiArray
            let ml_array = tensor_to_mlmultiarray(input)?;
            println!("  ✓ MLMultiArray created");

            // Step 2: MLMultiArray → MLFeatureValue
            let feature_value = mlmultiarray_to_feature_value(&ml_array)?;
            println!("  ✓ MLFeatureValue created");

            // Step 3: Create NSDictionary
            let input_name_ns = NSString::from_str(&self.input_name);
            let feature_value_obj = unsafe {
                std::mem::transmute::<_, objc2::rc::Retained<objc2::runtime::AnyObject>>(
                    feature_value
                )
            };
            let dict = NSDictionary::from_keys_and_objects(
                &[&*input_name_ns],
                &[&*feature_value_obj],
            );
            println!("  ✓ Input dictionary created");

            // Step 4: Create MLDictionaryFeatureProvider
            let input_provider = unsafe {
                let allocated = MLDictionaryFeatureProvider::alloc();
                MLDictionaryFeatureProvider::initWithDictionary_error(allocated, &dict)
                    .map_err(|e| CoreMLError::ConversionError(
                        format!("Failed to create feature provider: {:?}", e)
                    ))?
            };
            println!("  ✓ Feature provider created");

            // Step 5: Cast to ProtocolObject
            let provider_protocol: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*input_provider);

            // Step 6: Run prediction on Neural Engine
            println!("  → Running Neural Engine inference...");
            let output_provider = unsafe {
                ml_model.predictionFromFeatures_error(provider_protocol)
                    .map_err(|e| CoreMLError::InferenceError(
                        format!("Prediction failed: {:?}", e)
                    ))?
            };
            println!("  ✓ Neural Engine inference completed");

            // Step 7: Extract output MLFeatureValue
            let output_name_ns = NSString::from_str(&self.output_name);
            let output_value = unsafe {
                output_provider.featureValueForName(&output_name_ns)
                    .ok_or_else(|| CoreMLError::ConversionError(
                        format!("Output '{}' not found", self.output_name)
                    ))?
            };
            println!("  ✓ Output feature extracted: {}", self.output_name);

            // Step 8: Extract MLMultiArray from output
            let output_array = unsafe {
                output_value.multiArrayValue()
                    .ok_or_else(|| CoreMLError::ConversionError(
                        "Output is not MLMultiArray".to_string()
                    ))?
            };
            println!("  ✓ Output MLMultiArray extracted");

            // Step 9: Convert MLMultiArray back to Tensor
            let output_tensor = super::conversion::mlmultiarray_to_tensor(
                &device,
                &output_array,
                self.output_shape.clone(),
            )?;
            println!("  ✓ Output tensor created");

            println!("=== Neural Engine inference successful ===");
            Ok(output_tensor)
        } else {
            Err(CoreMLError::ModelLoadError("No MLModel loaded".to_string()))
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Non-macOS: Return dummy output tensor
        println!("Running CoreML inference (non-macOS placeholder)...");
        Tensor::zeros(&device, self.output_shape.clone())
            .map_err(CoreMLError::TensorError)
    }
}
```

### エラーハンドリング
- 各ステップでの詳細なエラーメッセージ
- ml_modelが存在しない場合
- 推論実行失敗
- 出力取得失敗

### デバッグ出力
- 各変換ステップの成功/失敗
- Neural Engine実行時間（オプション）
- 出力形状の検証

---

## 🧪 Phase 7: テストとドキュメント

**工数**: 2-3時間
**ファイル**:
- `tests/coreml_integration_test.rs`
- `claudedocs/coreml_full_integration.md`
- `benchmarks/coreml_neural_engine.rs`

### 統合テスト

#### 1. 基本推論テスト
```rust
#[test]
#[cfg(target_os = "macos")]
fn test_full_prediction_pipeline() {
    // 実際のモデルファイルが必要
    // CI/CDでは skip または モックモデル使用
}
```

#### 2. エラーケーステスト
```rust
#[test]
fn test_invalid_model_path() {
    let result = CoreMLModel::load("nonexistent.mlmodelc");
    assert!(result.is_err());
}

#[test]
fn test_shape_mismatch() {
    // 形状不一致のエラーハンドリング
}
```

### ドキュメント更新

#### claudedocs/coreml_full_integration.md
- 実装の詳細説明
- 使用例
- トラブルシューティング
- パフォーマンス特性

#### README.md更新
```markdown
## CoreML/Neural Engine Support

TensorLogic supports real-time inference on Apple's Neural Engine:

```rust
use tensorlogic::coreml::CoreMLModel;

let model = CoreMLModel::load("model.mlmodelc")?;
let output = model.predict(&input_tensor)?;
```

Optimized for Apple Silicon (M1/M2/M3/M4) Neural Engine.
```

### ベンチマーク

#### benches/coreml_neural_engine.rs
```rust
fn benchmark_neural_engine_vs_metal(c: &mut Criterion) {
    // Neural Engine vs Metal GPU の性能比較
}
```

---

## 📊 工数見積もり詳細

| Phase | タスク | 最小工数 | 最大工数 | 優先度 |
|-------|--------|----------|----------|--------|
| 1 | 事前調査 | 1h | 1h | 完了✅ |
| 2 | Cargo.toml設定 | 0.5h | 1h | 高 |
| 3 | MLModelDescription | 2h | 3h | 高 |
| 4 | MLFeatureValue | 3h | 5h | 高 |
| 5 | MLDictionaryFeatureProvider | 2h | 4h | 高 |
| 6 | 完全prediction() | 3h | 5h | 高 |
| 7 | テスト・ドキュメント | 2h | 3h | 中 |
| **合計** | | **13.5h** | **22h** | |

**推奨実施順序**: Phase 2 → 3 → 4 → 5 → 6 → 7

---

## 🎯 成功基準

### 機能要件
- [ ] 実際のMLModelで推論実行成功
- [ ] input/output名の自動取得
- [ ] 入力形状検証
- [ ] エラーハンドリング完備

### 性能要件
- [ ] Neural Engine利用確認
- [ ] Metal GPUとの性能比較
- [ ] オーバーヘッド < 10%

### 品質要件
- [ ] 全テスト成功（298+ tests）
- [ ] コンパイラ警告0件
- [ ] ドキュメント完備

---

## 🚨 リスクと対策

### リスク1: objc2-core-ml API制約
**影響**: 高
**対策**:
- API調査完了で大部分のリスク軽減済み
- 不明点は実装中に段階的に解決

### リスク2: Neural Engine利用の検証困難
**影響**: 中
**対策**:
- Activity Monitor で ane プロセス確認
- Instruments で Neural Engine 使用率測定
- パフォーマンス比較で間接的に確認

### リスク3: 実装時間の超過
**影響**: 低（オプション実装のため）
**対策**:
- Phase単位で進捗管理
- 各Phase完了後にコミット
- 問題発生時は早期に判断

---

## 📝 実装メモ

### unsafe使用箇所
- objc2のFFI呼び出しはすべてunsafe
- 型変換（transmute）は最小限に
- 各unsafe blockにコメント必須

### メモリ管理
- Retained<T> を使用してARC管理
- 手動でのrelease不要
- ライフタイムに注意

### クロスプラットフォーム
- #[cfg(target_os = "macos")] で条件コンパイル
- 非macOSでもビルド可能を維持
- placeholderは最小限の実装

---

## 🔗 参考資料

- [objc2-core-ml documentation](https://docs.rs/objc2-core-ml/0.2.2/)
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Neural Engine Overview](https://github.com/hollance/neural-engine)
- TensorLogic既存実装:
  - `src/coreml/model.rs`
  - `src/coreml/conversion.rs`
  - `benches/coreml_benchmark.rs`

---

**次のステップ**: Phase 2から順次実装開始
**最終目標**: v1.0リリースでNeural Engine完全対応
