# macOS LLM構築ガイド：MetalとNeural Engineの使い分け

最終更新: 2025年10月

## 目次

1. [はじめに：Metal vs Neural Engine](#はじめにmetal-vs-neural-engine)
2. [重要な誤解の訂正：ANEは推論専用](#重要な誤解の訂正aneは推論専用)
3. [推論でのパフォーマンス比較（2025年最新）](#推論でのパフォーマンス比較2025年最新)
4. [CoreMLの自動最適化の仕組み](#coremlの自動最適化の仕組み)
5. [ANE非対応層への対処戦略](#ane非対応層への対処戦略)
6. [実践的な選択マトリックス](#実践的な選択マトリックス)
7. [推奨ワークフロー](#推奨ワークフロー)
8. [クイックリファレンス](#クイックリファレンス)
9. [モデルの入手方法](#モデルの入手方法)

---

## はじめに：Metal vs Neural Engine

### 現状の実態（2025年）

**重要**: 現在、ほとんどのLLMフレームワーク（llama.cpp、MLXなど）は**Metal経由でGPUを使用**しており、Neural Engine（ANE）は直接サポートしていません。

### それぞれの特徴

#### 🎮 Metal（GPU） - 主要オプション

**メリット:**
- llama.cpp、MLX、PyTorchなど主要フレームワークが対応
- 柔軟性が高く、カスタムモデルやオペレーションに対応
- 開発者が直接制御可能で、デバッグやチューニングが容易

**パフォーマンス例（実測値）:**
- Llama 7B: M3 Maxで30-40 tokens/s（4-bit量子化）
- Llama 3B: M3 Maxで最大50 tokens/s
- Llama 70B: M2 Ultra（192GB RAM）で8-12 tokens/s

**最適な用途:**
- 汎用的なLLM推論
- カスタムモデルや実験的なアーキテクチャ
- 大規模モデル（7B以上）

#### 🧠 Neural Engine（ANE） - 効率重視

**メリット:**
- 電力効率が非常に高い（最大14倍省メモリ、10倍高速化の可能性）
- iPhoneなどモバイルデバイスで特に効率的
- GPU負荷を軽減（GPU並列利用時）

**デメリット:**
- CoreMLフレームワーク経由でのみアクセス可能
- 低レベルAPIが公開されておらず、カスタマイズが困難
- 対応フレームワークが限定的
- 4-bit量子化（LUT4）の品質が低い傾向

**最適な用途:**
- バッテリー駆動デバイスでの推論
- 小〜中規模モデル（〜8B程度）
- Apple公式サポートのあるモデル（Llama 3.1など）

---

## 重要な誤解の訂正：ANEは推論専用

### ANEは学習には使えない

Apple Neural Engine（ANE）は**推論専用**のハードウェアで、**モデル学習には使用できません**。

#### なぜ学習できないのか

**1. 精度の問題**
- ANE: **FP16（16-bit浮動小数点）のみ対応**
- 学習に必要: **FP32（32-bit）** ← backpropagationでの微小な勾配調整に必須

**2. 設計目的**
- ANE: 既存モデルの**実行（推論）**に特化
- 低消費電力・高スループットな推論処理が目標
- 新しいモデルの作成（学習）は想定外

**3. Apple公式の見解**
- Apple Developer ForumsやAI資料で、ANEは学習チェーンに含まれていない
- 開発者の報告: 「学習においてANEはGPUよりあらゆる面で劣る」

### macOSでの学習：Metal（GPU）一択

#### Metal（GPU）の学習性能

| 項目 | 詳細 |
|------|------|
| **対応フレームワーク** | PyTorch MPS, JAX Metal, TensorFlow, **MLX** |
| **Unified Memory** | GPU/CPU/ANEが同じメモリを共有 → 大規模モデル対応 |
| **メモリ帯域** | M4 Max: 546 GB/s（全コンピュートユニットで共有） |
| **消費電力** | 40-80W（高負荷時）← NVIDIAより80%低い |

#### 学習のコード例

**PyTorch MPS（Metal Performance Shaders）**
```python
import torch

# Metal GPUを使用
device = torch.device("mps")
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

# 学習ループ
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**MLX（Apple Silicon専用・最適化）**
```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# MLXはデフォルトでMetal GPU使用
model = YourMLXModel()
optimizer = optim.Adam(learning_rate=1e-3)

# 学習（自動でMetal GPU活用）
def train_step(model, data, target):
    def loss_fn(model):
        return nn.losses.cross_entropy(model(data), target)

    loss, grads = mx.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    return loss
```

#### ハードウェア別の役割

| ハードウェア | 学習 | 推論 |
|-------------|------|------|
| **Neural Engine（ANE）** | ❌ 不可 | ✅ 高効率 |
| **Metal GPU** | ✅ 推奨 | ✅ 高性能 |
| **CPU** | △ 可能だが遅い | △ 最終手段 |

---

## 推論でのパフォーマンス比較（2025年最新）

### 2025年最新ベンチマーク

#### iPhone 17（2025年）の結果

| 処理 | GPU改善 | ANE改善 |
|------|---------|---------|
| 一般的なAI処理 | **3.1倍高速化** | 1.15倍高速化 |

**注目**: 特定モデル（Nvidia Parakeet v3 0.6B）では**ANEがGPU比4.3倍速い**

#### M5チップ（2025年10月発表）

- GPU内に**Neural Accelerator統合**
- AI性能: M4比4倍、M1比6倍
- ANEとGPUの境界が曖昧に → **ハイブリッド時代到来**

### 実測パフォーマンス比較

#### ANEが圧勝するケース

| モデル | デバイス | ANE | GPU | ANE優位性 |
|--------|---------|-----|-----|-----------|
| MobileNetV2 | M1 | - | - | **3.5倍速い** |
| 行列演算 | M2 | 217ms | 1316ms | **4倍速い** |
| DistilBERT | iPhone 13 | 3.47ms @ 0.454W | - | **超低消費電力** |

#### GPUが優位なケース

- **大規模LLM**（7B以上）: Metalフレームワークが充実
- **カスタムモデル**: 直接プログラム可能
- **柔軟性**: llama.cpp、MLXなど豊富なツール

### 電力効率：決定的な違い

| 実行方法 | 消費電力 | バッテリー影響 |
|---------|---------|---------------|
| **CoreML（ANE使用）** | 0.454W | 最小 |
| **Metal直接（GPU）** | 高 | **2倍程度** |
| Bender（GPU最適化） | 最高 | 最大（ただしFPS 2倍） |

**重要**: CoreMLの`.all`オプション（ANE/GPU/CPU自動選択）が最もバランス良い

### 実用的なパフォーマンス例

#### CoreML + ANE（iPhone 13）
```
DistilBERT: 3.47ms @ 0.454W
→ バッテリー駆動で数千回推論可能
```

#### Metal + GPU（M3 Max）
```
Llama 3B (4-bit): 50 tokens/s
Llama 7B (4-bit): 30-40 tokens/s
→ 実用的なチャット速度
```

#### CoreML Hybrid（M1 Max）
```
Llama 3.1-8B: 33 tokens/s
→ ANE/GPUの自動切り替え
```

---

## CoreMLの自動最適化の仕組み

### computeUnits = .all の動作

```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // ← これがデフォルト

// CoreMLが自動で:
// ├─ ANE向きの層 → ANE
// ├─ GPU向きの層 → GPU
// └─ CPU向きの層 → CPU
```

### CoreMLがやってくれること

- **動的グラフ分割**: ネットワークグラフをレイヤー単位で分析
- **最適エンジン選択**: 各レイヤーに最適なハードウェアを自動選択
- **シームレスな実行**: 開発者はハードウェアを意識不要
- **パフォーマンス最大化**: 速度・メモリ・電力を自動最適化

### 重要な制約

#### 1. ANEは「使えれば使う」（保証なし）

```
.allを指定 ≠ ANEが必ず使われる

実際の動作:
├─ ANEサポート層 → ANEで実行
├─ ANE非サポート層 → GPU/CPUに自動フォールバック
└─ デバイスにANEなし → GPU/CPUのみ
```

#### 2. Float32モデルはANE使えない

```
モデルの精度    ANE使用
├─ Float32  → ❌ GPUとCPUのみ
├─ Float16  → ✅ ANE使用可能
└─ INT8     → ✅ ANE使用可能
```

**重要**: Float32モデルは自動的にANEから除外される

#### 3. レイヤー単位の切り替え

```
モデル全体 ≠ 単一ハードウェアで実行

例: Transformer推論
├─ Embedding層    → ANE
├─ Attention層    → GPU（ANE非対応オペレーション）
├─ Feed Forward   → ANE
├─ Softmax       → GPU
└─ Output層       → ANE
```

**データ転送オーバーヘッド**: 切り替えが多いと遅くなる可能性

### computeUnits オプション完全ガイド

| オプション | 使用ハードウェア | 用途 |
|-----------|-----------------|------|
| `.all` | **ANE + GPU + CPU** | 推奨・デフォルト |
| `.cpuAndGPU` | GPU + CPU（ANE除外） | GPU専用ワークロード |
| `.cpuAndNeuralEngine` | ANE + CPU（GPU除外） | GPUを他で使う場合 |
| `.cpuOnly` | CPUのみ | デバッグ・比較用 |

### 実行結果の確認方法

#### パフォーマンスレポート（Xcode Instruments）

```swift
// Xcodeでモデル実行後:
// Instruments → Core ML → Performance Report

レイヤー情報:
├─ レイヤー名: conv1
├─ タイプ: Convolution
├─ ✓ ANE (実際に実行)
├─ ○ GPU (サポートされるが未使用)
└─ ○ CPU (サポートされるが未使用)

✓ = 実際に実行された
○ = サポートされるが選ばれなかった
```

### モデル変換時の最適化

```python
import coremltools as ct

# Float16に変換してANE対応
model = ct.convert(
    pytorch_model,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16  # ← ANE有効化
)
```

---

## ANE非対応層への対処戦略

ANE非対応の計算が多い場合、フォールバックのオーバーヘッドを避けるための戦略があります。

### アプローチ1: MLX/Metal直接使用（最大制御）

**完全なGPU制御、ANEフォールバックなし**

```python
# MLXはANEを一切使わず、GPU直接制御
import mlx.core as mx
import mlx.nn as nn

# すべての演算がGPU上で実行
# ANEフォールバックの心配なし
model = YourModel()
output = model(input)  # → 100% GPU、転送オーバーヘッドゼロ
```

**メリット:**
- ANE非対応による予期しないフォールバックがない
- すべての演算がGPU上で完結
- Unified Memoryでデータ転送オーバーヘッドゼロ
- カスタムMetal kernelも書ける

**デメリット:**
- ANEの省電力性を活用できない
- CoreMLエコシステムの外

**最適な用途:**
- 開発・実験
- カスタムモデル
- パフォーマンス予測可能性重視

### アプローチ2: CoreML モデル分割（中程度の制御）

**ANE向きとGPU向きのモデルを手動分割**

```swift
// アプローチ: モデルを2つに分割

// パート1: ANE最適化された部分
let config1 = MLModelConfiguration()
config1.computeUnits = .cpuAndNeuralEngine
let model1 = try Model1(configuration: config1)

// パート2: ANE非対応が多い部分
let config2 = MLModelConfiguration()
config2.computeUnits = .cpuAndGPU  // ANE除外
let model2 = try Model2(configuration: config2)

// 実行時に接続
let intermediate = try model1.prediction(input: input)
let output = try model2.prediction(input: intermediate)
```

**実装方法:**
```python
# モデル変換時に分割
import coremltools as ct

# 元のモデルを分析して分割ポイントを特定
# 例: Transformer → Embedding + Attention + FFN
model_part1 = ct.convert(embedding_layers, ...)  # ANE向き
model_part2 = ct.convert(attention_layers, ...)  # GPU向き
model_part3 = ct.convert(ffn_layers, ...)        # ANE向き
```

**メリット:**
- ANE/GPUを意図的に使い分け可能
- 各パートの最適化を細かく制御
- フォールバックのオーバーヘッド削減

**デメリット:**
- モデル分割の手間
- パート間のデータ転送コスト
- エンドツーエンド最適化の喪失

### アプローチ3: モデル手術（ANE対応への変換）

**ANE非対応レイヤーを対応版に置き換え**

```python
# mlmodelファイルを編集してANE非対応層を置き換え
import coremltools as ct

model = ct.models.MLModel("model.mlmodel")

# 例: GELUをSiLUに置き換え（ANE対応）
# 例: LayerNormの特定実装をANE対応版に変更
# 例: カスタム演算を標準演算の組み合わせに分解
```

**公式ドキュメントより:**
> "If your model has just one or two layers that are unsupported by the ANE, it's smart to edit the mlmodel file and replace those layers with an alternative that does work on the ANE"

**メリット:**
- モデル全体をANEで実行可能に
- 大幅な速度・省電力化

**デメリット:**
- 技術的に高度
- 数学的等価性の検証必要
- すべてのケースで可能とは限らない

### アプローチ4: ComputeUnits設定（粗い制御）

**モデル全体のハードウェア選択**

```swift
let config = MLModelConfiguration()

// 選択肢:
config.computeUnits = .all              // デフォルト（ANE+GPU+CPU）
config.computeUnits = .cpuAndGPU        // ANE除外、GPU優先
config.computeUnits = .cpuAndNeuralEngine  // GPU除外、ANE優先
```

**制御可能な範囲:**
- ❌ レイヤー単位の指定: **不可能**（公開APIなし）
- ✅ モデル全体の方針: **可能**
- ✅ ハードウェアの除外: **可能**

**実際の挙動:**
```
.cpuAndGPU を指定した場合:
├─ ANE完全除外
├─ すべてGPU/CPUで実行
└─ フォールバックオーバーヘッドなし

ただし:
└─ ANEの省電力性を完全に失う
```

### 制御レベル比較

| アプローチ | 制御レベル | 実装難易度 | 電力効率 | 柔軟性 |
|-----------|-----------|-----------|---------|--------|
| **MLX/Metal直接** | ⭐⭐⭐⭐⭐ 完全 | ⭐⭐ 簡単 | ⭐⭐ 低 | ⭐⭐⭐⭐⭐ 最高 |
| **モデル分割** | ⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中程度 | ⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中 |
| **モデル手術** | ⭐⭐⭐ 中 | ⭐⭐⭐⭐⭐ 高度 | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐ 低 |
| **computeUnits設定** | ⭐⭐ 粗い | ⭐ 非常に簡単 | ⭐⭐⭐ 中 | ⭐⭐⭐ 中 |

---

## 実践的な選択マトリックス

### 用途別推奨戦略

| 状況 | 推奨アプローチ | 理由 |
|------|---------------|------|
| **開発・実験** | **MLX直接使用** | 柔軟性最大、デバッグ容易 |
| **ANE非対応層が少数** | **モデル手術** | ANE完全活用で最速 |
| **ANE非対応層が多い** | **モデル分割 or GPU専用** | フォールバック削減 |
| **バッテリー最優先** | **モデル手術 → ANE最適化** | 省電力性確保 |
| **パフォーマンス最優先** | **MLX/Metal直接 or GPU専用** | 予測可能な速度 |
| **他でGPU使用中** | **.cpuAndNeuralEngine** | リソース競合回避 |

### Transformer LLMの具体例

#### 問題分析

```
Transformerの典型的なANE対応状況:
✅ Token Embedding      → ANE対応
❌ Scaled Dot Product   → ANE非対応（カスタム演算）
✅ Linear (Feed Forward) → ANE対応
❌ GELU activation      → ANE非対応（近似必要）
✅ LayerNorm           → ANE対応（実装次第）
❌ Softmax（大規模）    → ANE非対応
```

#### 戦略オプション

**オプションA: MLX使用（シンプル）**
```python
# すべてGPUで実行、ANE考慮不要
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.1-8B-4bit")
response = generate(model, tokenizer, prompt)
# → すべてMetal GPU、転送なし、予測可能
```

**オプションB: CoreML 3分割（最適化）**
```python
# 変換時に分割
part1_embeddings = convert_to_coreml(embeddings)     # ANE
part2_attention = convert_to_coreml(attention)       # GPU
part3_ffn = convert_to_coreml(feed_forward)         # ANE

# Swift側で接続
let emb = try part1.prediction(tokens)
let att = try part2.prediction(emb)  // config: .cpuAndGPU
let out = try part3.prediction(att)
```

**オプションC: CoreML GPU専用（バランス）**
```swift
// シンプルだが省電力性は犠牲
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU  // ANE完全除外

let model = try LlamaModel(configuration: config)
// → すべてGPU、フォールバックなし
```

### ANE→GPU切り替えコスト

```
CoreML内部での切り替え:
ANE → GPU: 高コスト（メモリコピー）
ANE → CPU: 低コスト
CPU → ANE: 低コスト

理由:
└─ Unified Memoryだが、ANE/GPUは異なるメモリレイアウト
```

---

## 推奨ワークフロー

### ステップ1: プロファイリング

```bash
# CoreMLモデルをInstrumentsで実行
# → 各レイヤーがどこで実行されたか確認

結果例:
├─ ANE: 60%の層
├─ GPU: 30%の層（フォールバック）
└─ CPU: 10%の層

問題: GPU切り替え頻度高い → オーバーヘッド大
```

### ステップ2: 戦略決定

```
GPU切り替え < 20% → そのまま .all 使用
GPU切り替え 20-50% → モデル分割検討
GPU切り替え > 50% → MLX or .cpuAndGPU
```

### ステップ3: 実装

```python
# パターンA: MLX（開発推奨）
import mlx.core as mx
# → シンプル、フォールバック心配なし

# パターンB: CoreML最適化（デプロイ用）
# → バッテリー駆動で有利
```

### ステップ4: 検証

```swift
// 実行時間測定
let start = Date()
let prediction = try model.prediction(input: input)
let elapsed = Date().timeIntervalSince(start)

print("推論時間: \(elapsed * 1000)ms")
// → Instrumentsで各ハードウェアの使用率確認
```

---

## クイックリファレンス

### 状況別推奨一覧

| あなたの状況 | 最適選択 | 理由 |
|-------------|---------|------|
| **バッテリー駆動** | **ANE（CoreML）** | 電力効率2倍以上 |
| **AC電源Mac** | **GPU（Metal/MLX）** | 最大スループット |
| **iPhone実装** | **ANE（CoreML）** | バッテリー寿命最優先 |
| **大規模LLM（>7B）** | **GPU（Metal）** | フレームワーク充実 |
| **小型モデル（<1B）** | **ANE（CoreML）** | 3-4倍高速 |
| **開発・実験** | **GPU（Metal）** | 柔軟性・デバッグ容易 |
| **プロダクション** | **Hybrid（.all）** | 自動最適化 |

### モデル最適化チェックリスト

```
□ モデルをFloat16/INT8に量子化（ANE対応）
□ ANE非対応層を確認（Instrumentsで）
□ 必要に応じて戦略選択（MLX / 分割 / 手術 / GPU専用）
□ パフォーマンステスト実施
□ 電力消費測定（バッテリー駆動の場合）
```

### コードスニペット集

#### MLXでのLLM推論
```python
import mlx.core as mx
from mlx_lm import load, generate

# モデルロード
model, tokenizer = load("mlx-community/Llama-3.1-8B-4bit")

# 推論実行
response = generate(
    model,
    tokenizer,
    prompt="こんにちは",
    max_tokens=100
)
```

#### CoreML最適化変換
```python
import coremltools as ct

# PyTorchモデルをCoreMLに変換
model = ct.convert(
    pytorch_model,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,  # ANE有効化
    minimum_deployment_target=ct.target.iOS17
)

model.save("optimized_model.mlpackage")
```

#### CoreML実行（Swift）
```swift
import CoreML

// モデルロード
let config = MLModelConfiguration()
config.computeUnits = .all  // ANE+GPU+CPU自動選択
let model = try YourModel(configuration: config)

// 推論実行
let input = YourModelInput(data: inputData)
let output = try model.prediction(input: input)
```

#### パフォーマンス測定
```swift
import Foundation

func measurePerformance(iterations: Int = 100) {
    var times: [Double] = []

    for _ in 0..<iterations {
        let start = Date()
        let _ = try? model.prediction(input: input)
        let elapsed = Date().timeIntervalSince(start)
        times.append(elapsed)
    }

    let avg = times.reduce(0, +) / Double(times.count)
    print("平均推論時間: \(avg * 1000)ms")
}
```

### 意思決定フローチャート

```
LLM推論を実装したい
│
├─ 学習が必要？
│  └─ YES → Metal（GPU）一択
│            ├─ PyTorch MPS
│            └─ MLX
│
└─ 推論のみ
   │
   ├─ バッテリー駆動？
   │  └─ YES → CoreML + ANE
   │            └─ Float16に変換必須
   │
   ├─ 大規模モデル（>7B）？
   │  └─ YES → MLX/Metal
   │            └─ llama.cpp, MLX
   │
   ├─ カスタムモデル？
   │  └─ YES → MLX/Metal直接
   │            └─ 柔軟性最大
   │
   ├─ Apple公式モデル？
   │  └─ YES → CoreML Hybrid
   │            └─ .all で自動最適化
   │
   └─ ANE非対応層が多い？
      ├─ YES → MLX or .cpuAndGPU
      └─ NO  → CoreML .all（推奨）
```

### 重要な注意点まとめ

1. **ANEは推論専用** - 学習には使えない
2. **Float16必須** - Float32モデルはANE使用不可
3. **レイヤー単位制御不可** - CoreMLは自動判断のみ
4. **モデル変換が鍵** - 適切な最適化で性能が大きく変わる
5. **バッテリー vs 速度** - 用途に応じた選択が重要
6. **2025年はハイブリッド** - M5世代は自動最適化が進化

---

## モデルの入手方法

### Llama 3.1-8B CoreML版

#### ❌ Apple公式の事前変換版は配布されていません

Appleは変換方法のガイドを公開していますが、事前に変換されたモデルファイルは配布していません。

#### ✅ 入手オプション

##### オプション1: コミュニティ変換版（すぐ使える）

Hugging Faceにコミュニティが作成した変換版があります：

```bash
# Hugging Faceからダウンロード
git lfs install
git clone https://huggingface.co/andmev/Llama-3.1-8B-Instruct-CoreML
```

**リンク**: [andmev/Llama-3.1-8B-Instruct-CoreML](https://huggingface.co/andmev/Llama-3.1-8B-Instruct-CoreML)

**特徴**:
- llama-to-coremlプロジェクトを使って変換済み
- すぐに使用可能
- ただしコミュニティ版（公式ではない）

**その他のCoreML変換版**:
- [smpanaro/Llama-3.2-1B-Instruct-CoreML](https://huggingface.co/smpanaro/Llama-3.2-1B-Instruct-CoreML) - Llama 3.2 1B版
- [andmev/Llama-3.2-3B-Instruct-CoreML](https://huggingface.co/andmev/Llama-3.2-3B-Instruct-CoreML) - Llama 3.2 3B版

##### オプション2: 自分で変換（Apple公式ガイド準拠）

Apple Machine Learning Researchの公式ガイド（2024年11月公開）に従って変換：

**公式ガイド**: [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)

**変換手順**:
```bash
# 1. 必要なツールのインストール
pip install coremltools huggingface_hub

# 2. 元のLlamaモデルをHugging Faceから取得
# （Meta公式のLlama 3.1-8B-Instruct）
huggingface-cli login  # Metaのライセンス同意が必要
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# 3. CoreMLに変換（Apple公式のサンプルコード使用）
# 公式ガイドに詳細な変換スクリプトとコードあり
```

**Apple公式の最適化内容**:
- **Int4量子化**（ブロックサイズ32）でメモリフットプリント削減
- **Stateful KVキャッシュ**でデコード時の計算再利用
- **Float16精度**でANE対応
- M1 Maxで約33 tokens/sの性能を実現

**変換スクリプト例** (Apple公式ガイドより):
```python
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルロード
model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CoreMLに変換（詳細は公式ガイド参照）
coreml_model = ct.convert(
    model,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    # Int4量子化、KVキャッシュなどの最適化オプション
    # 詳細は公式ガイドを参照
)

coreml_model.save("Llama-3.1-8B-Instruct.mlpackage")
```

##### オプション3: MLX版を使う（推奨・より簡単）

CoreMLにこだわらないなら、MLXの方が簡単で実用的：

```bash
# MLX LMのインストール
pip install mlx-lm

# Llama 3.1-8Bのダウンロードと実行（ワンライン）
mlx_lm.generate \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --prompt "こんにちは"
```

**または、Pythonスクリプトで**:
```python
from mlx_lm import load, generate

# モデルロード（初回は自動ダウンロード）
model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

# 推論実行
response = generate(
    model,
    tokenizer,
    prompt="こんにちは、LLMについて教えてください",
    max_tokens=200
)
print(response)
```

**MLX版のメリット**:
- インストールと実行が非常に簡単
- Metal GPU直接使用（高性能）
- Hugging Faceに多数の量子化版あり
- llama.cppとの互換性も高い

**Hugging Faceで利用可能なMLX版**:
- [mlx-community/Meta-Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit)
- [mlx-community/Meta-Llama-3.1-8B-Instruct-8bit](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit)
- [mlx-community/Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit)

### 推奨される選択

| 目的 | 推奨 | 理由 |
|------|------|------|
| **すぐ試したい** | **MLX版** | 最も簡単、高性能 |
| **開発・実験** | **MLX版** | 柔軟性が高い、デバッグ容易 |
| **iPhoneアプリ開発** | **CoreML自分で変換** | 公式ガイド準拠が安心 |
| **CoreML学習目的** | **コミュニティ版 + 自分で変換** | 参考として確認後、実践 |
| **プロダクション（Mac）** | **MLX版** | パフォーマンス安定 |
| **プロダクション（iOS）** | **CoreML自分で変換** | バッテリー効率最優先 |

### ダウンロードサイズの目安

| モデル | サイズ | 必要メモリ |
|--------|--------|-----------|
| Llama 3.1-8B (FP16) | ~16GB | 18-20GB |
| Llama 3.1-8B (Int4) | ~4-5GB | 6-8GB |
| Llama 3.2-3B (Int4) | ~2GB | 4GB |
| Llama 3.2-1B (Int4) | ~1GB | 2GB |

### その他の有用なモデル

#### Apple公式モデル
Appleの[Hugging Face Organization](https://huggingface.co/apple)では以下を公開：
- [apple/coreml-stable-diffusion-2-base](https://huggingface.co/apple/coreml-stable-diffusion-2-base) - Stable Diffusion
- OpenELM、MobileCLIPなどApple独自モデル

**注意**: AppleはLlamaの公式CoreML版は配布していません。

#### その他のLLM CoreML版
- [TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML](https://huggingface.co/TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML)
- [coreml-community](https://huggingface.co/coreml-community) - コミュニティモデル集

---

## 参考リンク

### 公式ドキュメント
- [Apple Machine Learning Research](https://machinelearning.apple.com/research)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

### 有用なツール
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Metal対応LLM推論
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon最適化フレームワーク
- [coremltools](https://github.com/apple/coremltools) - CoreML変換ツール

### コミュニティリソース
- [Neural Engine Documentation](https://github.com/hollance/neural-engine)
- [Apple Developer Forums](https://developer.apple.com/forums/)

---

**最終更新**: 2025年10月
**対象デバイス**: M1/M2/M3/M4/M5, iPhone 13以降
**対象OS**: macOS 14.0+, iOS 17.0+
