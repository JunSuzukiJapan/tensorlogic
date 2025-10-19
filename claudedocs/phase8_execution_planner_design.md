# Phase 8.2: ExecutionPlanner 設計書

## 概要

ExecutionPlannerは、演算ごとに最適なデバイス（Metal GPU、Neural Engine、CPU）を自動的に選択するシステムです。

## 設計目標

1. **自動最適化**: ユーザーが明示的にデバイスを指定しなくても、最適なデバイスで演算を実行
2. **ヒューリスティック**: 演算の種類、テンソルサイズ、デバイス特性に基づいた選択
3. **ベンチマーク対応**: 実測に基づいた動的な選択（オプション）
4. **キャッシング**: 同じ条件での選択結果を再利用
5. **拡張性**: 新しいデバイスや演算の追加が容易

## アーキテクチャ

### コンポーネント

```
┌─────────────────────────────────────┐
│   ExecutionPlanner (Singleton)      │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  DeviceSelector              │  │
│  │  - Heuristic rules           │  │
│  │  - Operation characteristics │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  BenchmarkCache              │  │
│  │  - Performance measurements  │  │
│  │  - Size thresholds           │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  DecisionCache               │  │
│  │  - Operation + size → device │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### データ構造

```rust
pub struct ExecutionPlanner {
    /// Device selection strategy
    strategy: SelectionStrategy,

    /// Performance benchmark cache
    benchmark_cache: HashMap<OperationKey, DevicePerformance>,

    /// Decision cache (operation + size → device)
    decision_cache: HashMap<DecisionKey, Device>,

    /// Available devices
    metal_device: Option<MetalDevice>,
    neural_engine_available: bool,
}

pub enum SelectionStrategy {
    /// Use heuristic rules only
    Heuristic,

    /// Use benchmarks when available, fallback to heuristic
    Adaptive,

    /// Always use specified device
    Fixed(Device),
}

pub struct OperationKey {
    operation: OperationType,
    input_shapes: Vec<Vec<usize>>,
}

pub struct DevicePerformance {
    device: Device,
    avg_time_ms: f64,
    std_dev_ms: f64,
}

pub struct DecisionKey {
    operation: OperationType,
    total_elements: usize,
}

pub enum OperationType {
    Add,
    Mul,
    MatMul,
    Conv2D,
    ReLU,
    Softmax,
    // ...
}
```

## デバイス選択ヒューリスティック

### 1. Neural Engine優先演算

大規模な行列演算はNeural Engineが得意:
- **MatMul**: サイズ > 64x64 → Neural Engine
- **Conv2D**: 常に Neural Engine
- **Large FC layers**: M*N*K > 100,000 → Neural Engine

### 2. Metal GPU優先演算

並列処理可能な要素ごと演算はMetal GPUが得意:
- **Element-wise ops** (add, mul, relu, etc.): サイズ > 1000 → Metal
- **Reduction ops**: サイズ > 10,000 → Metal
- **Activation functions**: サイズ > 1000 → Metal

### 3. CPU フォールバック

小規模演算はCPUの方が効率的（オーバーヘッド回避）:
- **Small tensors**: サイズ < 1000 → CPU
- **Scalar operations**: 常にCPU
- **Control flow**: 常にCPU

### サイズ閾値の例

```rust
const METAL_MIN_SIZE: usize = 1000;
const NEURAL_ENGINE_MATMUL_MIN: usize = 64 * 64;
const REDUCTION_GPU_MIN: usize = 10000;
```

## API設計

### 基本的な使い方

```rust
// シングルトンアクセス
let planner = ExecutionPlanner::global();

// 演算に最適なデバイスを取得
let device = planner.select_device_for_add(&tensor_a, &tensor_b);

// または、より汎用的に
let device = planner.select_device(
    OperationType::Add,
    &[tensor_a.shape().dims(), tensor_b.shape().dims()],
);
```

### 設定API

```rust
// 戦略を変更
ExecutionPlanner::set_strategy(SelectionStrategy::Heuristic);

// 固定デバイスを使用
ExecutionPlanner::set_strategy(SelectionStrategy::Fixed(Device::Metal(device)));

// キャッシュをクリア
ExecutionPlanner::clear_cache();
```

### ベンチマーク機能（オプション）

```rust
// 特定演算のベンチマークを実行
planner.benchmark_operation(
    OperationType::MatMul,
    vec![vec![128, 128], vec![128, 128]],
);

// すべての主要演算をベンチマーク
planner.benchmark_all_operations();
```

## 実装フェーズ

### Phase 8.2.1: 基本実装
- [x] ExecutionPlanner構造体とシングルトン
- [x] ヒューリスティックベースのデバイス選択
- [x] 基本的なデシジョンキャッシュ
- [x] 主要演算への統合（add, mul, matmul）

### Phase 8.2.2: 拡張機能
- [ ] ベンチマーク機能
- [ ] アダプティブ戦略
- [ ] 詳細な統計情報
- [ ] パフォーマンスプロファイリング

## テスト戦略

### 単体テスト
- デバイス選択ロジックの検証
- サイズ閾値の動作確認
- キャッシュの動作確認

### 統合テスト
- 実際の演算での自動デバイス選択
- 異なるサイズでの動作確認
- パフォーマンス比較

### ベンチマークテスト
- Metal vs CPU の性能比較
- Neural Engine vs Metal の性能比較
- 閾値の妥当性検証

## パフォーマンス目標

- **小規模演算** (<1000要素): CPUで実行（GPUオーバーヘッド回避）
- **中規模演算** (1000-100,000要素): Metalで実行（並列化の恩恵）
- **大規模行列演算** (>100,000要素): Neural Engineで実行（専用ハードウェア活用）

## 将来の拡張

1. **機械学習ベースの選択**: 過去の実行履歴から学習
2. **マルチデバイス実行**: 大規模演算を複数デバイスで分散
3. **バッテリー状態考慮**: モバイルデバイスでの省電力モード
4. **メモリ圧考慮**: 利用可能メモリに基づいた選択
