# Phase 9.1: Optimizer実装設計

## 概要

AI学習に必須のOptimizer（最適化アルゴリズム）を実装し、実用的なニューラルネットワーク学習を可能にする。

## 設計目標

1. **基本Optimizer**: SGD, Momentum, Adam, AdamW
2. **統一API**: 共通のOptimizerトレイトで拡張性を確保
3. **効率的な実装**: GPU対応、in-place更新
4. **状態管理**: モメンタム、二次モーメント等の内部状態
5. **学習率制御**: 基本的な学習率設定とステップ管理

## 数学的基礎

### SGD (Stochastic Gradient Descent)

最もシンプルな最適化手法：

```
θ_{t+1} = θ_t - η ∇L(θ_t)
```

- θ: パラメータ
- η: 学習率 (learning rate)
- ∇L: 損失関数の勾配

### SGD with Momentum

勾配の移動平均を使用して収束を加速：

```
v_{t+1} = μ v_t + ∇L(θ_t)
θ_{t+1} = θ_t - η v_{t+1}
```

- v: モメンタム（速度）
- μ: モメンタム係数（通常0.9）

### Adam (Adaptive Moment Estimation)

一次・二次モーメントの移動平均を使用：

```
m_{t+1} = β₁ m_t + (1-β₁) ∇L(θ_t)           # 一次モーメント
v_{t+1} = β₂ v_t + (1-β₂) [∇L(θ_t)]²        # 二次モーメント
m̂_{t+1} = m_{t+1} / (1 - β₁^{t+1})          # バイアス補正
v̂_{t+1} = v_{t+1} / (1 - β₂^{t+1})          # バイアス補正
θ_{t+1} = θ_t - η m̂_{t+1} / (√v̂_{t+1} + ε)
```

- β₁: 一次モーメント減衰率（通常0.9）
- β₂: 二次モーメント減衰率（通常0.999）
- ε: 数値安定化定数（通常1e-8）

### AdamW (Adam with Weight Decay)

Adamに正則化を正しく適用：

```
m_{t+1} = β₁ m_t + (1-β₁) ∇L(θ_t)
v_{t+1} = β₂ v_t + (1-β₂) [∇L(θ_t)]²
m̂_{t+1} = m_{t+1} / (1 - β₁^{t+1})
v̂_{t+1} = v_{t+1} / (1 - β₂^{t+1})
θ_{t+1} = θ_t - η [m̂_{t+1} / (√v̂_{t+1} + ε) + λ θ_t]  # Weight decay追加
```

- λ: weight decay係数（通常0.01）

## アーキテクチャ

```
┌─────────────────────────────────────────┐
│         Optimizer Trait                 │
│  - step()                               │
│  - zero_grad()                          │
│  - state_dict()                         │
│  - load_state_dict()                    │
└─────────────────────────────────────────┘
                    ▲
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────┴────────┐    ┌────────┴────────┐
│  SGD           │    │  Adam           │
│  - lr          │    │  - lr           │
│  - momentum    │    │  - betas        │
│  - dampening   │    │  - eps          │
│  - weight_decay│    │  - weight_decay │
└────────────────┘    │  - amsgrad      │
                      └─────────────────┘
        │                       │
┌───────┴────────┐    ┌────────┴────────┐
│  Momentum      │    │  AdamW          │
│  (SGD variant) │    │  (Adam variant) │
└────────────────┘    └─────────────────┘
```

## データ構造

### Optimizer Trait

```rust
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self) -> TensorResult<()>;

    /// Zero out all gradients
    fn zero_grad(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);

    /// Get optimizer state dictionary for saving/loading
    fn state_dict(&self) -> OptimizerState;

    /// Load optimizer state from dictionary
    fn load_state_dict(&mut self, state: OptimizerState) -> TensorResult<()>;

    /// Add parameter group to optimizer
    fn add_param_group(&mut self, params: Vec<Tensor>);
}
```

### Parameter Group

```rust
pub struct ParamGroup {
    /// Parameters to optimize
    pub params: Vec<Tensor>,

    /// Learning rate for this group
    pub lr: f32,

    /// Weight decay for this group (L2 regularization)
    pub weight_decay: f32,

    /// Custom options per parameter group
    pub options: HashMap<String, f32>,
}
```

### SGD Implementation

```rust
pub struct SGD {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,

    /// Momentum coefficient (0 for no momentum)
    momentum: f32,

    /// Dampening for momentum
    dampening: f32,

    /// Nesterov momentum flag
    nesterov: bool,

    /// Velocity buffers for momentum (one per parameter)
    velocity_buffers: HashMap<usize, Tensor>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_options(params, lr, 0.0, 0.0, false, 0.0)
    }

    pub fn with_momentum(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        Self::with_options(params, lr, momentum, 0.0, false, 0.0)
    }

    pub fn with_options(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
        weight_decay: f32,
    ) -> Self;
}
```

### Adam Implementation

```rust
pub struct Adam {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,

    /// Coefficients for computing running averages
    betas: (f32, f32),  // (beta1, beta2)

    /// Term added to denominator for numerical stability
    eps: f32,

    /// Weight decay (L2 penalty)
    weight_decay: f32,

    /// Whether to use AMSGrad variant
    amsgrad: bool,

    /// First moment estimates (m_t)
    exp_avg: HashMap<usize, Tensor>,

    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<usize, Tensor>,

    /// Maximum second moment for AMSGrad
    max_exp_avg_sq: HashMap<usize, Tensor>,

    /// Step counter
    step_count: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-8, 0.0, false)
    }

    pub fn with_options(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self;
}
```

### AdamW Implementation

```rust
pub struct AdamW {
    /// Same structure as Adam
    inner: Adam,

    /// AdamW applies weight decay differently
    /// (directly to parameters, not to gradients)
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-8, 0.01)
    }

    pub fn with_options(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self;
}
```

## 実装戦略

### Phase 9.1.1: 基盤実装

1. **Optimizer Trait定義**
   - 共通インターフェース
   - ParamGroup構造体
   - OptimizerState (保存/読み込み用)

2. **SGD実装**
   - 基本SGD（モメンタムなし）
   - モメンタムバッファ管理
   - Weight decay対応

3. **テスト**
   - 単純な二次関数の最適化
   - 勾配降下の収束テスト

### Phase 9.1.2: Momentum対応

1. **SGD with Momentum**
   - 速度バッファの追加
   - モメンタム更新式
   - Nesterov momentum対応

2. **テスト**
   - モメンタムの収束速度検証
   - Nesterovとの比較

### Phase 9.1.3: Adam実装

1. **Adam Optimizer**
   - 一次・二次モーメント管理
   - バイアス補正
   - AMSGrad対応

2. **テスト**
   - 各種最適化問題でのテスト
   - PyTorchとの結果比較

### Phase 9.1.4: AdamW実装

1. **AdamW Optimizer**
   - Weight decayの正しい適用
   - Adamとの違いを明確化

2. **統合テスト**
   - 小規模NNの学習テスト
   - 過学習抑制の検証

## 使用例

### 基本的な使用方法

```rust
use tensorlogic::optim::{SGD, Adam, AdamW};
use tensorlogic::tensor::Tensor;

// モデルパラメータ
let mut w = Tensor::randn(vec![10, 5])?;
let mut b = Tensor::randn(vec![5])?;
w.set_requires_grad(true);
b.set_requires_grad(true);

// SGD optimizer
let mut optimizer = SGD::new(vec![w.clone(), b.clone()], 0.01);

// 学習ループ
for epoch in 0..100 {
    // Forward pass
    let output = model_forward(&w, &b, &input)?;
    let loss = compute_loss(&output, &target)?;

    // Backward pass
    loss.backward()?;

    // Update parameters
    optimizer.step()?;
    optimizer.zero_grad();
}
```

### Adam optimizer

```rust
// Adam with default parameters
let mut optimizer = Adam::new(vec![w.clone(), b.clone()], 0.001);

// Adam with custom parameters
let mut optimizer = Adam::with_options(
    vec![w.clone(), b.clone()],
    0.001,           // learning rate
    (0.9, 0.999),    // betas
    1e-8,            // epsilon
    0.0,             // weight decay
    false,           // amsgrad
);
```

### AdamW optimizer (推奨)

```rust
// AdamW with weight decay
let mut optimizer = AdamW::with_options(
    vec![w.clone(), b.clone()],
    0.001,           // learning rate
    (0.9, 0.999),    // betas
    1e-8,            // epsilon
    0.01,            // weight decay (正則化)
);

for epoch in 0..epochs {
    let loss = train_step(&mut model, &optimizer, &data)?;
    optimizer.step()?;
    optimizer.zero_grad();
}
```

### Parameter groups (異なる学習率)

```rust
// Backbone用とHead用で異なる学習率
let backbone_params = model.backbone_parameters();
let head_params = model.head_parameters();

let mut optimizer = Adam::new(vec![], 0.0);  // Empty initialization
optimizer.add_param_group(ParamGroup {
    params: backbone_params,
    lr: 0.0001,  // Backboneは低い学習率
    weight_decay: 0.01,
    options: HashMap::new(),
});
optimizer.add_param_group(ParamGroup {
    params: head_params,
    lr: 0.001,   // Headは高い学習率
    weight_decay: 0.01,
    options: HashMap::new(),
});
```

### State保存/読み込み

```rust
// Optimizer状態を保存
let state = optimizer.state_dict();
save_to_file("optimizer_state.bin", &state)?;

// 学習再開時に読み込み
let state = load_from_file("optimizer_state.bin")?;
optimizer.load_state_dict(state)?;
```

## テスト戦略

### 単体テスト

1. **SGD Tests**
   - 基本的な勾配降下
   - モメンタムの動作確認
   - Weight decayの効果

2. **Adam Tests**
   - バイアス補正の検証
   - 二次モーメントの計算
   - AMSGradの動作

3. **AdamW Tests**
   - Adamとの違い
   - Weight decayの適用方法

### 統合テスト

1. **収束テスト**
   - 二次関数の最小化
   - Rosenbrock関数の最適化
   - 小規模NNの学習

2. **比較テスト**
   - PyTorchとの結果比較（同じ初期値、同じデータで）
   - 収束速度の比較

3. **パフォーマンステスト**
   - GPU vs CPU
   - 大規模パラメータでの速度

## 実装の優先順位

**Priority 1 (MVP):**
- Optimizer trait
- 基本SGD（モメンタムなし）
- Adam（最も一般的）
- 基本テスト

**Priority 2 (実用):**
- SGD with Momentum
- AdamW（現代的な推奨optimizer）
- Parameter groups

**Priority 3 (完全):**
- Nesterov momentum
- AMSGrad
- State保存/読み込み

**Priority 4 (拡張):**
- RMSprop
- Adagrad
- その他のoptimizer

## 期待される結果

| Optimizer | 二次関数収束 | NN学習収束 | メモリ使用量 |
|-----------|-------------|-----------|-------------|
| SGD | 50 steps | 200 epochs | 1x |
| Momentum | 30 steps | 100 epochs | 2x |
| Adam | 20 steps | 50 epochs | 3x |
| AdamW | 20 steps | 50 epochs | 3x |

## 実装ノート

- **In-place更新**: パラメータの更新は`sub_()`等のin-place操作を使用
- **GPU対応**: 全ての演算はデバイスに応じて自動的にGPU/CPU選択
- **f16精度**: モメンタムバッファもf16で保持（メモリ効率）
- **Thread-safe**: マルチスレッド学習時の安全性確保
- **Registry更新**: パラメータ更新後にAutogradContextへ再登録

## 将来の拡張

1. **Learning Rate Scheduler**: Cosine annealing, Step decay, Warmup
2. **Gradient Clipping**: Norm clipping, Value clipping
3. **Mixed Precision**: FP16学習対応
4. **Distributed Optimization**: Data parallel, Model parallel
5. **Second-order Methods**: L-BFGS（Phase 8.6完全実装後）

## 参考文献

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
- [On the Convergence of Adam and Beyond (AMSGrad)](https://arxiv.org/abs/1904.09237)
