# Transformer & GNN Implementation Checklist

**目的**: TransformerとGraph Neural Networksを記述できるように、必要な組み込み関数を実装する

**開始日**: 2025-10-21
**ステータス**: 🔄 進行中

---

## 📋 実装計画概要

### 優先度と実装順序

1. ✅ **チェックリストドキュメント作成** ← 現在
2. ⏳ **Phase 1**: 基本数学関数 (exp, log, sqrt, pow, sin, cos, tan)
3. ⏳ **Phase 2**: 活性化関数 (sigmoid, tanh)
4. ⏳ **Phase 4**: テンソル操作 (concat, transpose)
5. ⏳ **Phase 3**: Layer Normalization
6. ⏳ **Phase 6**: Autograd対応
7. ⏳ **Phase 7**: Transformer & GNN サンプル実装

**総所要時間見積**: 2-3週間

---

## Phase 1: 基本数学関数 (3-4日) ✅ COMPLETED

### 目標
要素ごとの数学関数を実装（exp, log, sqrt, pow, sin, cos, tan）

### タスクリスト

#### 1.1 Metal Shader実装
- [x] `shaders/elementwise.metal` に以下を追加:
  - [x] `kernel void exp_f16(...)`
  - [x] `kernel void log_f16(...)`
  - [x] `kernel void sqrt_f16(...)`
  - [x] `kernel void pow_f16(...)`
  - [x] `kernel void sin_f16(...)`
  - [x] `kernel void cos_f16(...)`
  - [x] `kernel void tan_f16(...)`

#### 1.2 Rust実装 (`src/ops/elementwise.rs`)
- [x] `pub fn exp(&self) -> TensorResult<Tensor>`
  - [x] `exp_metal()` 実装
  - [x] `exp_cpu()` 実装
- [x] `pub fn log(&self) -> TensorResult<Tensor>`
  - [x] `log_metal()` 実装
  - [x] `log_cpu()` 実装
- [x] `pub fn sqrt(&self) -> TensorResult<Tensor>`
  - [x] `sqrt_metal()` 実装
  - [x] `sqrt_cpu()` 実装
- [x] `pub fn pow(&self, exponent: f32) -> TensorResult<Tensor>`
  - [x] `pow_metal()` 実装
  - [x] `pow_cpu()` 実装
- [x] `pub fn sin(&self) -> TensorResult<Tensor>`
  - [x] `sin_metal()` 実装
  - [x] `sin_cpu()` 実装
- [x] `pub fn cos(&self) -> TensorResult<Tensor>`
  - [x] `cos_metal()` 実装
  - [x] `cos_cpu()` 実装
- [x] `pub fn tan(&self) -> TensorResult<Tensor>`
  - [x] `tan_metal()` 実装
  - [x] `tan_cpu()` 実装

#### 1.3 テスト実装
- [x] `test_exp()`: Metal vs CPU, 既知の値との比較
- [x] `test_log()`: Metal vs CPU, 既知の値との比較
- [x] `test_sqrt()`: Metal vs CPU, 既知の値との比較
- [x] `test_pow()`: Metal vs CPU, 様々な指数
- [x] `test_sin()`: Metal vs CPU, 周期性確認
- [x] `test_cos()`: Metal vs CPU, 周期性確認
- [x] `test_tan()`: Metal vs CPU, 特異点確認

#### 1.4 検証
- [x] 全テストがパス (Metal + CPU) - 7/7 tests passing
- [x] 数値精度確認 (f16の範囲内)
- [x] パフォーマンス測定 (Metal vs CPU)

---

## Phase 2: 活性化関数 (1-2日) ✅ COMPLETED

### 目標
sigmoid と tanh を実装

### タスクリスト

#### 2.1 Metal Shader実装
- [x] `shaders/elementwise.metal` に追加:
  - [x] `kernel void sigmoid_f16(...)` - 1/(1+exp(-x))
  - [x] `kernel void tanh_f16(...)` - tanh(x)

#### 2.2 Rust実装 (`src/ops/activations.rs`)
- [x] `pub fn sigmoid(&self) -> TensorResult<Tensor>`
  - [x] `sigmoid_metal()` 実装
  - [x] `sigmoid_cpu()` 実装
- [x] `pub fn tanh(&self) -> TensorResult<Tensor>`
  - [x] `tanh_metal()` 実装
  - [x] `tanh_cpu()` 実装

#### 2.3 テスト実装
- [x] `test_sigmoid()`: Metal vs CPU, 範囲確認 (0-1)
- [x] `test_tanh()`: Metal vs CPU, 範囲確認 (-1 to 1)
- [x] 極値での動作確認（テスト内で実装）

#### 2.4 検証
- [x] 全テストがパス - 2/2 tests passing
- [x] 既存の活性化関数 (ReLU, GELU, Softmax) との統合確認

---

## Phase 4: テンソル操作 (2-3日) ✅ COMPLETED

### 目標
concat と transpose/permute を実装

### タスクリスト

#### 4.1 新規ファイル作成
- [x] `src/ops/tensor_ops.rs` 作成
- [x] `src/ops/mod.rs` に `pub mod tensor_ops;` 追加
- Note: Metal kernel実装は後回し（現在はCPUフォールバック）

#### 4.2 Concat実装
- [x] `pub fn concat(tensors: &[&Tensor], dim: usize) -> TensorResult<Tensor>`
  - [x] 次元チェック (dim以外の次元が一致)
  - [x] 新しい形状計算
  - [x] Metal実装: CPUフォールバック（TODO: 最適化）
  - [x] CPU実装: ループによるコピー
- [x] テスト:
  - [x] `test_concat_dim0()`: 次元0での連結
  - [x] `test_concat_dim1()`: 次元1での連結
  - [x] `test_concat_multiple()`: 3個以上のテンソル連結

#### 4.3 Transpose/Permute実装
- [x] `pub fn transpose(&self) -> TensorResult<Tensor>` (2Dのみ)
  - [x] CPU実装: ストライド計算とインデックス変換
- [x] `pub fn permute(&self, dims: Vec<usize>) -> TensorResult<Tensor>`
  - [x] 次元順序の検証
  - [x] ストライド再計算
  - [x] CPU実装: 完全なインデックス変換
- [x] テスト:
  - [x] `test_transpose_2d()`: 2D転置
  - [x] `test_permute_3d()`: 3D次元入れ替え
  - [x] `test_permute_identity()`: 恒等変換

#### 4.4 検証
- [x] 全テストがパス - 6/6 tests passing
- [x] メモリ効率確認
- Note: Metal kernel最適化は将来の改善として残す

---

## Phase 3: Layer Normalization (2-3日) ✅ COMPLETED

### 目標
Layer Normalization を実装

### タスクリスト

#### 3.1 新規ファイル作成
- [x] `src/ops/normalization.rs` 作成
- [x] `shaders/normalization.metal` 作成
- [x] `src/ops/mod.rs` に `pub mod normalization;` 追加

#### 3.2 Layer Norm実装
- [x] `pub fn layer_norm(&self, normalized_shape: Vec<usize>, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> TensorResult<Tensor>`
  - [x] 平均計算（カスタム実装）
  - [x] 分散計算
  - [x] 正規化: (x - mean) / sqrt(var + eps)
  - [x] アフィン変換: γ * normalized + β
  - [x] Metal実装（2つのカーネル: 通常版と小サイズ用）
  - [x] CPU実装

#### 3.3 Metal Shader実装
- [x] `kernel void layer_norm_f16(...)`:
  - [x] 平均・分散のreduction（parallel reduction使用）
  - [x] 正規化とアフィン変換
  - [x] threadgroup共有メモリ使用（最適化）
- [x] `kernel void layer_norm_simple_f16(...)`：小テンソル用の簡易版

#### 3.4 テスト実装
- [x] `test_layer_norm_basic()`: 基本動作（平均≈0、標準偏差≈1）
- [x] `test_layer_norm_with_affine()`: γ, β あり
- [x] `test_layer_norm_3d()`: 3D テンソルでの動作確認
- [x] `test_layer_norm_cpu()`: CPU実装の検証

#### 3.5 検証
- [x] 全テストがパス - 4/4 tests passing
- [x] 数値精度確認（f16精度で正規化確認）
- Note: PyTorch比較テストは将来の拡張として残す

---

## Phase 6: Autograd対応 (3-5日)

### 目標
全演算の勾配関数を実装

### タスクリスト

#### 6.1 基本数学関数の勾配
- [ ] `src/autograd/gradients/exp.rs` 作成
  - [ ] `exp_backward()`: d/dx exp(x) = exp(x)
  - [ ] Metal shader: `exp_backward_f16`
- [ ] `src/autograd/gradients/log.rs` 作成
  - [ ] `log_backward()`: d/dx log(x) = 1/x
  - [ ] Metal shader: `log_backward_f16`
- [ ] `src/autograd/gradients/sqrt.rs` 作成
  - [ ] `sqrt_backward()`: d/dx sqrt(x) = 1/(2*sqrt(x))
  - [ ] Metal shader: `sqrt_backward_f16`
- [ ] `src/autograd/gradients/pow.rs` 作成
  - [ ] `pow_backward()`: d/dx x^n = n*x^(n-1)
  - [ ] Metal shader: `pow_backward_f16`
- [ ] `src/autograd/gradients/sin.rs` 作成
  - [ ] `sin_backward()`: d/dx sin(x) = cos(x)
  - [ ] Metal shader: `sin_backward_f16`
- [ ] `src/autograd/gradients/cos.rs` 作成
  - [ ] `cos_backward()`: d/dx cos(x) = -sin(x)
  - [ ] Metal shader: `cos_backward_f16`

#### 6.2 活性化関数の勾配
- [ ] `src/autograd/gradients/sigmoid.rs` 作成
  - [ ] `sigmoid_backward()`: d/dx σ(x) = σ(x)*(1-σ(x))
  - [ ] Metal shader: `sigmoid_backward_f16`
- [ ] `src/autograd/gradients/tanh.rs` 作成
  - [ ] `tanh_backward()`: d/dx tanh(x) = 1-tanh²(x)
  - [ ] Metal shader: `tanh_backward_f16`

#### 6.3 テンソル操作の勾配
- [ ] `src/autograd/gradients/concat.rs` 作成
  - [ ] `concat_backward()`: 勾配を分割して各入力へ
  - [ ] Metal shader (必要に応じて)
- [ ] `src/autograd/gradients/transpose.rs` 作成
  - [ ] `transpose_backward()`: 転置を逆転
  - [ ] Metal shader (必要に応じて)

#### 6.4 Layer Normの勾配
- [ ] `src/autograd/gradients/layer_norm.rs` 作成
  - [ ] `layer_norm_backward()`: 複雑な勾配計算
  - [ ] Metal shader: `layer_norm_backward_f16`
  - [ ] weight/biasの勾配も計算

#### 6.5 Autograd統合
- [ ] `src/autograd/mod.rs` に全勾配関数登録
- [ ] 各演算の `record_operation()` 更新
- [ ] GradNode に新演算追加

#### 6.6 勾配チェック
- [ ] `test_gradient_exp()`: 数値微分との比較
- [ ] `test_gradient_log()`: 数値微分との比較
- [ ] `test_gradient_sqrt()`: 数値微分との比較
- [ ] `test_gradient_sigmoid()`: 数値微分との比較
- [ ] `test_gradient_tanh()`: 数値微分との比較
- [ ] `test_gradient_layer_norm()`: 数値微分との比較
- [ ] `test_gradient_concat()`: 数値微分との比較

#### 6.7 検証
- [ ] 全勾配チェックテストがパス
- [ ] エンドツーエンドの backward pass 確認
- [ ] 小規模学習テスト (収束確認)

---

## Phase 7: Transformer & GNN サンプル (2-3日)

### 目標
実装した機能を使ってTransformerとGNNを記述

### タスクリスト

#### 7.1 Positional Encoding サンプル
- [ ] `examples/positional_encoding.tl` 作成
  - [ ] sin/cosを使った位置エンコーディング
  - [ ] 実行テスト
  - [ ] 可視化 (オプション)

#### 7.2 Attention メカニズム
- [ ] `examples/attention.tl` 作成
  - [ ] Scaled Dot-Product Attention
  - [ ] Query/Key/Value計算
  - [ ] Softmax attention
  - [ ] 実行テスト

#### 7.3 Multi-Head Attention
- [ ] `examples/multi_head_attention.tl` 作成
  - [ ] ヘッド分割 (permute/reshape使用)
  - [ ] 並列attention計算
  - [ ] concat によるマージ
  - [ ] 実行テスト

#### 7.4 Transformer Block
- [ ] `examples/transformer_block.tl` 作成
  - [ ] Multi-head Self-Attention
  - [ ] Layer Normalization
  - [ ] Feed-Forward Network (MLP)
  - [ ] Residual connections
  - [ ] 実行テスト

#### 7.5 Full Transformer
- [ ] `examples/transformer.tl` 作成
  - [ ] 複数のTransformerブロック
  - [ ] 入力埋め込み + 位置エンコーディング
  - [ ] 出力層
  - [ ] 小規模学習テスト

#### 7.6 GNN サンプル
- [ ] `examples/gnn.tl` 作成
  - [ ] グラフ構造定義 (relation Neig)
  - [ ] Message Passing
  - [ ] Aggregation
  - [ ] Node Update
  - [ ] 実行テスト

#### 7.7 GNN タスク
- [ ] `examples/gnn_node_classification.tl`
  - [ ] ノード分類タスク
  - [ ] 学習ループ
- [ ] `examples/gnn_edge_prediction.tl`
  - [ ] エッジ予測タスク
  - [ ] 学習ループ

#### 7.8 ドキュメント作成
- [ ] `claudedocs/transformer_implementation.md`
  - [ ] 実装詳細
  - [ ] 使用例
  - [ ] パフォーマンス測定結果
- [ ] `claudedocs/gnn_implementation.md`
  - [ ] 実装詳細
  - [ ] 使用例
  - [ ] グラフ構造の扱い方

#### 7.9 検証
- [ ] Transformerサンプルが動作
- [ ] GNNサンプルが動作
- [ ] 学習が収束することを確認
- [ ] README更新

---

## Phase 5: インデックス操作 (オプション, 2-3日)

### 目標
gather と scatter を実装 (後回し可)

### タスクリスト

#### 5.1 新規ファイル作成
- [ ] `src/ops/indexing.rs` 作成
- [ ] `shaders/indexing.metal` 作成
- [ ] `src/ops/mod.rs` に追加

#### 5.2 Gather/Scatter実装
- [ ] `pub fn gather(&self, dim: usize, indices: &Tensor) -> TensorResult<Tensor>`
- [ ] `pub fn scatter(&self, dim: usize, indices: &Tensor, src: &Tensor) -> TensorResult<Tensor>`
- [ ] Metal実装
- [ ] CPU実装
- [ ] テスト

---

## 📊 進捗トラッキング

### 全体進捗
- **Phase 1**: ⬜⬜⬜⬜⬜⬜⬜ 0/7 完了
- **Phase 2**: ⬜⬜ 0/2 完了
- **Phase 4**: ⬜⬜⬜ 0/3 完了
- **Phase 3**: ⬜ 0/1 完了
- **Phase 6**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/10 完了
- **Phase 7**: ⬜⬜⬜⬜⬜⬜⬜⬜ 0/8 完了

**総合進捗**: 12/32 タスク完了 (38%) ✅ Phase 1-2-4 完了

### タイムライン
- **Week 1**: Phase 1 + Phase 2
- **Week 2**: Phase 4 + Phase 3
- **Week 3**: Phase 6 + Phase 7

---

## �� テスト戦略

### レベル1: ユニットテスト
各関数の基本動作確認

### レベル2: 数値テスト
既知の値や他のライブラリ (PyTorch) との比較

### レベル3: 勾配チェック
数値微分との比較 (gradient checking)

### レベル4: デバイステスト
Metal vs CPU の結果一致確認

### レベル5: 統合テスト
複数演算の組み合わせ、エンドツーエンド

---

## 📝 メモ・課題

### 技術的決定
- f16精度での数値安定性確保
- Metal shader の最適化 (threadgroup共有メモリ)
- Layer Normの効率的な実装

### 既知の制約
- f16の表現範囲: ±65504, 最小正規化数: 6.10e-5
- Metal GPUスレッド数制限
- Neural Engineへのフォールバック戦略

### 将来の拡張
- Batch Normalization
- Dropout
- Attention Mask
- Convolution (CNN用)

---

## ✅ 完了基準

### Phase完了条件
- [ ] 全ユニットテストがパス
- [ ] Metal/CPU両方の実装完了
- [ ] ドキュメント更新
- [ ] 次Phaseの依存関係クリア

### プロジェクト完了条件
- [ ] 全7 Phaseが完了
- [ ] Transformerサンプルが動作
- [ ] GNNサンプルが動作
- [ ] 学習が収束することを確認
- [ ] パフォーマンス測定完了
- [ ] ドキュメント整備完了
- [ ] README更新完了

---

**最終更新**: 2025-10-21
**次のアクション**: Phase 3 (Layer Normalization) 実装

## 実装完了サマリー (2025-10-21)

### Phase 1: 基本数学関数 ✅
- 実装: exp, log, sqrt, pow, sin, cos, tan
- Metal Shader: 7カーネル
- テスト: 7/7 passing
- ファイル: `shaders/elementwise.metal`, `src/ops/elementwise.rs`

### Phase 2: 活性化関数 ✅
- 実装: sigmoid, tanh
- Metal Shader: 2カーネル
- テスト: 2/2 passing
- ファイル: `shaders/elementwise.metal`, `src/ops/activations.rs`

### Phase 4: テンソル操作 ✅
- 実装: concat, transpose, permute
- CPU実装完了、Metal最適化は今後
- テスト: 6/6 passing
- ファイル: `src/ops/tensor_ops.rs`

**総テスト**: 15/15 passing
**総合進捗**: 38% (12/32タスク完了)
