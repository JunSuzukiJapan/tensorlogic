# Tensor Refactoring Plan

## 目的
Tensorのコードが長い（724行）ため、複数のtraitに分割して別ファイルに配置する。

## Trait分割設計

### 1. TensorCreation (tensor_creation.rs)
作成関連のメソッド（47-206行）

**メソッド**:
- `new(buffer, shape, device)` - 基本的なコンストラクタ
- `new_with_pool(buffer, shape, device, buffer_pool)` - プール付きコンストラクタ
- `from_vec(data, shape)` - CPU上にVecから作成
- `from_vec_metal(device, data, shape)` - Metal上にVecから作成
- `from_vec_metal_pooled(device, data, shape)` - プール使用
- `zeros(device, shape)` - ゼロ初期化
- `ones(device, shape)` - 1で初期化
- `scalar(device, value)` - スカラー値

**特徴**:
- 全てTensorを返す静的メソッドまたは関連関数
- f16とf32両方をサポート（ただしpoolはf16のみ）

### 2. TensorAccessors (tensor_accessors.rs)
アクセサー関連のメソッド（207-248行）

**メソッド**:
- `shape()` - TensorShapeへの参照
- `dims()` - 次元配列
- `rank()` - ランク（次元数）
- `numel()` - 要素数
- `strides()` - ストライド配列
- `device()` - デバイス
- `buffer()` - BufferHandleへの参照
- `buffer_pool()` - BufferPoolへの参照（内部用）

**特徴**:
- 全て不変参照を返す
- ジェネリック型Tに依存しない（一部を除く）

### 3. TensorTransform (tensor_transform.rs)
変形関連のメソッド（314-337行）

**メソッド**:
- `reshape(new_shape)` - 形状変更
- `flatten()` - 1次元化

**特徴**:
- 新しいTensorを返す
- バッファは共有、形状とストライドのみ変更

### 4. TensorIO (tensor_io.rs)
入出力関連のメソッド（273-457行）

**メソッド**:
- `to_cpu()` - CPUへ転送
- `to_metal(device)` - Metalへ転送
- `to_vec()` - Vec<T>として取得
- `to_vec_f32()` - Vec<f32>として取得
- `save(path)` - ファイルへ保存
- `load(device, path)` - ファイルから読み込み

**特徴**:
- デバイス間転送とシリアライゼーション
- 型サイズ検証（save/load）

### 5. TensorAutograd (tensor_autograd.rs)
自動微分関連のメソッド（244-581行）

**メソッド**:
- `requires_grad()` - 勾配計算が必要か
- `set_requires_grad(requires)` - 勾配計算フラグ設定
- `grad()` - 勾配取得
- `set_grad(grad)` - 勾配設定
- `zero_grad()` - 勾配ゼロ化
- `grad_node()` - ノードID取得
- `set_grad_node(node_id)` - ノードID設定
- `version()` - バージョン取得
- `backward()` - スカラーからの逆伝播
- `backward_with_grad(grad)` - 勾配指定の逆伝播
- `backward_create_graph()` - グラフ生成付き逆伝播

**特徴**:
- AutogradContextと密接に連携
- f16とf32両方をサポート（AutogradContextの変更が必要）

## AutogradContext のジェネリック化

### 課題
`TENSOR_REGISTRY: HashMap<NodeId, Tensor>` がTensorを直接格納しているが、
thread_localでジェネリック型を直接使用できない。

### 解決策: TensorVariant enum
```rust
pub enum TensorVariant {
    F16(Tensor<f16>),
    F32(Tensor<f32>),
}

impl TensorVariant {
    pub fn as_f16(&self) -> Option<&Tensor<f16>> { ... }
    pub fn as_f32(&self) -> Option<&Tensor<f32>> { ... }
    pub fn clone_f16(&self) -> Option<Tensor<f16>> { ... }
    pub fn clone_f32(&self) -> Option<Tensor<f32>> { ... }
}
```

### AutogradContext の変更
```rust
thread_local! {
    static TENSOR_REGISTRY: RefCell<HashMap<NodeId, TensorVariant>> = ...;
}

impl AutogradContext {
    pub fn register_tensor_f16(node_id: NodeId, tensor: Tensor<f16>) { ... }
    pub fn register_tensor_f32(node_id: NodeId, tensor: Tensor<f32>) { ... }

    pub fn get_tensor_f16(node_id: NodeId) -> Option<Tensor<f16>> { ... }
    pub fn get_tensor_f32(node_id: NodeId) -> Option<Tensor<f32>> { ... }

    pub fn backward_f16(node_id: NodeId, grad: Tensor<f16>) -> ... { ... }
    pub fn backward_f32(node_id: NodeId, grad: Tensor<f32>) -> ... { ... }
}
```

## 実装順序

1. ✅ 分析とプラン作成
2. TensorVariant enum作成（autograd/tensor_variant.rs）
3. TensorAccessors trait作成（tensor/tensor_accessors.rs）
4. TensorCreation trait作成（tensor/tensor_creation.rs）
5. TensorTransform trait作成（tensor/tensor_transform.rs）
6. TensorIO trait作成（tensor/tensor_io.rs）
7. TensorAutograd trait作成（tensor/tensor_autograd.rs）
8. AutogradContext ジェネリック化（autograd/context.rs）
9. tensor.rs リファクタリング（各traitのimpl）
10. コンパイルエラー修正と動作確認

## ファイル構成（完成後）

```
src/
├── tensor/
│   ├── mod.rs                  # 全exportとTensor構造体定義
│   ├── tensor.rs               # 基本実装（Drop, PartialEq等）
│   ├── tensor_creation.rs      # TensorCreation trait + impl
│   ├── tensor_accessors.rs     # TensorAccessors trait + impl
│   ├── tensor_transform.rs     # TensorTransform trait + impl
│   ├── tensor_io.rs            # TensorIO trait + impl
│   ├── tensor_autograd.rs      # TensorAutograd trait + impl
│   ├── buffer_handle.rs        # BufferHandle<T>
│   ├── shape.rs                # TensorShape
│   ├── tensor_like.rs          # TensorLike trait
│   └── float_type.rs           # FloatType trait
└── autograd/
    ├── context.rs              # AutogradContext（ジェネリック対応）
    ├── tensor_variant.rs       # TensorVariant enum
    └── ...
```

## 期待される効果

1. **可読性向上**: 各トレイトが明確な責務を持つ
2. **メンテナンス性向上**: 機能ごとにファイルが分離
3. **ジェネリック化完了**: f16とf32の両方をサポート
4. **コンパイル成功**: Phase 1の完了
