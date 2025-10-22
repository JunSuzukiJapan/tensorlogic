# TensorLogic TODO リスト

このドキュメントはソースコード内に残っているTODOコメントを整理し、優先度と実装方針を記録しています。

**最終更新**: 2025-01-22

---

## 📊 概要

| カテゴリ | TODO数 | 優先度 |
|---------|--------|--------|
| インタープリター | 5 | 低〜中 |
| 最適化（Metal Kernel） | 4 | 低 |
| Python統合 | 2 | 低 |
| オプティマイザー | 2 | 低 |
| 型チェック | 2 | 低 |
| CoreML | 1 | 低 |
| Main | 2 | 低 |
| **合計** | **18** | - |

**注意**: `unimplemented!()`や`todo!()`マクロは**0個**です。すべての機能は動作しています。

---

## 🟡 優先度：中

### 1. インタープリター機能拡張

#### 1.1 制約ベースフィルタリング
**ファイル**: `src/interpreter/mod.rs`
**コメント**: `// TODO: Filter results based on constraints`

**現状**: 推論結果のフィルタリングが未実装

**提案実装**:
```rust
// 制約条件に基づいて結果をフィルタリング
fn filter_by_constraints(&self, results: Vec<Value>, constraints: &[Constraint]) -> Vec<Value> {
    results.into_iter()
        .filter(|v| self.satisfies_constraints(v, constraints))
        .collect()
}
```

**影響範囲**: 論理推論機能

**優先度**: 中（論理プログラミング機能を使う場合）

---

#### 1.2 リスト型の追加

**ファイル**: `src/interpreter/mod.rs`
**コメント**:
- `// TODO: Add list type to return all splits`
- `// TODO: Add list type to return all chunks`

**現状**: `split()`と`chunk()`が単一テンソルのみ返却

**問題点**:
```tensorlogic
let parts = split(tensor, 2, 0)  // 複数の結果を返すべき
```

**提案実装**:
```rust
pub enum Value {
    Tensor(Tensor),
    List(Vec<Value>),  // ← 追加
    // ...
}
```

**影響範囲**:
- `split()`, `chunk()`の戻り値
- イテレーション構文（for-loopなど）

**優先度**: 中（よく使われる機能）

**実装方針**:
1. `Value::List`型を追加
2. `split()`と`chunk()`を修正
3. リストのインデックスアクセス構文を追加
4. for-loopでのイテレーション対応

---

#### 1.3 カスタム次元サポート

**ファイル**: `src/interpreter/mod.rs`
**コメント**: `// TODO: support custom dimension when needed`

**現状**: 一部の操作で次元が固定されている

**提案**: 必要に応じて実装（現状で問題なし）

**優先度**: 低

---

#### 1.4 Layer Normalizationの引数パース

**ファイル**: `src/interpreter/mod.rs`
**コメント**: `// TODO: parse normalized_shape from argument`

**現状**: `layer_norm()`がデフォルト動作のみ

**提案実装**:
```tensorlogic
let normalized = layer_norm(tensor, [64], gamma, beta, 1e-5)
```

**優先度**: 低（現在のデフォルト動作で十分）

---

## 🟢 優先度：低

### 2. Metal Kernel最適化

すべて**性能最適化**のためのTODOです。現在の実装は動作しますが、最適化の余地があります。

#### 2.1 Concat操作の最適化

**ファイル**: `src/ops/tensor_ops.rs`
**コメント**: `// TODO: Implement proper Metal kernel for concat`

**現状**: CPUフォールバック実装

**提案**: 専用のMetal kernelで高速化

**期待効果**: 2-3x高速化

**実装方針**:
```metal
kernel void concat_kernel(
    device const half *src1 [[buffer(0)]],
    device const half *src2 [[buffer(1)]],
    device half *dst [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // ...
}
```

---

#### 2.2 Permute操作の最適化

**ファイル**: `src/ops/tensor_ops.rs`
**コメント**: `// TODO: Implement proper Metal kernel for permute`

**現状**: CPUフォールバック実装

**提案**: 専用のMetal kernelで高速化

**期待効果**: 2-5x高速化（特に大きなテンソルで効果的）

---

#### 2.3 Broadcast操作の最適化

**ファイル**: `src/ops/broadcast.rs`
**コメント**: `// TODO: Implement efficient Metal kernel for broadcasting`

**現状**: 汎用実装

**提案**: ブロードキャストパターンごとに最適化

**期待効果**: 2-4x高速化

---

#### 2.4 TensorShape::numel()の修正

**ファイル**: `src/ops/reduce.rs`（2箇所）
**コメント**: `// TODO: Fix TensorShape::numel() to return 1 for empty shapes`

**現状**: 空の形状で0を返す可能性

**提案**:
```rust
pub fn numel(&self) -> usize {
    if self.dims.is_empty() {
        return 1;  // スカラーテンソル
    }
    self.dims.iter().product()
}
```

**影響**: スカラーテンソルの処理

**優先度**: 低（エッジケース）

---

### 3. Python統合

#### 3.1 Python関数の戻り値

**ファイル**: `src/python/interpreter.rs`
**コメント**: `// TODO: Return actual result instead of None`

**現状**: Python関数が常にNoneを返す

**問題**: Python関数からの値の受け取りができない

**実装方針**:
```rust
pub fn call_python_function(&mut self, func_name: &str, args: Vec<Tensor>) -> Result<Value, String> {
    let py_result = self.py_env.call(func_name, args)?;
    Ok(self.convert_py_to_value(py_result))
}
```

**優先度**: 低（Python統合機能はオプション）

---

#### 3.2 Python関数の型推論

**ファイル**: `src/typecheck/mod.rs`
**コメント**: `// TODO: Python function type inference`

**現状**: Python関数の型情報なし

**提案**: 型ヒント（Type Hints）からの推論

**優先度**: 低（型チェックシステム全体が未完成）

---

### 4. オプティマイザーの状態シリアライズ

#### 4.1 Adam/AdamWの状態保存

**ファイル**:
- `src/optim/adam.rs`
- `src/optim/adamw.rs`

**コメント**: `// TODO: Serialize exp_avg, exp_avg_sq`

**現状**: オプティマイザーの状態（モーメント）を保存できない

**問題**: 学習を中断して再開できない

**実装方針**:
```rust
#[derive(Serialize, Deserialize)]
pub struct AdamState {
    pub exp_avg: HashMap<String, Tensor>,
    pub exp_avg_sq: HashMap<String, Tensor>,
}

impl Adam {
    pub fn save_state(&self, path: &Path) -> Result<()> {
        let state = AdamState {
            exp_avg: self.exp_avg.clone(),
            exp_avg_sq: self.exp_avg_sq.clone(),
        };
        // Serialize to file
    }
}
```

**優先度**: 低（学習機能の拡張時に実装）

---

### 5. CoreML統合

#### 5.1 MLFeatureDescriptionからの形状抽出

**ファイル**: `src/coreml/model.rs`
**コメント**: `// TODO: Extract actual shapes from MLFeatureDescription`

**現状**: CoreMLモデルの入出力形状が取得できない

**優先度**: 低（CoreML統合は実験的機能）

---

### 6. Main関連

#### 6.1 変数一覧の取得

**ファイル**: `src/main.rs`（2箇所）
**コメント**: `// TODO: Implement get_all_variables() method in Interpreter`

**現状**: REPLで定義された変数の一覧を表示できない

**提案実装**:
```rust
impl Interpreter {
    pub fn get_all_variables(&self) -> Vec<(String, Value)> {
        self.env.variables.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}
```

**用途**: REPL/デバッグツール

**優先度**: 低（便利機能）

---

### 7. 型チェック

#### 7.1 関数本体の型チェック

**ファイル**: `src/typecheck/mod.rs`
**コメント**: `// TODO: Type check function body statements`

**現状**: 関数シグネチャのみチェック、本体は未チェック

**優先度**: 低（型システム全体が未完成）

---

## 📋 実装ロードマップ

### Phase 1（短期 - 1-2週間）
- [ ] リスト型の追加（split/chunk対応）
- [ ] 変数一覧取得の実装
- [ ] TensorShape::numel()の修正

### Phase 2（中期 - 1-2ヶ月）
- [ ] Metal kernel最適化（concat, permute, broadcast）
- [ ] 制約ベースフィルタリング
- [ ] Layer Normalization引数パース

### Phase 3（長期 - 3-6ヶ月）
- [ ] オプティマイザー状態シリアライズ
- [ ] Python統合の完全実装
- [ ] 型システムの完全実装

---

## 🚫 対応不要

以下のTODOは**意図的に対応しない**または**低優先度すぎて無視**：

1. **Python統合**: オプション機能なので急がない
2. **CoreML統合**: 実験的機能
3. **型チェック**: 完全な型システムは将来の大規模リファクタリング時に実装

---

## 📝 新規TODO追加時のガイドライン

新しいTODOを追加する際は：

1. **優先度を明記**: 🔴高 / 🟡中 / 🟢低
2. **コンテキストを記載**: なぜ必要か、影響範囲は？
3. **実装方針を提案**: 具体的なコード例
4. **このドキュメントを更新**: TODO.mdに追記

---

## 🔗 関連ドキュメント

- [追加された操作](./added_operations_2025.md) - 今回追加した29個の操作
- [Transformers in Tensor Logic](../Papers/Transformers_in_tensor_logic.md) - アーキテクチャ設計
- [Graph Neural Networks](../Papers/graph_neural_networks_in_tensor_logic.md) - GNN実装

---

このドキュメントは定期的に更新し、完了したTODOは削除してください。
