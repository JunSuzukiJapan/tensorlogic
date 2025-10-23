# Neural-Symbolic Learning 基礎ガイド

**対象読者**: 深層学習の基礎知識がない開発者向け
**最終更新**: 2025-10-23

## 目次

1. [学習に必要なデータ](#1-学習に必要なデータ)
2. [学習時に使う関数の役割](#2-学習時に使う関数の役割)
3. [学習の全体像](#3-学習の全体像)
4. [learnブロックの内部動作](#4-learnブロックの内部動作)
5. [learnableキーワードの詳細](#5-learnableキーワードの詳細)
6. [まとめ](#6-まとめ)

---

## 1. 学習に必要なデータ

### 1.1 既存のテンソル学習（現在実装済み）

まず、現在の`learn`ブロックで使っているデータを振り返ります：

```tensorlogic
// パラメータ（学習対象）
tensor w: float16[10] learnable = [...]

// データ
tensor x: float16[10] = [1.0, 2.0, ...]
tensor y_true: float16[1] = [5.0]

main {
    learn {
        // 予測を計算
        y_pred := w * x

        // 誤差を計算
        error := y_pred - y_true
        loss := error * error

        objective: loss,      // ← 最小化したい値
        optimizer: sgd(lr: 0.1),
        epochs: 100
    }
}
```

**データの種類**：
- **学習パラメータ** (`learnable`): 学習で調整される値（重み `w`）
- **入力データ**: 固定値（`x`）
- **正解ラベル**: 目標値（`y_true`）
- **損失関数**: 予測と正解の差（`loss`）

---

### 1.2 論理プログラミングでの学習データ

論理プログラミングと統合する場合、データの形式が変わります：

#### 知識グラフ埋め込み学習の例

```tensorlogic
// 関係の定義（学習対象の埋め込みを持つ）
relation Friend(x: entity, y: entity) embed float16[64] learnable

// エンティティの埋め込み（学習対象）
embedding person_embed {
    entities: {alice, bob, charlie, diana}
    dimension: 64
    init: xavier
} learnable

main {
    // ========================================
    // データ1: 観測されたファクト（正例）
    // ========================================
    Friend(alice, bob)      // aliceとbobは友達
    Friend(bob, charlie)    // bobとcharlieは友達
    Friend(charlie, diana)  // charlieとdianaは友達

    // ========================================
    // データ2: 成り立たないファクト（負例）
    // ========================================
    // これは通常、自動生成されます
    // 例: Friend(alice, diana) は偽

    // ========================================
    // データ3: 論理的制約
    // ========================================
    // 対称性: 友達関係は双方向
    // Friend(X, Y) -> Friend(Y, X)

    // 推移性: 友達の友達は友達かも
    // Friend(X, Y), Friend(Y, Z) -> Friend(X, Z)

    learn {
        // 学習の目標:
        // 1. 正例のファクトに高いスコアを付ける
        // 2. 負例のファクトに低いスコアを付ける
        // 3. 論理的制約を満たすように調整

        objective: ranking_loss + constraint_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }
}
```

#### データの種類詳細

**1. 正例ファクト（Positive Facts）**
```tensorlogic
Friend(alice, bob)
Friend(bob, charlie)
```
- 観測された真のファクト
- "aliceとbobは実際に友達"という情報
- これらに高いスコア（確信度）を付けたい

**2. 負例ファクト（Negative Facts）**
```tensorlogic
Friend(alice, diana) は偽
Friend(eve, frank) は偽
```
- 成り立たないファクト
- 通常は自動生成（ランダムサンプリング）
- これらに低いスコアを付けたい

**3. エンティティ埋め込み（Entity Embeddings）**
```
alice  → [0.1, -0.3, 0.5, ..., 0.2]  (64次元ベクトル)
bob    → [0.2, -0.1, 0.4, ..., 0.3]
charlie → [0.3, 0.0, 0.6, ..., 0.1]
```
- 各人物を数値ベクトルで表現
- 学習で最適化される

**4. 関係埋め込み（Relation Embeddings）**
```
Friend → [0.5, 0.3, -0.2, ..., 0.4]  (64次元ベクトル)
```
- 関係も数値ベクトルで表現
- 学習で最適化される

**5. 論理的制約（Logic Constraints）**
- 対称性、推移性などのルール
- これらを"ソフト制約"として損失に追加

---

## 2. 学習時に使う関数の役割

### 2.1 基本的な関数の流れ

```
入力データ → スコア計算 → 損失計算 → 勾配計算 → パラメータ更新
```

各関数の詳細を説明します。

---

### 2.2 スコア関数（Scoring Function）

**役割**: ファクトがどれくらい"正しそうか"を数値で評価

**例**: TransE（知識グラフ埋め込みの代表的手法）

```tensorlogic
// Friend(alice, bob) のスコアを計算
fn score_fact(subject: entity, relation: entity, object: entity) -> float16 {
    // エンティティと関係の埋め込みを取得
    let h = person_embed[subject]      // alice の埋め込み [64次元]
    let r = relation_embed[relation]   // Friend の埋め込み [64次元]
    let t = person_embed[object]       // bob の埋め込み [64次元]

    // TransEスコア: h + r ≈ t なら高スコア
    // つまり「alice + Friend関係 = bob」という関係性
    let diff = h + r - t
    let score = -norm(diff)  // 距離が小さいほど高スコア

    return score
}

// 使い方
score1 := score_fact(alice, Friend, bob)      // 正例: 高いスコアを期待
score2 := score_fact(alice, Friend, diana)    // 負例: 低いスコアを期待
```

**イメージ**:
```
正例 Friend(alice, bob):
  alice_vec + friend_vec ≈ bob_vec
  [0.1, 0.2] + [0.5, 0.3] ≈ [0.6, 0.5]  ← 近い！スコア高い

負例 Friend(alice, diana):
  alice_vec + friend_vec ≠ diana_vec
  [0.1, 0.2] + [0.5, 0.3] ≠ [0.9, 0.1]  ← 遠い！スコア低い
```

---

### 2.3 損失関数（Loss Function）

**役割**: 現在の予測がどれくらい"間違っているか"を数値化

#### 2.3.1 ランキング損失（Margin Ranking Loss）

```tensorlogic
fn margin_ranking_loss(
    positive_score: float16,
    negative_score: float16,
    margin: float16
) -> float16 {
    // 正例のスコアは負例より margin 以上高くあるべき
    // positive_score > negative_score + margin

    let loss = max(0, margin - positive_score + negative_score)
    return loss
}

// 例
positive_score := score_fact(alice, Friend, bob)       // 0.8
negative_score := score_fact(alice, Friend, diana)     // 0.3
margin := 1.0

loss := margin_ranking_loss(0.8, 0.3, 1.0)
// = max(0, 1.0 - 0.8 + 0.3)
// = max(0, 0.5)
// = 0.5  ← まだ差が足りない、もっと学習が必要
```

**イメージ**:
```
目標: positive_score - negative_score >= margin

現在: 0.8 - 0.3 = 0.5  ← margin (1.0) より小さい
→ loss = 0.5 (まだ改善の余地あり)

理想: 0.9 - 0.1 = 0.8  ← margin (1.0) より小さいが近づいた
→ loss = 0.2 (改善した)

完璧: 1.5 - 0.3 = 1.2  ← margin (1.0) を超えた
→ loss = 0.0 (完璧！)
```

#### 2.3.2 論理制約損失（Logic Constraint Loss）

```tensorlogic
fn symmetry_constraint_loss() -> float16 {
    // 対称性: Friend(X, Y) なら Friend(Y, X) も成り立つべき

    let score_forward = score_fact(alice, Friend, bob)   // 0.8
    let score_backward = score_fact(bob, Friend, alice)  // 0.6

    // 両方向のスコアの差を最小化
    let diff = score_forward - score_backward
    let loss = diff * diff  // 二乗誤差

    return loss
    // = (0.8 - 0.6)^2 = 0.04
}

fn transitivity_constraint_loss() -> float16 {
    // 推移性: Friend(X,Y) かつ Friend(Y,Z) なら Friend(X,Z) も成り立つべき

    let score_xy = score_fact(alice, Friend, bob)      // 0.8
    let score_yz = score_fact(bob, Friend, charlie)    // 0.9
    let score_xz = score_fact(alice, Friend, charlie)  // 0.4  ← 低い！

    // Friend(alice, charlie) のスコアを
    // min(score_xy, score_yz) に近づける（ファジー論理）
    let expected = min(score_xy, score_yz)  // min(0.8, 0.9) = 0.8
    let diff = score_xz - expected
    let loss = diff * diff

    return loss
    // = (0.4 - 0.8)^2 = 0.16  ← 大きい！改善が必要
}
```

---

### 2.4 類似度関数（Similarity Function）

**役割**: 2つのベクトルがどれくらい似ているか測る

```tensorlogic
fn cosine_similarity(vec1: float16[?], vec2: float16[?]) -> float16 {
    // コサイン類似度: -1 (正反対) ~ 1 (同じ方向)
    let dot_product = sum(vec1 * vec2)
    let norm1 = sqrt(sum(vec1 * vec1))
    let norm2 = sqrt(sum(vec2 * vec2))

    return dot_product / (norm1 * norm2)
}

// 使い方
alice_vec := person_embed[alice]
bob_vec := person_embed[bob]
similarity := cosine_similarity(alice_vec, bob_vec)
// 0.8 → とても似ている（友達かも）
// 0.2 → あまり似ていない（友達じゃなさそう）
```

---

### 2.5 埋め込み取得関数

```tensorlogic
fn get_embedding(entity_name: entity) -> float16[64] {
    // エンティティ名から数値ベクトルを取得
    return person_embed[entity_name]
}

// 例
alice_vec := get_embedding(alice)
// → [0.1, -0.3, 0.5, 0.2, ..., 0.4]
```

---

## 3. 学習の全体像

実際の学習プロセスを追ってみます：

```tensorlogic
relation Friend(x: entity, y: entity) embed float16[64] learnable

embedding person_embed {
    entities: {alice, bob, charlie, diana}
    dimension: 64
    init: random  // ランダム初期化
} learnable

main {
    // ========================================
    // ステップ1: データの準備
    // ========================================

    // 正例
    Friend(alice, bob)
    Friend(bob, charlie)

    // ========================================
    // ステップ2: 学習ループ
    // ========================================

    learn {
        // --- Epoch 1 ---

        // 2.1 スコア計算
        pos_score_1 := score_fact(alice, Friend, bob)     // 0.3 (初期は低い)
        pos_score_2 := score_fact(bob, Friend, charlie)   // 0.4

        // 2.2 負例のスコア計算（自動生成）
        neg_score_1 := score_fact(alice, Friend, diana)   // 0.6 (初期は高い！)
        neg_score_2 := score_fact(bob, Friend, diana)     // 0.5

        // 2.3 損失計算
        ranking_loss_1 := margin_ranking_loss(0.3, 0.6, 1.0)  // 1.3
        ranking_loss_2 := margin_ranking_loss(0.4, 0.5, 1.0)  // 1.1

        // 2.4 制約損失
        symmetry_loss := symmetry_constraint_loss()  // 0.1

        // 2.5 総損失
        total_loss := ranking_loss_1 + ranking_loss_2 + symmetry_loss
        // = 1.3 + 1.1 + 0.1 = 2.5

        // --- Epoch 50 (学習が進んだ後) ---

        pos_score_1 := 0.8  // 上がった！
        neg_score_1 := 0.2  // 下がった！

        ranking_loss_1 := margin_ranking_loss(0.8, 0.2, 1.0)  // 0.4
        total_loss := 0.6  // 減った！

        // --- Epoch 100 (学習完了) ---

        pos_score_1 := 0.95  // さらに上がった！
        neg_score_1 := 0.05  // さらに下がった！

        total_loss := 0.05  // ほぼゼロ！

        objective: total_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    // ========================================
    // ステップ3: 学習後の推論
    // ========================================

    // 新しい質問: alice と charlie は友達？
    infer {
        forward Friend(alice, charlie)?
    }
    // → スコア 0.75 が計算される
    // → "おそらく友達"と推論できる
}
```

### データの流れ

```
1. 入力
   ├─ 正例ファクト: Friend(alice, bob)
   ├─ 負例ファクト: Friend(alice, diana) (自動生成)
   └─ エンティティ: {alice, bob, charlie, diana}

2. 埋め込み（学習パラメータ）
   ├─ alice  → [0.1, -0.3, ..., 0.2]
   ├─ bob    → [0.2, -0.1, ..., 0.3]
   └─ Friend → [0.5, 0.3, ..., 0.4]

3. スコア計算
   ├─ score(alice, Friend, bob) = 0.8  (高い)
   └─ score(alice, Friend, diana) = 0.3  (低い)

4. 損失計算
   ├─ ranking_loss = 0.5
   └─ constraint_loss = 0.1

5. 最適化
   └─ 埋め込みを更新して損失を減らす
```

### 関数の役割表

| 関数 | 入力 | 出力 | 役割 |
|------|------|------|------|
| **score_fact** | (alice, Friend, bob) | 0.8 | ファクトの確信度 |
| **margin_ranking_loss** | (pos=0.8, neg=0.3) | 0.5 | 正負の差の評価 |
| **symmetry_loss** | (前後スコア) | 0.1 | 対称性制約違反 |
| **cosine_similarity** | (vec1, vec2) | 0.75 | ベクトルの類似度 |
| **get_embedding** | alice | [0.1, ...] | 埋め込み取得 |

---

## 4. learnブロックの内部動作

### 4.1 隠されたループの実装

**重要**: `learn`ブロックには隠されたループが含まれています。

**実装箇所**: [src/interpreter/mod.rs:4498-4510](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/interpreter/mod.rs#L4498-L4510)

```rust
// Training loop with detailed progress display
println!("\n--- Training Progress ---");
for epoch in 0..spec.epochs {           // ← 隠されたループ！
    // Zero gradients before computing loss
    if epoch > 0 {
        opt.zero_grad();
    }

    // Re-execute statements for each epoch (recompute intermediate variables)
    for stmt in &spec.statements {      // ← さらに内側のループ！
        self.execute_statement(stmt)?;
    }

    // Compute loss
    let loss_val = self.eval_expr(&spec.objective)?;
    // ...勾配計算とパラメータ更新
}
```

### 4.2 ユーザーが書くコード vs 実際の実行

**ユーザーが書くコード**:
```tensorlogic
learn {
    pred := x * w
    loss := (pred - y) * (pred - y)

    objective: loss,
    epochs: 100          // ← これがループ回数
}
```

**実際の実行**（展開版）:
```rust
for epoch in 0..100 {
    // エポックごとに文を再実行
    pred := x * w                    // 毎回計算
    loss := (pred - y) * (pred - y)  // 毎回計算

    // 損失から勾配を計算
    backward(loss)

    // パラメータを更新
    w = w - lr * grad_w
}
```

---

## 5. learnableキーワードの詳細

### 5.1 構文の位置（前置 vs 後置）

#### 現在の構文（後置）
```tensorlogic
tensor w: float16[10] learnable
relation Friend(x: entity, y: entity) embed float16[64] learnable
```

#### 提案された構文（前置）
```tensorlogic
learnable tensor w: float16[10]
learnable relation Friend(x: entity, y: entity) embed float16[64]
```

#### 利点と欠点の比較

| 観点 | 現在（後置） | 提案（前置） |
|------|------------|------------|
| **宣言種別の判別** | ⭕ 即座 (`tensor`, `relation`) | ❌ 2トークン先を読む必要 |
| **オプション性** | ⭕ `learnable?` で表現可能 | ❌ 3パターン必要<br>(`learnable tensor`, `frozen tensor`, `tensor`) |
| **文法の複雑さ** | ⭕ シンプル | ❌ 選択肢が増える |
| **学習関連の検索** | ❌ 正規表現が必要 | ⭕ `^learnable` で検索可能 |

#### 現在の構文（後置）の利点

1. **宣言の種類が最初**
   ```
   tensor ...     ← すぐに「テンソル宣言だ」とわかる
   relation ...   ← すぐに「関係宣言だ」とわかる
   ```

2. **PEG文法と相性が良い**
   ```pest
   declaration = {
       tensor_decl      // "tensor" で始まる
       | relation_decl   // "relation" で始まる
       | rule_decl       // "rule" で始まる
   }
   ```

3. **オプション性が明確**
   ```pest
   tensor_type = { base_type ~ "[" ~ dimensions ~ "]" ~ learnable? }
   ```

#### 推奨

**現在の構文（後置）を維持することを推奨**

理由：
- PEG文法との相性
- オプション性の表現
- パーサーの実装がシンプル

---

### 5.2 処理系内部でのメリット

`learnable` キーワードは**非常に重要な最適化ヒント**です。

#### 5.2.1 メモリ効率化

```rust
// learnable なテンソル
tensor w: float16[1000, 1000] learnable
→ requires_grad = true
→ メモリ確保:
   - データ: 1000×1000×2 bytes = 2MB
   - 勾配: 1000×1000×2 bytes = 2MB  ← 勾配用に追加メモリ確保
   - 計算グラフノード
   合計: 約 4MB

// learnable でない テンソル
tensor x: float16[1000, 1000]
→ requires_grad = false
→ メモリ確保:
   - データ: 1000×1000×2 bytes = 2MB のみ
   合計: 2MB  ← 勾配不要なので半分！
```

**実例**:
```tensorlogic
// 学習パラメータ（少数）
tensor w: float16[10] learnable      // 勾配バッファ確保
tensor b: float16[1] learnable       // 勾配バッファ確保

// 入力データ（大量）
tensor x_train: float16[10000, 10]   // 勾配バッファ不要！
tensor y_train: float16[10000, 1]    // 勾配バッファ不要！
```

**メモリ比較**:
```
learnable ありの場合: 22 bytes + 22 bytes (勾配) = 44 bytes
learnable なしの場合: 220KB + 220KB (勾配) = 440KB ← 1万倍！
```

---

#### 5.2.2 計算グラフの構築効率

**実装箇所**: [src/tensor/tensor.rs:171-176](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/tensor/tensor.rs#L171-L176)

```rust
pub fn set_requires_grad(&mut self, requires: bool) {
    self.requires_grad = requires;

    // Allocate a node ID for this tensor if requires_grad is true
    if requires && self.grad_node.is_none() {
        let node_id = AutogradContext::allocate_id();
        self.grad_node = Some(node_id);
    }
}
```

**計算グラフの例**:
```
learnable tensor w
    ↓
  w * x  ← この演算を記録（逆伝播用）
    ↓
  loss
    ↓
backward() ← w への勾配を計算
```

**learnable でないテンソルは計算グラフから除外**:
```
non-learnable tensor x
    ↓
  w * x  ← x への勾配は計算しない（不要）
    ↓
  loss
```

---

#### 5.2.3 GPU メモリ割り当ての最適化

**全てのテンソルは Metal GPU に作成されますが、扱いが異なります**

```rust
// 全てのテンソルは Metal GPU に作成される
let device = self.env.metal_device();
let tensor = Tensor::new(device, shape, data)?;

// ただし learnable な テンソルは特別扱い
if learnable == LearnableStatus::Learnable {
    tensor.set_requires_grad(true);  // 勾配計算を有効化
    // → GPU上に勾配バッファも確保
}
```

**メモリレイアウト（GPU上）**:

```
GPU Memory:

[learnable tensor w]
  ├─ データバッファ (Metal Buffer)
  └─ 勾配バッファ (Metal Buffer)  ← learnable のみ

[通常の tensor x]
  └─ データバッファ (Metal Buffer) のみ
```

---

#### 5.2.4 最適化器への登録

**実装箇所**: [src/interpreter/mod.rs:4439](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/interpreter/mod.rs#L4439)

```rust
// Collect parameter tensors
let params: Vec<Tensor> = learnable_params.iter()
    .map(|(_, t)| t.clone())
    .collect();

// Create optimizer based on spec
let mut opt: Box<dyn Optimizer> = match spec.optimizer.name.as_str() {
    "sgd" => Box::new(SGD::new(params.clone(), lr)),
    "adam" => Box::new(Adam::new(params.clone(), lr)),
    //...
};
```

**最適化器は learnable なパラメータのみを管理**:
```
Optimizer (Adam)
  ├─ w: 学習率、モーメンタム、二次モーメント を管理
  ├─ b: 学習率、モーメンタム、二次モーメント を管理
  └─ (x は含まれない！)
```

---

#### 5.2.5 逆伝播の効率化

**実装箇所**: [src/tensor/tensor.rs:373-377](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/tensor/tensor.rs#L373-L377)

```rust
pub fn backward(&mut self) -> TensorResult<()> {
    if !self.requires_grad {
        return Err(TensorError::InvalidOperation(
            "Cannot call backward on tensor with requires_grad=False".to_string(),
        ));
    }
    // 勾配計算を実行...
}
```

**計算例**:
```tensorlogic
tensor w: float16[10] learnable
tensor x: float16[10]

main {
    pred := w * x
    loss := sum(pred * pred)

    // backward を呼ぶと:
    // - w の勾配は計算される ✅
    // - x の勾配は計算されない ❌（不要なので）
}
```

---

### 5.3 まとめ表

| 項目 | learnable あり | learnable なし |
|------|---------------|--------------|
| **メモリ** | データ + 勾配バッファ | データのみ |
| **計算グラフ** | ノード作成 | ノード不作成 |
| **GPU割り当て** | データ + 勾配 バッファ両方 | データバッファのみ |
| **最適化器** | 登録される | 登録されない |
| **逆伝播** | 勾配計算される | 勾配計算されない |
| **更新** | パラメータ更新される | 固定値 |

---

## 6. まとめ

### 6.1 学習データの種類

**テンソル学習**:
- 学習パラメータ（`learnable`）
- 入力データ
- 正解ラベル
- 損失関数

**Neural-Symbolic学習**:
- 正例ファクト
- 負例ファクト
- エンティティ埋め込み
- 関係埋め込み
- 論理的制約

### 6.2 重要な関数

| 関数 | 役割 |
|------|------|
| **スコア関数** | ファクトの確信度を計算 |
| **ランキング損失** | 正例と負例の差を評価 |
| **制約損失** | 論理的制約の違反を評価 |
| **類似度関数** | ベクトル間の類似性を測定 |

### 6.3 learnブロックの特徴

- ✅ 隠されたループ（`epochs`回の反復）
- ✅ 各エポックで文を再実行
- ✅ 自動的に勾配計算とパラメータ更新

### 6.4 learnableキーワードの重要性

**最適化効果**:
1. メモリ使用量を最大50%削減
2. 計算グラフを最小化
3. GPU メモリを効率的に利用
4. 最適化器の管理を最適化
5. 逆伝播の計算を最小化

**推奨事項**:
- ✅ 学習パラメータには必ず `learnable` を付ける
- ✅ 入力データには付けない
- ✅ 後置構文を維持（現在の実装）

---

**次のステップ**:
- [論理プログラミング統合の文法設計](https://github.com/JunSuzukiJapan/tensorlogic)（開発中）
- [言語リファレンス](language_reference.md)
