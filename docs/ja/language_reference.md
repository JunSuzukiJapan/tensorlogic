# TensorLogic 言語リファレンス

**バージョン**: 0.2.0-alpha
**最終更新**: 2025-10-20

## 目次

1. [はじめに](#はじめに)
2. [プログラム構造](#プログラム構造)
3. [データ型](#データ型)
4. [宣言](#宣言)
5. [式](#式)
6. [文](#文)
7. [演算子](#演算子)
8. [組み込み関数](#組み込み関数)
9. [学習システム](#学習システム)
10. [論理プログラミング](#論理プログラミング)

---

## 1. はじめに

TensorLogicは、テンソル代数と論理プログラミングを統合したプログラミング言語で、ニューラル・シンボリックAIを実現します。微分可能なテンソル演算と論理的推論を組み合わせ、次世代のAIシステムを構築できます。

### 主な機能

- **テンソル演算**: GPU加速による高性能計算
- **自動微分**: 組み込みの勾配計算
- **学習システム**: 複数の最適化アルゴリズム対応
- **論理プログラミング**: 関係、ルール、クエリ
- **ニューラル・シンボリック統合**: エンティティと関係の埋め込み

---

## 2. プログラム構造

### 2.1 基本構造

```tensorlogic
// 宣言
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// メイン実行ブロック
main {
    // 文
    result := w * w

    // 学習
    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

### 2.2 外部ファイルのインポート

TensorLogicは外部ファイルから宣言をインポートできます:

```tensorlogic
// 別のファイルから宣言をインポート
import "path/to/module.tl"
import "../lib/constants.tl"

main {
    // インポートしたテンソルと関数を使用
    result := imported_tensor * 2
}
```

**機能**:
- 相対パス解決（インポートするファイルからの相対パス）
- 循環依存検出（無限ループを防止）
- 重複インポート防止（同じファイルは2回インポートされない）
- 宣言のみがインポートされる（メインブロックは実行されない）

**例**:

ファイル: `lib/constants.tl`
```tensorlogic
tensor pi: float16[1] = [3.14159]
tensor e: float16[1] = [2.71828]
```

ファイル: `main.tl`
```tensorlogic
import "lib/constants.tl"

main {
    tensor circumference: float16[1] = [2.0]
    result := circumference * pi  // インポートしたpiを使用
    print("結果:", result)
}
```

### 2.3 コメント

```tensorlogic
// 単一行コメント

/* 複数行コメント
   未実装 */
```

---

## 3. データ型

### 3.1 基本型

| 型 | 説明 | 精度 |
|------|-------------|-----------|
| `float16` | 16ビット浮動小数点（f16） | 半精度（Apple Silicon最適化） |
| `float32` | 32ビット浮動小数点 | 単精度 |
| `float64` | 64ビット浮動小数点 | 倍精度 |
| `int32` | 32ビット整数 | 符号付き整数 |
| `int64` | 64ビット整数 | 符号付き長整数 |
| `bool` | 真偽値 | true/false |
| `complex64` | 64ビット複素数 | 複素float32 |

**注意**: TensorLogicは、Apple Silicon（Metal GPUおよびNeural Engine）での最適なパフォーマンスのために主に`float16`を使用します。

### 3.1.1 数値リテラル

TensorLogicは正負の数値リテラルをサポートします:

```tensorlogic
tensor positive: float16[1] = [3.14]
tensor negative: float16[1] = [-2.71]
tensor zero: float16[1] = [0.0]
tensor neg_int: float16[1] = [-42.0]
```

### 3.2 テンソル型

```tensorlogic
tensor x: float32[10]           // 1次元テンソル
tensor W: float32[3, 4]         // 2次元行列
tensor T: float32[2, 3, 4]      // 3次元テンソル
tensor D: float32[n, m]         // 可変次元
tensor F: float32[?, ?]         // 動的次元
```

### 3.3 エンティティ型

```tensorlogic
entity      // 論理プログラミングエンティティ
concept     // 高レベル概念
```

---

## 4. 宣言

### 4.1 テンソル宣言

#### 基本構文

```tensorlogic
tensor 名前: 型[次元] 属性 = 初期値
```

#### 例

```tensorlogic
// シンプルな宣言
tensor x: float32[5]

// 初期化あり
tensor w: float32[3] = [1.0, 2.0, 3.0]

// 学習可能パラメータ
tensor w: float32[10] learnable

// 初期値と学習可能
tensor b: float32[1] learnable = [0.0]

// 凍結（訓練不可）
tensor const: float32[5] frozen = [1.0, 1.0, 1.0, 1.0, 1.0]
```

### 4.2 関係宣言

```tensorlogic
// 基本関係
relation Parent(x: entity, y: entity)

// 埋め込み付き関係
relation Friend(x: entity, y: entity) embed float32[64]

// 多パラメータ関係
relation WorksAt(person: entity, company: entity, role: concept)
```

### 4.3 埋め込み宣言

```tensorlogic
// 明示的エンティティ集合
embedding person_embed {
    entities: {alice, bob, charlie}
    dimension: 64
    init: xavier
}

// 自動エンティティ集合（動的）
embedding word_embed {
    entities: auto
    dimension: 128
    init: random
}
```

#### 初期化方法

| 方法 | 説明 | 用途 |
|--------|-------------|----------|
| `random` | 一様分布(-0.1, 0.1) | 汎用 |
| `xavier` | Xavier/Glorot初期化 | 深層ネットワーク |
| `he` | He初期化 | ReLUネットワーク |
| `zeros` | 全てゼロ | バイアス初期化 |
| `ones` | 全て1 | 特殊ケース |

---

## 5. 式

### 5.1 テンソルリテラル

```tensorlogic
[1.0, 2.0, 3.0]                // 1次元配列
[[1.0, 2.0], [3.0, 4.0]]       // 2次元行列（将来）
```

### 5.2 変数

```tensorlogic
x                               // 変数参照
w                               // テンソル変数
```

### 5.3 二項演算

```tensorlogic
a + b                          // 要素ごとの加算
a - b                          // 要素ごとの減算
a * b                          // 要素ごとの乗算
a / b                          // 要素ごとの除算
a @ b                          // 行列乗算
a ** b                         // べき乗
```

### 5.4 単項演算

```tensorlogic
-x                             // 否定
!x                             // 論理NOT
```

### 5.5 埋め込み参照

```tensorlogic
person_embed["alice"]          // リテラルエンティティ参照
person_embed[entity_var]       // 変数エンティティ参照
```

### 5.6 アインシュタイン総和

```tensorlogic
einsum("ij,jk->ik", A, B)      // 行列乗算
einsum("ii->", A)              // トレース
einsum("ij->ji", A)            // 転置
```

---

## 6. 文

### 6.1 代入

```tensorlogic
x := 10
result := w * w + b
```

### 6.2 テンソル方程式

```tensorlogic
y = w @ x + b                  // 厳密な等式
y ~ w @ x + b                  // 近似等式
```

### 6.3 制御フロー

#### if文

```tensorlogic
if x > 0 {
    result := x
} else {
    result := 0
}
```

#### forループ

```tensorlogic
for i in range(10) {
    sum := sum + i
}
```

#### whileループ

```tensorlogic
while x > 0 {
    x := x - 1
}
```

### 6.4 クエリ文

```tensorlogic
query Parent(alice, X)
query Parent(X, Y) where X != Y
```

### 6.5 推論文

```tensorlogic
infer forward query Ancestor(alice, X)
infer backward query HasProperty(X, color)
infer gradient query Relation(X, Y)
infer symbolic query Rule(X)
```

### 6.6 学習文

```tensorlogic
learn {
    objective: loss_expression,
    optimizer: sgd(lr: 0.1),
    epochs: 100
}
```

---

## 7. 演算子

### 7.1 算術演算子

| 演算子 | 名前 | 説明 | 例 |
|----------|------|-------------|---------|
| `+` | 加算 | 要素ごとの加算 | `a + b` |
| `-` | 減算 | 要素ごとの減算 | `a - b` |
| `*` | 乗算 | 要素ごとの乗算（アダマール積） | `a * b` |
| `/` | 除算 | 要素ごとの除算 | `a / b` |
| `@` | 行列乗算 | テンソル縮約 | `A @ B` |
| `**` | べき乗 | 要素ごとのべき乗 | `a ** 2` |

### 7.2 比較演算子

| 演算子 | 名前 | 例 |
|----------|------|---------|
| `==` | 等しい | `x == y` |
| `!=` | 等しくない | `x != y` |
| `<` | より小さい | `x < y` |
| `>` | より大きい | `x > y` |
| `<=` | 以下 | `x <= y` |
| `>=` | 以上 | `x >= y` |
| `≈` | ほぼ等しい | `x ≈ y` |

### 7.3 論理演算子

| 演算子 | 名前 | 例 |
|----------|------|---------|
| `and` | 論理AND | `a and b` |
| `or` | 論理OR | `a or b` |
| `not` | 論理NOT | `not a` |

### 7.4 演算子の優先順位

高い順に：

1. `()` - 括弧
2. `**` - べき乗
3. `-` (単項), `!` - 単項演算子
4. `@` - 行列乗算
5. `*`, `/` - 乗算、除算
6. `+`, `-` - 加算、減算
7. `<`, `>`, `<=`, `>=`, `==`, `!=`, `≈` - 比較
8. `not` - 論理NOT
9. `and` - 論理AND
10. `or` - 論理OR

---

## 8. 組み込み関数

### 8.1 テンソル演算

#### 活性化関数

```tensorlogic
relu(x)                        // ReLU活性化
gelu(x)                        // GELU活性化
sigmoid(x)                     // シグモイド（将来）
tanh(x)                        // tanh（将来）
```

#### 行列演算

```tensorlogic
transpose(A)                   // 転置（将来）
inverse(A)                     // 逆行列（将来）
determinant(A)                 // 行列式（将来）
```

#### 縮約演算

```tensorlogic
sum(x)                         // 全要素の和
mean(x)                        // 平均値（将来）
max(x)                         // 最大値（将来）
min(x)                         // 最小値（将来）
```

### 8.2 形状演算

```tensorlogic
shape(x)                       // テンソル形状取得（制約）
rank(x)                        // テンソルランク取得（制約）
```

### 8.3 アインシュタイン総和

```tensorlogic
einsum(equation, tensors...)   // アインシュタイン総和記法
```

**例**:

```tensorlogic
// 行列乗算
C := einsum("ij,jk->ik", A, B)

// トレース
trace := einsum("ii->", A)

// 転置
B := einsum("ij->ji", A)

// バッチ行列乗算
C := einsum("bij,bjk->bik", A, B)
```

---

## 9. 学習システム

### 9.1 学習可能パラメータ

```tensorlogic
tensor w: float32[10] learnable = [...]
tensor b: float32[1] learnable = [0.0]
```

### 9.2 最適化アルゴリズム

#### SGD（確率的勾配降下法）

```tensorlogic
optimizer: sgd(lr: 0.1)
```

**パラメータ**:
- `lr`: 学習率（デフォルト: 0.01）

#### Adam

```tensorlogic
optimizer: adam(lr: 0.001)
```

**パラメータ**:
- `lr`: 学習率（デフォルト: 0.001）
- `beta1`: 一次モーメント減衰（デフォルト: 0.9）
- `beta2`: 二次モーメント減衰（デフォルト: 0.999）
- `epsilon`: 小さな定数（デフォルト: 1e-8）

#### AdamW

```tensorlogic
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

**パラメータ**:
- `lr`: 学習率（デフォルト: 0.001）
- `weight_decay`: 重み減衰係数（デフォルト: 0.01）

### 9.3 学習仕様

```tensorlogic
learn {
    // オプション: 中間計算のためのローカル変数宣言
    intermediate := some_expression
    another_var := other_expression

    objective: loss_expression,
    optimizer: optimizer_spec,
    epochs: number
}
```

**要件**:
- `objective`はスカラーテンソル式である必要があります
- `learnable`マークされた全テンソルが最適化されます
- 勾配は自動微分により計算されます
- `objective`の前に`:=`でローカル変数を中間計算に使用できます
- ローカル変数は各エポックで再計算されます

**例 - 基本的な学習**:

```tensorlogic
tensor w: float32[10] learnable = [...]
tensor x: float32[10] = [...]

main {
    pred := w * x
    loss := pred * pred

    learn {
        objective: loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }
}
```

**例 - ローカル変数を使用**:

```tensorlogic
tensor W: float16[1] learnable = [0.5]
tensor x1: float16[1] = [1.0]
tensor y1: float16[1] = [3.0]
tensor x2: float16[1] = [-2.0]  // 負の数がサポートされています
tensor y2: float16[1] = [-6.0]

main {
    learn {
        // 中間計算のためのローカル変数
        pred1 := x1 * W
        pred2 := x2 * W

        // 誤差を計算
        err1 := pred1 - y1
        err2 := pred2 - y2

        // 二乗誤差の合計
        total_loss := err1 * err1 + err2 * err2

        objective: total_loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }

    print("学習したW:", W)  // 3.0に近い値になるはず
}
```

**注意**: `learnable`キーワードで明示的に宣言されたテンソルのみが最適化されます。`learn`ブロック内で計算されたローカル変数は学習可能パラメータとして扱われません。

---

## 10. 論理プログラミング

### 10.1 関係

```tensorlogic
relation Parent(x: entity, y: entity)
relation Sibling(x: entity, y: entity)
```

### 10.2 ルール（将来）

```tensorlogic
rule Grandparent(X, Z) <- Parent(X, Y), Parent(Y, Z)
rule Ancestor(X, Y) <- Parent(X, Y)
rule Ancestor(X, Z) <- Parent(X, Y), Ancestor(Y, Z)
```

### 10.3 クエリ（部分実装）

```tensorlogic
query Parent(alice, X)
query Parent(X, Y) where X != Y
```

### 10.4 制約

```tensorlogic
shape(w) == [10, 20]           // 形状制約
rank(A) == 2                   // ランク制約
norm(w) < 1.0                  // ノルム制約
```

### 10.5 推論方法

```tensorlogic
infer forward query Q          // 論理 → テンソル
infer backward query Q         // テンソル → 論理
infer gradient query Q         // 微分的推論
infer symbolic query Q         // 記号操作
```

---

## 付録A: 予約語

```
and, auto, bool, complex64, concept, dimension, einsum, else, embed, embedding,
entities, entity, epochs, float32, float64, for, frozen, function, gelu, if,
in, init, int32, int64, infer, learn, learnable, main, norm, not, objective,
ones, optimizer, or, query, random, range, rank, relation, relu, rule, shape,
tensor, where, while, xavier, zeros
```

## 付録B: 文法要約

完全なBNF仕様については[TensorLogic文法](../../Papers/実装/tensorlogic_grammar.md)を参照してください。

## 付録C: 例

実際の例については[チュートリアル01: 線形回帰](../../claudedocs/tutorial_01_linear_regression.md)および他のチュートリアルを参照してください。

---

**言語リファレンス終わり**

質問や貢献については: https://github.com/JunSuzukiJapan/tensorlogic
