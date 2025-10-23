# 論理プログラミングとテンソルの統一設計

**対象読者**: TensorLogicの設計思想を理解したい開発者・研究者
**最終更新**: 2025-10-23

## 目次

1. [概要と目的](#1-概要と目的)
2. [論理プログラミング要素とテンソル要素の対応関係](#2-論理プログラミング要素とテンソル要素の対応関係)
3. [アインシュタイン和が重要な理由](#3-アインシュタイン和が重要な理由)
4. [統一文法の提案](#4-統一文法の提案)
5. [エンティティ型の設計](#5-エンティティ型の設計)
6. [学習と推論の統合方法](#6-学習と推論の統合方法)
7. [実装上の考慮点](#7-実装上の考慮点)
8. [まとめ](#8-まとめ)

---

## 1. 概要と目的

### 1.1 背景

TensorLogicは、論理プログラミングと深層学習を統合したニューロシンボリックAI言語です。従来、これらは別々のパラダイムとして扱われてきました：

| パラダイム | 強み | 弱み |
|-----------|------|------|
| **論理プログラミング** | 推論が正確、説明可能 | 学習できない、不確実性に弱い |
| **深層学習** | 学習能力が高い、柔軟 | ブラックボックス、論理推論が苦手 |

### 1.2 統一の目的

両者を**テンソル演算**という共通基盤で統一することで：

- ✅ 論理推論の正確性と説明可能性
- ✅ 深層学習の学習能力と柔軟性
- ✅ GPU加速による高速計算
- ✅ 統一された簡潔な構文

を同時に実現します。

### 1.3 設計方針

本ドキュメントの設計方針：

- **簡潔性優先**: なるべく短く簡潔に書ける文法
- **型安全性**: コンパイル時にエラーを検出
- **自動最適化**: 型情報から最適な演算を自動生成
- **内部統一**: すべての論理演算をテンソルで表現

---

## 2. 論理プログラミング要素とテンソル要素の対応関係

### 2.1 基本的な対応

論理プログラミングの3要素とテンソル表現の対応：

| 論理プログラミング | テンソル表現 | 説明 |
|------------------|------------|------|
| **ファクト** | テンソルの要素 | 真理値をテンソル要素として表現 |
| **ルール** | アインシュタイン和 | 論理結合をテンソル積として表現 |
| **クエリ** | インデックス/スライス | 変数をテンソルの次元として表現 |

### 2.2 ファクト → テンソル要素

```tensorlogic
// 論理プログラミング
Parent(alice, bob)       // 真
Parent(alice, charlie)   // 真
Parent(bob, diana)       // 真

// テンソル表現
tensor parent: float16[4, 4] = [
    //        alice  bob  charlie diana
    /* alice */  [0,    1,    1,      0],
    /* bob   */  [0,    0,    0,      1],
    /* charlie*/ [0,    0,    0,      0],
    /* diana */  [0,    0,    0,      0]
]
```

**対応関係**:
- ファクト `Parent(alice, bob)` → テンソル要素 `parent[0, 1] = 1.0`
- ファクト `Parent(alice, charlie)` → テンソル要素 `parent[0, 2] = 1.0`
- エンティティ名は整数インデックスにマッピング

### 2.3 ルール → アインシュタイン和

```tensorlogic
// 論理プログラミング
Grandparent(X, Z) :- Parent(X, Y), Parent(Y, Z)

// テンソル表現（アインシュタイン和）
grandparent := einsum('xy,yz->xz', parent, parent)
//              ↑       ↑   ↑   ↑
//              |       X   Y   Z
//              |       共通変数Yで結合される
//              演算
```

**対応関係**:
- 変数 `X, Y, Z` → テンソルのインデックス `x, y, z`
- 共通変数 `Y` → 縮約される次元 `y`
- カンマ `,` → テンソル積
- `:-` → 代入 `:=`

### 2.4 クエリ → インデックス/スライス

```tensorlogic
// 論理プログラミング
Parent(alice, X)?        // aliceの子供は誰？

// テンソル表現
result := parent[alice_id, :]
// → [0, 1, 1, 0]
// → bob (インデックス1) と charlie (インデックス2) が解
```

**対応関係**:
- 定数 `alice` → 固定インデックス `alice_id`
- 変数 `X` → スライス `:`
- 結果 → 非ゼロ要素のインデックス

---

## 3. アインシュタイン和が重要な理由

### 3.1 変数バインディングとの対応

論理プログラミングの変数バインディングは、アインシュタイン和のインデックス共有と**完全に対応**します。

```
論理プログラミング:
  Ancestor(X, Z) :- Parent(X, Y), Parent(Y, Z)
                           ↑         ↑
                           Y が共通変数（結合点）

アインシュタイン和:
  einsum('xy, yz -> xz', parent, parent)
              ↑   ↑
              y が共通インデックス（縮約される）
```

### 3.2 具体例：祖父母関係の計算

#### データ

```tensorlogic
tensor parent: float16[4, 4] = [
    [0, 1, 1, 0],  // alice → bob, charlie
    [0, 0, 0, 1],  // bob → diana
    [0, 0, 0, 1],  // charlie → diana
    [0, 0, 0, 0]   // diana → (none)
]
```

#### ルール適用

```tensorlogic
// Grandparent(X, Z) :- Parent(X, Y), Parent(Y, Z)
grandparent := einsum('xy,yz->xz', parent, parent)
```

#### 計算過程（alice, diana の例）

```
grandparent[alice, diana] = Σ(parent[alice, Y] * parent[Y, diana])
                          = parent[alice,alice]*parent[alice,diana]
                          + parent[alice,bob]*parent[bob,diana]      ← 1*1=1
                          + parent[alice,charlie]*parent[charlie,diana] ← 1*1=1
                          + parent[alice,diana]*parent[diana,diana]
                          = 0*0 + 1*1 + 1*1 + 0*0
                          = 2  ← 2つの経路がある！
```

#### 結果

```tensorlogic
grandparent = [
    [0, 0, 0, 2],  // alice → diana (bob経由とcharlie経由)
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
```

### 3.3 複雑なルールでの威力

#### 3つ以上の関係の結合

```tensorlogic
// Friend(X, Y), Likes(Y, Z), Recommends(Z, W) → Suggestion(X, W)

suggestion := einsum('xy,yz,zw->xw', friend, likes, recommends)
//                     ↑   ↑   ↑
//                     y   z が自動的に結合される
```

**利点**:
- 複数の共通変数を一度に処理
- GPU最適化されたeinsum演算を活用
- 並列計算が容易

#### 複数の論理パスの統合

```tensorlogic
// ルール1: 直接の友達
// ルール2: 友達の友達
// ルール3: 共通の趣味を持つ

rule Connected(x, y) :-
    Friend(x, y)

rule Connected(x, z) :-
    Friend(x, y), Friend(y, z)

rule Connected(x, y) :-
    SharedHobby(x, y)

// テンソル計算
connected := Friend
           + einsum('xy,yz->xz', Friend, Friend)  // 推移性
           + SharedHobby                          // 別の根拠
```

### 3.4 アインシュタイン和の利点まとめ

| 利点 | 説明 |
|------|------|
| **自然な対応** | 論理変数 ↔ テンソルインデックス |
| **自動最適化** | GPUカーネルで最適化済み |
| **並列計算** | 複数の結合を同時に処理 |
| **微分可能** | 学習との統合が容易 |
| **簡潔な表現** | 複雑なルールも短く記述 |

---

## 4. 統一文法の提案

### 4.1 提案の評価基準

各提案を以下の基準で評価します：

- **簡潔性**: コードの短さ、書きやすさ
- **明確性**: 意図の分かりやすさ
- **型安全性**: コンパイル時エラー検出
- **自動最適化**: コンパイラによる最適化の可能性
- **実装の容易さ**: パーサーとコンパイラの実装難易度

---

### 4.2 提案1: 明示的テンソルルール構文

```tensorlogic
// 関係をテンソルとして宣言
relation Parent(x: entity, y: entity) as tensor float16[N, N]

// ルールをアインシュタイン和で記述
rule Grandparent(x: entity, z: entity) {
    // 論理的な定義
    Parent(x, y), Parent(y, z)

    // テンソル計算（自動生成も可能）
    compute: einsum('xy,yz->xz', Parent, Parent)
}

// クエリ
main {
    // 通常のクエリ
    Grandparent(alice, X)?

    // テンソルとしてアクセス
    result := Grandparent[alice_id, :]
}
```

**評価**:

| 基準 | 評価 | 理由 |
|------|------|------|
| 簡潔性 | ⭐⭐⭐ | やや冗長 |
| 明確性 | ⭐⭐⭐⭐⭐ | 論理とテンソルが明確に分離 |
| 型安全性 | ⭐⭐⭐⭐ | 型を明示 |
| 自動最適化 | ⭐⭐⭐⭐ | 最適化ヒントを提供 |
| 実装の容易さ | ⭐⭐⭐⭐ | 比較的シンプル |

**利点**:
- ✅ 論理的定義とテンソル計算を分離
- ✅ 既存の論理プログラミング構文を維持
- ✅ 最適化のためのヒントを明示可能

**欠点**:
- ❌ 冗長（computeブロックが必要）
- ❌ 同じ情報を2回書く（論理とテンソル）

---

### 4.3 提案2: 暗黙的変換構文

```tensorlogic
// 関係宣言（テンソルバック）
relation Parent(x: entity, y: entity) tensor

// ルール：論理的に書くだけで自動的にeinsumに変換
rule Grandparent(x: entity, z: entity) :-
    Parent(x, y), Parent(y, z)
    // 自動的に: einsum('xy,yz->xz', Parent, Parent)

rule Ancestor(x: entity, z: entity) :-
    Parent(x, z)  // ベースケース
rule Ancestor(x: entity, z: entity) :-
    Parent(x, y), Ancestor(y, z)  // 再帰ケース
    // 自動的に反復計算

// クエリはそのまま
main {
    Ancestor(alice, X)?
    // 内部的には: ancestor[alice_id, :] でテンソルスライス
}
```

**変数から次元への自動マッピング**:
```
Parent(x, y)          → インデックス 'xy'
Parent(y, z)          → インデックス 'yz'
共通変数 y            → 縮約される次元
結果 Grandparent(x,z) → インデックス 'xz'
```

**評価**:

| 基準 | 評価 | 理由 |
|------|------|------|
| 簡潔性 | ⭐⭐⭐⭐⭐ | 最も簡潔 |
| 明確性 | ⭐⭐⭐⭐ | 論理プログラミングそのまま |
| 型安全性 | ⭐⭐⭐ | 型推論に依存 |
| 自動最適化 | ⭐⭐⭐⭐⭐ | コンパイラが完全に制御 |
| 実装の容易さ | ⭐⭐ | 高度なコンパイラが必要 |

**利点**:
- ✅ 既存の論理プログラミング構文そのまま
- ✅ 最も自然な統合
- ✅ コンパイラが自動的にeinsum生成

**欠点**:
- ❌ 高度なコンパイラが必要
- ❌ 最適化のヒントを与えにくい

---

### 4.4 提案3: ハイブリッド構文

```tensorlogic
// 基本的な関係
relation Parent(x: entity, y: entity) tensor

// シンプルなルール：自動変換
rule Grandparent(x, z) :- Parent(x, y), Parent(y, z)

// 複雑なルール：明示的に指定
rule WeightedInfluence(x: entity, w: entity) {
    logic: Friend(x, y), Influence(y, z), Recommends(z, w)

    // カスタム計算（重み付き）
    compute {
        friend_weight := Friend * 0.5
        influence_weight := Influence * 0.3
        recommend_weight := Recommends * 0.2

        result := einsum('xy,yz,zw->xw',
                        friend_weight,
                        influence_weight,
                        recommend_weight)
    }
}

// ファジールール（スコア付き）
rule Similar(x: entity, y: entity) score {
    // 複数の根拠から総合スコアを計算
    let friend_score = Friend(x, y)              // 0.8
    let hobby_score = SharedHobby(x, y)          // 0.6
    let location_score = SameLocation(x, y)      // 0.9

    // 加重平均
    return 0.4 * friend_score + 0.3 * hobby_score + 0.3 * location_score
}
```

**評価**:

| 基準 | 評価 | 理由 |
|------|------|------|
| 簡潔性 | ⭐⭐⭐⭐ | シンプルケースは簡潔 |
| 明確性 | ⭐⭐⭐⭐ | 意図が明確 |
| 型安全性 | ⭐⭐⭐⭐ | 型チェック可能 |
| 自動最適化 | ⭐⭐⭐ | シンプルケースのみ |
| 実装の容易さ | ⭐⭐⭐ | 中程度 |

**利点**:
- ✅ シンプルなケースは簡潔
- ✅ 複雑なケースは明示的にコントロール
- ✅ ファジー論理との統合

**欠点**:
- ❌ 2つのモードを学ぶ必要
- ❌ 一貫性がやや低い

---

### 4.5 提案4: 型付きアインシュタイン構文（推奨）

```tensorlogic
// エンティティ型を次元として扱う
entity Person = {alice, bob, charlie, diana}
entity Item = {book, movie, game}

// 関係の型シグネチャが自動的にテンソル形状を決定
relation Parent(x: Person, y: Person)
// 自動的に: tensor float16[|Person|, |Person|]
//                           ↑          ↑
//                           4          4

relation Likes(x: Person, y: Item)
// 自動的に: tensor float16[|Person|, |Item|]
//                           ↑        ↑
//                           4        3

// ルール：変数の型から自動的にeinsum生成
rule Recommends(x: Person, z: Item) :-
    Friend(x, y),      // Person × Person
    Likes(y, z)        // Person × Item
    // 型から自動推論:
    // Friend: [Person, Person] → インデックス 'ab'
    // Likes:  [Person, Item]   → インデックス 'bc'
    // Result: [Person, Item]   → インデックス 'ac'
    // einsum: 'ab,bc->ac' (Personの2つ目が共通)

// クエリも型安全
main {
    Recommends(alice, X)?
    // X: Item 型 → bookかmovieかgameだけが返る
}
```

**型推論の流れ**:
```
1. Friend(x, y) where x: Person, y: Person
   → Friend テンソル形状: [Person, Person]

2. Likes(y, z) where y: Person, z: Item
   → Likes テンソル形状: [Person, Item]

3. 共通変数 y: Person
   → Friend の第2次元と Likes の第1次元を結合

4. 結果 Recommends(x, z) where x: Person, z: Item
   → Recommends テンソル形状: [Person, Item]

5. einsum 生成: 'ab,bc->ac'
```

**評価**:

| 基準 | 評価 | 理由 |
|------|------|------|
| 簡潔性 | ⭐⭐⭐⭐⭐ | エンティティ定義だけで形状決定 |
| 明確性 | ⭐⭐⭐⭐⭐ | 型が意図を明示 |
| 型安全性 | ⭐⭐⭐⭐⭐ | コンパイル時にエラー検出 |
| 自動最適化 | ⭐⭐⭐⭐⭐ | 型情報から最適なeinsum生成 |
| 実装の容易さ | ⭐⭐⭐ | 型推論エンジンが必要 |

**利点**:
- ✅ **最も簡潔**: エンティティ定義だけでテンソル形状が決まる
- ✅ **型安全**: コンパイル時にエラー検出
- ✅ **自動最適化**: 型情報からeinsum自動生成
- ✅ **GPU効率**: 事前にメモリ確保、最適な演算順序

**欠点**:
- ⚠️ 型推論エンジンの実装が必要
- ⚠️ エンティティ型の学習が必要

---

### 4.6 提案5: 宣言的クエリ with einsum

```tensorlogic
relation Parent(x: entity, y: entity) tensor
relation Friend(x: entity, y: entity) tensor

main {
    // 標準的なクエリ
    infer forward Parent(alice, X)?

    // einsumクエリ（複雑な推論）
    infer {
        // 「aliceの友達の親」を一発で計算
        result := query einsum('xy,yz->xz', Friend, Parent)[alice_id, :]
    }

    // 再帰的クエリ（到達可能性）
    infer {
        // 推移閉包を計算
        ancestor := Parent
        for i in 1..10 {
            ancestor := ancestor + einsum('xy,yz->xz', ancestor, Parent)
        }

        // aliceから到達可能なすべてのノード
        result := ancestor[alice_id, :]
    }
}
```

**評価**:

| 基準 | 評価 | 理由 |
|------|------|------|
| 簡潔性 | ⭐⭐⭐ | 複雑なクエリ向け |
| 明確性 | ⭐⭐⭐ | 手続き的すぎる |
| 型安全性 | ⭐⭐ | チェックしにくい |
| 自動最適化 | ⭐⭐ | ユーザーが制御 |
| 実装の容易さ | ⭐⭐⭐⭐ | 比較的簡単 |

**利点**:
- ✅ 複雑なクエリを効率的に表現
- ✅ 反復計算との統合
- ✅ 柔軟性が高い

**欠点**:
- ❌ 論理プログラミングらしさが薄れる
- ❌ 手続き的になりすぎる

---

### 4.7 提案の比較表

| 提案 | 簡潔性 | 明確性 | 型安全性 | 自動最適化 | 実装 | 総合 |
|------|--------|--------|---------|-----------|------|------|
| 1. 明示的テンソルルール | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 2. 暗黙的変換 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 3. ハイブリッド | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **4. 型付きアインシュタイン** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐** | **⭐⭐⭐⭐⭐** |
| 5. 宣言的クエリ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 4.8 推奨案

**提案4: 型付きアインシュタイン構文**を推奨します。

理由：
1. **最も簡潔**: エンティティ定義だけで自動的にテンソル形状決定
2. **型安全**: コンパイル時にエラー検出
3. **自動最適化**: 型情報から最適なeinsum生成
4. **設計方針に合致**: 簡潔性、型安全性、自動最適化のすべてを満たす

---

## 5. エンティティ型の設計

### 5.1 エンティティ型とは

エンティティ型は、**論理プログラミングのドメイン**と**テンソルの次元**を統一する概念です。

```tensorlogic
entity Person = {alice, bob, charlie, diana}
//     ↑       ↑
//     型名     値の集合（4要素）

// この定義により：
// - Person型の変数は alice, bob, charlie, diana のいずれか
// - Person型を使う関係のテンソル次元は 4
```

### 5.2 列挙型との類似性と違い

#### 類似点

```rust
// Rustの列挙型
enum Person {
    Alice,
    Bob,
    Charlie,
    Diana,
}

// TensorLogicのエンティティ型
entity Person = {alice, bob, charlie, diana}
```

両方とも：
- ✅ 有限の値の集合
- ✅ コンパイル時に決定可能
- ✅ 内部的に整数インデックスにマッピング

#### 重要な違い

| 観点 | 列挙型 | エンティティ型 |
|------|--------|--------------|
| **用途** | 状態・カテゴリ | **テンソル次元の定義** |
| **演算** | 比較・パターンマッチ | **テンソル演算の形状決定** |
| **サイズ** | 型定義で固定 | 型定義 or データから決定 |

**エンティティ型 = 列挙型 + テンソル次元定義**

### 5.3 静的 vs 動的エンティティ型

#### 静的エンティティ型（推奨）

```tensorlogic
entity Person = {alice, bob, charlie, diana}
// コンパイル時に固定、サイズ4
```

**利点**:
- ✅ メモリを事前確保（高速）
- ✅ GPU最適化が容易
- ✅ バグが少ない

**欠点**:
- ❌ 実行時にエンティティ追加不可

#### 動的エンティティ型（非推奨）

```tensorlogic
entity Person  // サイズ未定義

main {
    // 実行時に追加
    Person.add("alice")
    Person.add("eve")

    // テンソルは動的にリサイズ
}
```

**利点**:
- ✅ 柔軟性が高い

**欠点**:
- ❌ テンソルの再確保が必要（遅い）
- ❌ GPU メモリ管理が複雑
- ❌ バグの温床

**結論**: 動的型は不要。静的型のみをサポート。

### 5.4 データからの自動構築

静的型の柔軟性を高めるために、**データからの自動構築**をサポートします。

エンティティ定義で明示的な値の列挙を省略した場合、データから自動的に構築されます。

```tensorlogic
entity Person
//     ↑
//     データから自動的にエンティティ集合を構築（= が省略されている）

main {
    // ファクトからエンティティを抽出
    Friend(alice, bob)      // alice, bob を Person に追加
    Friend(bob, charlie)    // charlie を Person に追加
    Friend(charlie, diana)  // diana を Person に追加

    // この時点で Person = {alice, bob, charlie, diana} に固定
}
```

**構文**:
- `entity T = {...}` → 明示的列挙
- `entity T` → データから自動構築（`=`の有無で判別）

**利点**:
- ✅ 手動でエンティティを列挙する必要なし
- ✅ データファイルから自動的に構築可能
- ✅ 実行前に固定されるため静的型の利点を保持
- ✅ 構文が簡潔（`=`の有無だけで判別）

### 5.5 `with` ブロック構文

**問題**: `entity T` だけでは、どこでエンティティが固定されるか不明確

**解決**: `with` ブロックで明示的なスコープを提供

```tensorlogic
entity Person

main {
    // ファクト定義はwithブロック内に配置
    with Person {
        // ファクトから自動的にエンティティを抽出
        Friend(alice, bob)      // alice, bob を Person に追加
        Friend(bob, charlie)    // charlie を Person に追加
        Friend(charlie, diana)  // diana を Person に追加
    }
    // ← ここで Person 固定: {alice, bob, charlie, diana}

    // ここから先はエンティティ追加不可

    Friend(eve, alice)  // ❌ コンパイルエラー！
    // Error: Cannot add entity 'eve' to Person outside 'with' block

    learn {
        // Person は固定済み
        // tensor Friend: [4, 4]
    }
}
```

#### `with` ブロックの利点

| 利点 | 説明 |
|------|------|
| **明確なスコープ** | エンティティ収集のフェーズが視覚的に明確 |
| **安全性** | ブロック外でのエンティティ追加を禁止 |
| **複数エンティティ** | 複数のエンティティ型を独立に管理 |
| **エラー検出** | ブロック外でのエンティティ使用をコンパイル時に検出 |

#### 複数エンティティ型の管理

```tensorlogic
entity Person
entity Location
entity Item

main {
    // Phase 1: Person エンティティ収集
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }
    // Person 固定: {alice, bob, charlie}

    // Phase 2: Location エンティティ収集
    with Location {
        LivesIn(alice, tokyo)
        LivesIn(bob, osaka)
    }
    // Location 固定: {tokyo, osaka}

    // Phase 3: Item エンティティ収集
    with Item {
        Likes(alice, book)
        Likes(bob, movie)
    }
    // Item 固定: {book, movie}

    // すべてのエンティティが固定された
    // Person: [3], Location: [2], Item: [2]
}
```

### 5.6 インデックスマッピング

エンティティ名から整数インデックスへのマッピング：

```tensorlogic
entity Person

main {
    with Person {
        Friend(alice, bob)
        Friend(charlie, diana)
    }
}
```

**内部処理**:
```rust
// エンティティIDマッピング
let person_ids: HashMap<&str, usize> = hashmap! {
    "alice" => 0,
    "bob" => 1,
    "charlie" => 2,
    "diana" => 3,
};

// ファクト投入
// Friend(alice, bob) → friend_tensor[0, 1] = 1.0
// Friend(charlie, diana) → friend_tensor[2, 3] = 1.0
```

---

## 6. 学習と推論の統合方法

論理演算を学習と推論で使う方法を3つ提案します。

### 6.1 方法1: 埋め込みベース（TransEスタイル）

#### 概要

関係をベクトル埋め込みとして表現し、スコア関数で学習します。

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) embed float16[64] learnable

main {
    with Person {
        Friend(alice, bob)      // 正例
        Friend(bob, charlie)    // 正例
    }

    learn {
        // スコア関数（TransE）
        for each (s, r, o) in positive_facts {
            pos_score := -norm(embed[s] + rel_embed[r] - embed[o])
        }

        // 負例サンプリング
        for each negative in sample_negatives() {
            neg_score := -norm(embed[negative.s] + rel_embed[negative.r] - embed[negative.o])
        }

        // ランキング損失
        loss := sum(max(0, margin - pos_score + neg_score))

        objective: loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    infer {
        // 埋め込み空間でクエリ
        forward Friend(alice, X)?

        // 内部処理:
        // 1. alice の埋め込みを取得
        // 2. Friend 関係の埋め込みを取得
        // 3. alice_embed + friend_embed ≈ ? を探す
        // 4. 全エンティティとの距離を計算
        // 5. 距離が近い（スコアが高い）エンティティを返す
    }
}
```

#### 評価

| 観点 | 評価 |
|------|------|
| **実装の容易さ** | ⭐⭐⭐⭐ |
| **既存研究** | ⭐⭐⭐⭐⭐（TransE, RotatE, ComplEx等） |
| **表現力** | ⭐⭐⭐（シンプルなパターンのみ） |
| **説明可能性** | ⭐⭐（ブラックボックス的） |

**利点**:
- ✅ 実装が比較的簡単
- ✅ 既存研究が豊富（TransE, RotatE, ComplEx等）
- ✅ 大規模データに対応可能

**欠点**:
- ❌ 複雑な論理ルールを扱いにくい
- ❌ 説明可能性が低い

---

### 6.2 方法2: テンソルルールベース

#### 概要

テンソルの値自体を学習し、論理ルールを直接適用します。

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) learnable

rule Connected(x: Person, z: Person) :-
    Friend(x, y), Friend(y, z)

main {
    with Person {
        Friend(alice, bob)      // 観測
        Friend(bob, charlie)    // 観測
        // Friend(alice, charlie) = ? (未知、学習で推測)
    }

    learn {
        // ルールから予測を計算
        predicted := Connected(alice, charlie)
        // = einsum('xy,yz->xz', Friend, Friend)[alice, charlie]
        // = Friend[alice,bob] * Friend[bob,charlie]
        // = 1.0 * 1.0 = 1.0

        // 観測値（疎テンソル）
        observed := Friend[alice, charlie]  // 未知（0または学習対象）

        // 損失
        loss := (predicted - observed) ** 2

        // 正則化（テンソルの値を0-1の範囲に制約）
        regularization := sum((Friend - sigmoid(Friend)) ** 2)

        total_loss := loss + 0.1 * regularization

        objective: total_loss,
        optimizer: sgd(lr: 0.1),
        epochs: 100
    }

    infer {
        // ルールを適用して推論
        forward Connected(alice, X)?
        // ルールに基づいて計算
    }
}
```

#### 評価

| 観点 | 評価 |
|------|------|
| **実装の容易さ** | ⭐⭐⭐ |
| **既存研究** | ⭐⭐（限定的） |
| **表現力** | ⭐⭐⭐⭐（論理ルールと直接統合） |
| **説明可能性** | ⭐⭐⭐⭐（ルールベース） |

**利点**:
- ✅ 論理ルールと自然に統合
- ✅ 説明可能性が高い
- ✅ ルールベースの推論が可能

**欠点**:
- ❌ 実装が複雑
- ❌ 既存研究が少ない
- ❌ スケーラビリティに課題

---

### 6.3 方法3: ハイブリッド（推奨）

#### 概要

埋め込みベースの学習と論理ルールによる制約を組み合わせます。

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) embed float16[64] learnable

// ソフト制約としてのルール
rule Symmetric constraint :-
    Friend(x, y) <-> Friend(y, x)
    // 対称性: 友達関係は双方向

rule Transitive constraint :-
    Friend(x, y), Friend(y, z) -> Friend(x, z)
    // 推移性: 友達の友達は友達かも

main {
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }

    learn {
        // ========================================
        // データ損失（埋め込みベース）
        // ========================================
        data_loss := ranking_loss(positive_facts, negative_facts)

        // ========================================
        // 制約損失（ルールベース）
        // ========================================

        // 対称性制約
        symmetric_loss := sum(
            (score(x, Friend, y) - score(y, Friend, x)) ** 2
            for all (x,y) pairs
        )

        // 推移性制約
        transitive_loss := sum(
            max(0, min(score(x,y), score(y,z)) - score(x,z))
            for all (x,y,z) triples
        )

        // ========================================
        // 統合損失
        // ========================================
        total_loss := data_loss
                    + 0.1 * symmetric_loss
                    + 0.1 * transitive_loss

        objective: total_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    infer {
        // 埋め込みとルールの両方を使って推論
        forward Friend(alice, X)?

        // 内部処理:
        // 1. 埋め込みベースのスコア計算
        // 2. ルールベースの推論
        // 3. 両方を組み合わせて最終スコア決定
    }
}
```

#### 評価

| 観点 | 評価 |
|------|------|
| **実装の容易さ** | ⭐⭐ |
| **既存研究** | ⭐⭐⭐（Neural-Symbolic AI） |
| **表現力** | ⭐⭐⭐⭐⭐（最も柔軟） |
| **説明可能性** | ⭐⭐⭐⭐（ルールで説明可能） |

**利点**:
- ✅ 最も柔軟
- ✅ 表現力が高い
- ✅ データ駆動と知識駆動の両方を活用
- ✅ 説明可能性を保持

**欠点**:
- ❌ 最も複雑
- ❌ ハイパーパラメータ調整が必要

---

### 6.4 推奨アプローチ

#### 段階的実装

**Phase 1**: 方法1（埋め込みベース）から開始
- 実装が比較的簡単
- 既存研究の知見を活用
- 基本的な学習機能を確立

**Phase 2**: 方法3（ハイブリッド）に拡張
- ルール制約を追加
- 説明可能性を向上
- より複雑な推論をサポート

**Phase 3**（将来）: 方法2（テンソルルールベース）の研究
- 完全な統合を目指す
- 新しい研究領域

---

## 7. 実装上の考慮点

### 7.1 構文解析

#### PEG文法の拡張

```pest
// エンティティ型宣言
entity_decl = { "entity" ~ IDENT ~ ("=" ~ "{" ~ entity_list ~ "}")? }
// "=" が省略された場合、データから自動構築
// entity Person = {alice, bob}  → 明示的列挙
// entity Person                 → データから自動構築

// withブロック
with_block = { "with" ~ entity_type ~ "{" ~ statement* ~ "}" }

// 関係宣言
relation_decl = { "relation" ~ IDENT ~ "(" ~ param_list ~ ")" ~ relation_spec? }
relation_spec = {
    ("embed" ~ tensor_type ~ "learnable"?)   // 埋め込み
    | "learnable"                             // テンソルとして学習可能
}

// ルール定義
rule_def = { "rule" ~ IDENT ~ "(" ~ param_list ~ ")" ~ rule_body }
rule_body = {
    ":-" ~ rule_conditions                    // 論理ルール
    | "constraint" ~ ":-" ~ rule_conditions   // 制約ルール
}
```

### 7.2 エンティティレジストリ

#### データ構造

```rust
struct EntityRegistry {
    types: HashMap<String, EntityType>,
}

struct EntityType {
    name: String,
    entities: HashSet<String>,     // {"alice", "bob", ...}
    frozen: bool,                   // withブロック後にtrue
    entity_to_id: HashMap<String, usize>,  // "alice" -> 0
    id_to_entity: Vec<String>,      // [0 -> "alice", 1 -> "bob", ...]
}

impl EntityRegistry {
    fn add_entity(&mut self, type_name: &str, entity: &str) -> Result<usize> {
        let entity_type = self.types.get_mut(type_name)?;

        if entity_type.frozen {
            return Err(Error::EntityTypeFrozen(
                format!("Cannot add '{}' to frozen type '{}'", entity, type_name)
            ));
        }

        if !entity_type.entities.contains(entity) {
            let id = entity_type.entities.len();
            entity_type.entities.insert(entity.to_string());
            entity_type.entity_to_id.insert(entity.to_string(), id);
            entity_type.id_to_entity.push(entity.to_string());
        }

        Ok(entity_type.entity_to_id[entity])
    }

    fn freeze(&mut self, type_name: &str) {
        if let Some(entity_type) = self.types.get_mut(type_name) {
            entity_type.frozen = true;
            println!("Entity type '{}' frozen with {} entities",
                     type_name, entity_type.entities.len());
        }
    }

    fn get_id(&self, type_name: &str, entity: &str) -> Result<usize> {
        let entity_type = self.types.get(type_name)?;
        entity_type.entity_to_id.get(entity)
            .copied()
            .ok_or(Error::UnknownEntity(entity.to_string()))
    }
}
```

### 7.3 型推論エンジン

#### 型推論の流れ

```rust
struct TypeInference {
    entity_types: HashMap<String, EntityType>,
    relation_types: HashMap<String, RelationType>,
}

struct RelationType {
    params: Vec<(String, String)>,  // [(x, Person), (y, Person)]
    shape: Vec<usize>,               // [4, 4]
}

impl TypeInference {
    fn infer_rule(&self, rule: &Rule) -> Result<EinsumSpec> {
        // 1. ルールの条件から変数の型を推論
        let mut var_types: HashMap<String, String> = HashMap::new();

        for condition in &rule.conditions {
            let relation = self.relation_types.get(&condition.name)?;

            for (arg, (param_name, param_type)) in
                condition.args.iter().zip(&relation.params) {

                if let Some(var_name) = arg.as_variable() {
                    // 変数の型を記録
                    var_types.insert(var_name.clone(), param_type.clone());
                }
            }
        }

        // 2. einsum文字列を生成
        let einsum_spec = self.generate_einsum(&rule.conditions, &var_types)?;

        Ok(einsum_spec)
    }

    fn generate_einsum(
        &self,
        conditions: &[Condition],
        var_types: &HashMap<String, String>
    ) -> Result<EinsumSpec> {
        // 変数名からインデックス文字へのマッピング
        let mut var_to_index: HashMap<String, char> = HashMap::new();
        let mut next_index = 'a';

        let mut input_specs = Vec::new();
        let mut output_vars = Vec::new();

        for condition in conditions {
            let mut indices = String::new();

            for arg in &condition.args {
                if let Some(var_name) = arg.as_variable() {
                    let index = *var_to_index.entry(var_name.clone())
                        .or_insert_with(|| {
                            let idx = next_index;
                            next_index = (next_index as u8 + 1) as char;
                            idx
                        });
                    indices.push(index);
                }
            }

            input_specs.push((condition.name.clone(), indices));
        }

        // 出力仕様を決定
        // （ルールの結論部の変数のみ）

        Ok(EinsumSpec {
            inputs: input_specs,
            output: output_spec,
        })
    }
}
```

### 7.4 テンソル確保とメモリ管理

#### withブロック終了時の処理

```rust
impl Interpreter {
    fn execute_with_block(
        &mut self,
        entity_type: &str,
        statements: &[Statement]
    ) -> Result<()> {
        // 1. エンティティ型を「収集モード」に設定
        self.entity_registry.set_collecting_mode(entity_type, true);

        // 2. ブロック内の文を実行（エンティティを収集）
        for stmt in statements {
            self.execute_statement(stmt)?;
        }

        // 3. エンティティ型を固定
        self.entity_registry.freeze(entity_type);

        // 4. この型を使う全ての関係のテンソルを確保
        self.allocate_tensors_for_entity(entity_type)?;

        Ok(())
    }

    fn allocate_tensors_for_entity(&mut self, entity_type: &str) -> Result<()> {
        let entity_count = self.entity_registry.get_count(entity_type)?;

        // この型を使う全ての関係を探す
        for (rel_name, rel_type) in &self.relation_types {
            let mut shape = Vec::new();

            for (_, param_type) in &rel_type.params {
                if param_type == entity_type {
                    shape.push(entity_count);
                } else {
                    // 他のエンティティ型のサイズ
                    let size = self.entity_registry.get_count(param_type)?;
                    shape.push(size);
                }
            }

            // テンソル確保
            let device = self.metal_device();
            let tensor = Tensor::zeros(device, &shape)?;
            self.relation_tensors.insert(rel_name.clone(), tensor);

            println!("Allocated tensor for '{}' with shape {:?}", rel_name, shape);
        }

        Ok(())
    }
}
```

### 7.5 GPU最適化

#### Metal GPUでのeinsum実装

```rust
impl Tensor {
    pub fn einsum(
        spec: &str,
        tensors: &[&Tensor]
    ) -> TensorResult<Tensor> {
        // 1. einsum仕様を解析
        let (inputs, output) = parse_einsum_spec(spec)?;

        // 2. 最適な計算順序を決定
        let plan = optimize_einsum_plan(&inputs, &output, tensors)?;

        // 3. Metal GPUで実行
        match plan {
            EinsumPlan::MatrixMultiply(a, b) => {
                // 行列積として実行（最速）
                Self::matmul(tensors[a], tensors[b])
            }
            EinsumPlan::Contraction(dims) => {
                // 一般的なテンソル縮約
                Self::contract_gpu(tensors, &dims)
            }
        }
    }

    fn contract_gpu(
        tensors: &[&Tensor],
        contraction: &ContractionSpec
    ) -> TensorResult<Tensor> {
        // Metal Compute Shader を使った実装
        let device = tensors[0].device();
        let command_queue = device.new_command_queue();

        // カーネル実行
        // ...
    }
}
```

---

## 8. まとめ

### 8.1 設計の核心

TensorLogicの論理プログラミングとテンソルの統一は、以下の3つの核心的アイデアに基づいています：

1. **型付きエンティティ**: エンティティ型がテンソル次元を決定
2. **アインシュタイン和**: 論理ルールを自動的にテンソル演算に変換
3. **withブロック**: 静的型の利点と柔軟性を両立

### 8.2 推奨構文

```tensorlogic
// エンティティ型定義
entity Person
entity Item

// 関係定義
relation Friend(x: Person, y: Person)
relation Likes(x: Person, y: Item)

// ルール定義
rule Recommends(x: Person, z: Item) :-
    Friend(x, y),
    Likes(y, z)

main {
    // エンティティ収集
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }

    with Item {
        Likes(alice, book)
        Likes(bob, movie)
    }

    // 学習
    learn {
        // ハイブリッドアプローチ
    }

    // 推論
    infer {
        forward Recommends(alice, X)?
    }
}
```

### 8.3 実装ロードマップ

**Phase 1: 基盤**（優先度：最高）
- ✅ `entity T` の解析（明示的列挙とデータからの自動構築）
- ✅ `with T { ... }` ブロックの解析
- ✅ エンティティレジストリの実装
- ✅ テンソル確保

**Phase 2: ルール**（優先度：高）
- ✅ ルール定義の解析
- ✅ 型推論エンジン
- ✅ einsum自動生成

**Phase 3: 学習統合**（優先度：中）
- ✅ 埋め込みベースの学習（方法1）
- ✅ 制約ルールのサポート（方法3の一部）

**Phase 4: 高度な機能**（優先度：低）
- ✅ 完全なハイブリッドアプローチ（方法3）
- ✅ テンソルルールベース（方法2、研究的）

### 8.4 次のステップ

1. **ドキュメントレビュー**: この設計を検討・改善
2. **プロトタイプ実装**: Phase 1の基盤実装
3. **実例作成**: チュートリアルと使用例
4. **コミュニティフィードバック**: ユーザーからの意見収集

---

**参考文献**:
- [Einstein Summation Convention](https://en.wikipedia.org/wiki/Einstein_notation)
- [TransE: Translating Embeddings for Knowledge Graphs](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [Neural-Symbolic Computing: An Effective Methodology](https://arxiv.org/abs/1905.06088)
