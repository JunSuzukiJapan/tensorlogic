# GGUF ファイル形式 詳細仕様書

## 概要

GGUF（GPT-Generated Unified Format / Generic GPT Unified Format）は、機械学習モデル、特に大規模言語モデル（LLM）を効率的に保存・実行するために設計されたバイナリファイル形式です。GGMLエグゼキューターと共に使用されることを前提としており、以前のフォーマット（GGML、GGMF、GGJT）の後継として2023年8月21日に導入されました。

### 主な特徴

- **単一ファイル構造**: モデルの全ての必要情報を1つのファイルに格納
- **明確な構造**: 曖昧さのない（unambiguous）構造定義
- **拡張性**: 新しいメタデータの追加が既存モデルとの互換性を壊さない
- **効率性**: 量子化されたモデルの効率的なストレージと実行
- **CPU推論**: llama.cppなどを使用したCPUベースの推論をサポート

---

## ファイル全体構造

GGUFファイルは以下の4つのセクションで構成されます：

```
[Header] → [Metadata KV Pairs] → [Tensor Info] → [Padding] → [Tensor Data]
```

### アライメント

ファイル全体は `general.alignment` メタデータフィールドで指定されるグローバルアライメントに従います。デフォルトは32バイトです。セクション間のパディングには `0x00` バイトが使用され、アライメント値の倍数に揃えられます。

---

## 1. ヘッダー構造

ヘッダーは固定サイズの基本情報を含みます（最小16バイト）。

### ヘッダーフィールド（バイトレベル）

| フィールド | 型 | サイズ | オフセット | 説明 |
|-----------|-----|--------|-----------|------|
| `magic` | `uint32_t` | 4 bytes | 0x00 | マジックナンバー：`0x47475546` (ASCII: "GGUF") |
| `version` | `uint32_t` | 4 bytes | 0x04 | フォーマットバージョン（現在は3） |
| `n_tensors` | `uint64_t` | 8 bytes | 0x08 | テンソルの総数 |
| `n_kv` | `uint64_t` | 8 bytes | 0x10 | メタデータのキー・バリューペアの数 |

### バージョン情報

- **Version 1**: 初期バージョン
- **Version 2**: 改善版
- **Version 3**: ビッグエンディアンモデルのサポート追加（現在の推奨バージョン）

### エンディアン

- **デフォルト**: リトルエンディアン
- **ビッグエンディアン**: サポートされているが、明示的なマーカーなし
- **推奨**: 特に文書化されていない限り、リトルエンディアンを仮定

---

## 2. メタデータ キー・バリュー ペア

ヘッダーの直後に続くセクションで、モデルに関する柔軟な情報を格納します。

### メタデータ構造（1つのKVペア）

各キー・バリューペアは以下の構造を持ちます：

1. **キー文字列** (`gguf_str`)
   - `n` (uint64_t, 8 bytes): 文字列の長さ
   - `data` (char array, n bytes): UTF-8文字列データ（ヌル終端なし）

2. **バリュー型** (`uint32_t`, 4 bytes)
   - 型列挙値（0-12）

3. **バリュー** (可変長)
   - 型に応じたデータ

### メタデータ値の型

| 型ID | 型名 | サイズ | 説明 |
|------|------|--------|------|
| 0 | `UINT8` | 1 byte | 符号なし8ビット整数 |
| 1 | `INT8` | 1 byte | 符号付き8ビット整数 |
| 2 | `UINT16` | 2 bytes | 符号なし16ビット整数 |
| 3 | `INT16` | 2 bytes | 符号付き16ビット整数 |
| 4 | `UINT32` | 4 bytes | 符号なし32ビット整数 |
| 5 | `INT32` | 4 bytes | 符号付き32ビット整数 |
| 6 | `FLOAT32` | 4 bytes | 32ビット浮動小数点数 |
| 7 | `BOOL` | 1 byte | ブール値 |
| 8 | `STRING` | 可変 | UTF-8文字列（gguf_str構造） |
| 9 | `ARRAY` | 可変 | 配列（型 + カウント + データ） |
| 10 | `UINT64` | 8 bytes | 符号なし64ビット整数 |
| 11 | `INT64` | 8 bytes | 符号付き64ビット整数 |
| 12 | `FLOAT64` | 8 bytes | 64ビット浮動小数点数 |

### 配列型の構造

配列値の場合、以下の構造になります：

- **type** (uint32_t, 4 bytes): 配列要素の型ID
- **n** (uint64_t, 8 bytes): 要素数
- **data** (可変長): type_size × n バイトのデータ

### キーの命名規則

- **階層構造**: ドット（`.`）で区切られたセグメント
- **命名スタイル**: `lower_snake_case`
- **例**:
  - `general.architecture`
  - `llama.context_length`
  - `tokenizer.ggml.tokens`

### 一般的なメタデータキー

#### 一般情報

| キー | 型 | 説明 |
|------|-----|------|
| `general.architecture` | STRING | モデルアーキテクチャ（llama, falcon, gpt2など） |
| `general.name` | STRING | 人間が読めるモデル名 |
| `general.alignment` | UINT32 | グローバルアライメント値（デフォルト32） |
| `general.quantization_version` | UINT32 | 量子化手法のバージョン |

#### アーキテクチャ固有（[arch]はアーキテクチャ名）

| キー | 型 | 説明 |
|------|-----|------|
| `[arch].context_length` | UINT32 | コンテキストウィンドウサイズ |
| `[arch].embedding_length` | UINT32 | 埋め込み次元数 |
| `[arch].block_count` | UINT32 | トランスフォーマーブロック数 |
| `[arch].feed_forward_length` | UINT32 | フィードフォワード層の次元数 |
| `[arch].attention.head_count` | UINT32 | アテンションヘッド数 |

#### トークナイザー

| キー | 型 | 説明 |
|------|-----|------|
| `tokenizer.ggml.model` | STRING | トークナイザーの種類 |
| `tokenizer.ggml.tokens` | ARRAY[STRING] | 語彙リスト |
| `tokenizer.ggml.scores` | ARRAY[FLOAT32] | トークンスコア |
| `tokenizer.ggml.token_type` | ARRAY[INT32] | トークンタイプ |

---

## 3. テンソル情報セクション

このセクションには、モデル内の各テンソルのメタデータが含まれます（実際のデータは含まれません）。

### テンソル情報構造（`gguf_tensor_info_t`）

各テンソルについて、以下の情報が格納されます：

| フィールド | 型 | サイズ | 説明 |
|-----------|-----|--------|------|
| `name` | `gguf_str` | 可変 | テンソル名（最大64バイト推奨） |
| `n_dims` | `uint32_t` | 4 bytes | 次元数（通常1-4） |
| `dims[4]` | `uint64_t[4]` | 32 bytes | 各次元のサイズ（ne配列） |
| `type` | `uint32_t` | 4 bytes | GGMLデータ型 |
| `offset` | `uint64_t` | 8 bytes | テンソルデータセクション内のオフセット |

### テンソル名の例

```
blk.0.attn_norm.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.ffn_norm.weight
output.weight
```

### ⚠️ 重要: テンソル次元の格納順序

**GGUFファイルでは、テンソルの次元配列（`dims`）が逆順で保存されています。**

#### 詳細説明

GGUFファイルに保存されているテンソルの次元配列は、多くのMLフレームワーク（PyTorch、NumPyなど）の次元順序とは**逆**になっています。

**例**:
- PyTorchで `[vocab_size, d_model]` = `[32000, 2048]` の形状を持つ埋め込みテンソル
- GGUFファイル内では `dims = [2048, 32000]` として保存されます
- 同様に、線形層の重み `[out_features, in_features]` も逆順で保存されます

#### 実装上の対応

**必須処理**: GGUFファイルからテンソル情報を読み込んだ後、次元配列を**必ず反転**してください。

```rust
// Rustでの実装例
let mut shape: Vec<usize> = tensor_info.shape.dimensions
    .iter()
    .map(|&d| d as usize)
    .collect();

// 必須: 次元配列を反転
shape.reverse();
```

```python
# Pythonでの実装例
dims = tensor_info.dims.copy()
dims.reverse()  # または dims[::-1]
```

```c++
// C++での実装例（llama.cpp、candle）
std::vector<uint64_t> dimensions = tensor_info.ne;
std::reverse(dimensions.begin(), dimensions.end());
```

#### 参考実装

主要な実装例:
- **llama.cpp**: 次元を反転して処理
- **candle**: `dimensions.reverse()` を明示的に呼び出し
- **gguf-py**: 読み込み時に次元を反転

#### 技術的背景

- GGMLライブラリの内部表現との互換性のため、この順序が採用されています
- 参考: [llama.cpp Issue #6040](https://github.com/ggml-org/llama.cpp/issues/6040)
- この仕様は公式ドキュメントには明記されていませんが、すべての参照実装で一貫しています

#### 実装チェックリスト

GGUFローダーを実装する際は、以下を確認してください:

- [ ] テンソル情報読み込み後、`dims`配列を反転している
- [ ] 埋め込み層が正しい形状 `[vocab_size, embedding_dim]` になっている
- [ ] 線形層の重みが正しい形状 `[out_features, in_features]` になっている
- [ ] 実際のテンソルデータと形状が一致している

### GGMLデータ型

GGUFは多様なデータ型をサポートしています：

#### 浮動小数点型

| 型ID | 型名 | 説明 |
|------|------|------|
| 0 | `F32` | 32ビット浮動小数点数 |
| 1 | `F16` | 16ビット浮動小数点数 |
| 28 | `F64` | 64ビット浮動小数点数 |
| 30 | `BF16` | Brain Float 16 |

#### 量子化型（Q系列）

| 型ID | 型名 | 説明 |
|------|------|------|
| 2 | `Q4_0` | 4ビット量子化 v0 |
| 3 | `Q4_1` | 4ビット量子化 v1 |
| 6 | `Q5_0` | 5ビット量子化 v0 |
| 7 | `Q5_1` | 5ビット量子化 v1 |
| 8 | `Q8_0` | 8ビット量子化 v0 |
| 9 | `Q8_1` | 8ビット量子化 v1 |

#### K量子化型

| 型ID | 型名 | 説明 |
|------|------|------|
| 12 | `Q2_K` | 2ビット K量子化 |
| 13 | `Q3_K` | 3ビット K量子化 |
| 14 | `Q4_K` | 4ビット K量子化 |
| 15 | `Q5_K` | 5ビット K量子化 |
| 16 | `Q6_K` | 6ビット K量子化 |
| 17 | `Q8_K` | 8ビット K量子化 |

#### 整数型

| 型ID | 型名 | 説明 |
|------|------|------|
| 24 | `I8` | 8ビット整数 |
| 25 | `I16` | 16ビット整数 |
| 26 | `I32` | 32ビット整数 |
| 27 | `I64` | 64ビット整数 |

#### 特殊型

| 型ID | 型名 | 説明 |
|------|------|------|
| 18 | `IQ2_XXS` | IQ 2ビット超小型 |
| 19 | `IQ2_XS` | IQ 2ビット小型 |
| 20 | `IQ3_XXS` | IQ 3ビット超小型 |
| 21 | `IQ1_S` | IQ 1ビット |
| 22 | `IQ4_NL` | IQ 4ビット非線形 |
| 23 | `IQ3_S` | IQ 3ビット |
| 29 | `IQ4_XS` | IQ 4ビット小型 |
| 31 | `TQ1_0` | 3元量子化 v0 |
| 32 | `TQ2_0` | 3元量子化 v1 |

### オフセットとアライメント

- **オフセット**: テンソルデータセクションの開始位置からの相対位置
- **アライメント要件**: オフセットは `general.alignment` の倍数でなければならない
- **計算**: パディングを考慮して次のテンソルのオフセットを計算

---

## 4. テンソルデータセクション

実際のテンソルの数値データが格納されるセクションです。

### 特徴

- **単一の連続したブロブ**: 全テンソルデータが1つの連続領域に格納
- **アライメント**: 各テンソルの開始位置はアライメント要件を満たす
- **パディング**: テンソル間に必要に応じて `0x00` バイトが挿入される
- **順序**: テンソル情報セクションで指定された順序とオフセットに従う

### データレイアウト

```
[Padding to alignment]
[Tensor 0 data]
[Padding to alignment]
[Tensor 1 data]
[Padding to alignment]
...
[Tensor n-1 data]
```

---

## ファイル命名規則

GGUFファイル名は標準的なパターンに従います：

```
<BaseName>-<SizeLabel>-<FineTune>-<Version>-<Encoding>-<Type>-<Shard>.gguf
```

### コンポーネント

| コンポーネント | 説明 | 例 |
|--------------|------|-----|
| `BaseName` | モデル名 | `llama-2`, `mistral` |
| `SizeLabel` | パラメータ数（スケール接頭辞付き） | `7B`, `13B`, `70B` |
| `FineTune` | ファインチューン情報（オプション） | `chat`, `instruct` |
| `Version` | モデルバージョン | `v1.0`, `v2.1` |
| `Encoding` | 量子化方式 | `Q4_K_M`, `Q8_0` |
| `Type` | モデルタイプ（オプション） | `fp16`, `int8` |
| `Shard` | シャード情報（オプション） | `00001-of-00009` |

### 例

```
llama-2-7B-chat-v1.0-Q4_K_M.gguf
mistral-7B-instruct-v0.2-Q8_0.gguf
llama-2-70B-Q4_K_M-00001-of-00009.gguf
```

### スケール接頭辞

- **K**: Thousand（千） - 例: 7K = 7,000
- **M**: Million（百万） - 例: 7M = 7,000,000
- **B**: Billion（十億） - 例: 7B = 7,000,000,000
- **T**: Trillion（一兆） - 例: 1T = 1,000,000,000,000
- **Q**: Quadrillion（千兆）

---

## GGUFの利点

### 1. GGMLからの改善点

| 側面 | GGML | GGUF |
|------|------|------|
| メタデータ構造 | 型なしリスト | キー・バリュー構造 |
| 拡張性 | 新規追加で互換性が破壊 | 新規メタデータ追加が安全 |
| 標準化 | 一貫性が低い | 高度に標準化 |
| 語彙管理 | 不明確 | 統一されたトークナイザー情報 |

### 2. 実用的な利点

- **単一ファイル展開**: モデル全体が1ファイルで完結
- **効率的なローディング**: 必要な情報が明確に構造化
- **バージョン互換性**: 新しいソフトウェアが古いファイルを読める設計
- **メモリ効率**: 量子化によるモデルサイズの大幅削減
- **CPU推論**: 高価なGPUなしでLLMを実行可能

---

## 実装例

### C/C++でのヘッダー読み込み

```c
struct gguf_header {
    uint32_t magic;      // 0x47475546
    uint32_t version;    // 3
    uint64_t n_tensors;  // テンソル数
    uint64_t n_kv;       // KVペア数
};

// ファイルからヘッダーを読み込み
FILE* file = fopen("model.gguf", "rb");
struct gguf_header header;
fread(&header, sizeof(header), 1, file);

// マジックナンバーの検証
if (header.magic != 0x47475546) {
    fprintf(stderr, "Invalid GGUF file\n");
    return -1;
}
```

### Rustでのメタデータ読み込み

```rust
use std::io::{Read, Seek, SeekFrom};
use std::fs::File;

struct GGUFString {
    length: u64,
    data: Vec<u8>,
}

fn read_gguf_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let length = u64::from_le_bytes(len_buf);

    let mut data = vec![0u8; length as usize];
    reader.read_exact(&mut data)?;

    String::from_utf8(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
```

---

## ツールとライブラリ

### 主要な実装

- **llama.cpp**: C++での参照実装、CPU推論エンジン
- **ggml**: 低レベルテンソル演算ライブラリ
- **Hugging Face Transformers**: Pythonでの統合サポート
- **Candle**: Rustでの実装例

### ユーティリティ

- **gguf-py**: Pythonでの読み書きライブラリ
- **gguf-dump**: ファイル内容の可視化ツール
- **convert.py**: 他フォーマットからの変換ツール

---

## トラブルシューティング

### 一般的な問題

#### 1. マジックナンバーの不一致

```
エラー: Invalid magic number
原因: ファイルが破損しているか、GGUF形式ではない
解決: ファイルの整合性を確認、再ダウンロード
```

#### 2. アライメントエラー

```
エラー: Tensor offset not aligned
原因: オフセット計算でアライメント要件を満たしていない
解決: general.alignmentメタデータを確認、パディング計算を修正
```

#### 3. エンディアンの問題

```
エラー: Unexpected values in header
原因: ビッグエンディアンファイルをリトルエンディアンとして読んでいる
解決: エンディアンを正しく処理、バイトスワップを実装
```

---

## 参考資料

### 公式ドキュメント

- **GGUF仕様書**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **ggml**: https://github.com/ggml-org/ggml

### コミュニティリソース

- GGUF解説記事: https://apxml.com/posts/gguf-explained-llm-file-format
- 実装ガイド: https://cca.informatik.uni-freiburg.de/debugging/ws23/FORMAT.html

---

## 付録

### A. 完全なヘッダー例（バイナリダンプ）

```
Offset(h) 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F

00000000  47 47 55 46 03 00 00 00 20 00 00 00 00 00 00 00  GGUF.... .......
00000010  0C 00 00 00 00 00 00 00 ...                      ........
```

### B. メタデータ例（JSON表現）

```json
{
  "general.architecture": "llama",
  "general.name": "LLaMA 2 7B",
  "general.alignment": 32,
  "llama.context_length": 4096,
  "llama.embedding_length": 4096,
  "llama.block_count": 32,
  "llama.feed_forward_length": 11008,
  "llama.attention.head_count": 32,
  "tokenizer.ggml.model": "llama",
  "tokenizer.ggml.tokens": ["<s>", "</s>", "<unk>", ...]
}
```

### C. データ型サイズ一覧

| 型 | ビット/要素 | バイト/要素 | 備考 |
|-----|----------|-----------|------|
| F32 | 32 | 4 | 標準float |
| F16 | 16 | 2 | Half precision |
| Q4_0 | 4 | 0.5 | 4ビット量子化 |
| Q4_K | 4.5 | ~0.56 | Kスケール量子化 |
| Q8_0 | 8 | 1 | 8ビット量子化 |
| I8 | 8 | 1 | 整数 |
| BF16 | 16 | 2 | Brain Float |

---

## 改訂履歴

- **2025-10-25**: テンソル次元の格納順序に関する重要な注意事項を追加（実装時の必須要件）
- **2025-01-25**: 初版作成（GGUF Version 3仕様に基づく）
- 出典: ggml-org/ggml公式ドキュメント、llama.cpp、candle実装、各種技術記事

---

**注意**: この仕様書は2025年1月時点の情報に基づいています。最新の仕様については公式のggml-org/ggmlリポジトリを参照してください。
