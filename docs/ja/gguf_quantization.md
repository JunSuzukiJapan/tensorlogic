# GGUF量子化モデルガイド

このドキュメントは、GGUF形式の量子化モデル（llama.cpp互換）をTensorLogicでロードして使用する方法を説明します。

## GGUFフォーマットについて

GGUF (GGML Universal Format) は、llama.cppプロジェクトで開発された大規模言語モデル用の効率的な量子化フォーマットです。

### 主な特徴

- 4bit/8bit量子化によるメモリ効率向上（最大8倍圧縮）
- ブロックベースの量子化で精度を保持
- llama.cpp、Ollama、LM Studioなどと互換性

### TensorLogicでサポートする量子化形式

- ✅ **Q4_0**: 4bit量子化（最も圧縮率が高い）
- ✅ **Q8_0**: 8bit量子化（精度と圧縮のバランス）
- ✅ **F16**: 16bit浮動小数点（高精度）
- ✅ **F32**: 32bit浮動小数点（最高精度）

## 基本的な使用法

### 1. 量子化モデルのロード

自動的にf16に逆量子化してMetal GPUにロードされます：

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. モデルからテンソルを取得

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## 量子化形式の選び方

### Q4_0 (4bit)

- **メモリ**: 最小使用量（元のモデルの約1/8）
- **速度**: 推論速度が最速
- **精度**: わずかな精度低下（通常は許容範囲内）
- **用途**: チャットボット、一般的なテキスト生成

### Q8_0 (8bit)

- **メモリ**: 中程度（元のモデルの約1/4）
- **速度**: 高速
- **精度**: 高い（F16とほぼ同等）
- **用途**: 高品質な生成、コーディングアシスタント

### F16 (16bit)

- **メモリ**: 元のモデルの約1/2
- **速度**: 標準
- **精度**: TensorLogicのネイティブ形式、Metal GPU最適化
- **用途**: 最高品質が必要な場合

## 実用例：トークン埋め込みの取得

```tensorlogic
// LLamaモデルのトークン埋め込みを取得
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// トークンIDから埋め込みベクトルを取得
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## 量子化のメモリ節約効果

LLama-7Bモデル (70億パラメータ) の例：

| フォーマット | メモリ使用量 | 圧縮率 |
|------------|------------|--------|
| F32 (元)   | ~28 GB     | 1x     |
| F16        | ~14 GB     | 2x     |
| Q8_0       | ~7 GB      | 4x     |
| Q4_0       | ~3.5 GB    | 8x     |

TensorLogicは全ての形式を読み込み時にf16に変換し、Metal GPUで効率的に実行します。

## モデルのダウンロードとインストール

### 1. HuggingFaceからGGUFモデルをダウンロード

例: https://huggingface.co/TheBloke

### 2. 推奨モデル（初心者向け）

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. TensorLogicでロード

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## 逆量子化の仕組み

### Q4_0の場合

1. 32個の4bit値をブロックとしてグループ化
2. 各ブロックに1つのf16スケール因子
3. 逆量子化式: `float_value = (quantized_value - 8) * scale`
4. Metal GPUでf16として実行

### Q8_0の場合

1. 32個の8bit値をブロックとしてグループ化
2. 各ブロックに1つのf16スケール因子
3. 逆量子化式: `float_value = quantized_value * scale`
4. Metal GPUでf16として実行

## 性能比較（推論速度）

Metal GPU (M1 Max) での推論速度：

- **Q4_0**: ~50 tokens/sec (最速)
- **Q8_0**: ~45 tokens/sec
- **F16**: ~40 tokens/sec (最高品質)

※ モデルサイズと複雑さによって変動

## よくある質問

### Q: 量子化モデルと非量子化モデルの精度差は？

A: Q4_0で約2-3%の品質低下、Q8_0ではほぼ無視できるレベル

### Q: どの量子化形式を選ぶべき？

A: メモリ制約がある → Q4_0、品質重視 → Q8_0、最高品質 → F16

### Q: TensorLogicで量子化モデルを作成できる？

A: 現在は読み込みのみ対応。保存はSafeTensors形式を使用

### Q: K-quants (Q4_K, Q5_K等) は対応している？

A: 現在は Q4_0 と Q8_0 のみ。K-quantsは将来のバージョンで対応予定

## 実践例：シンプルな推論

```tensorlogic
// モデルロード（自動的にMetal GPUに配置）
let model = load_model("mistral-7b-q4_0.gguf")

// トークン埋め込みテーブルを取得
let embeddings = model.get_tensor("token_embd.weight")

// 推論実行（TensorLogicの通常の演算を使用）
let input_ids = tensor<int>([1, 2, 3, 4], device: gpu)
let embedded = embeddings[input_ids, :]
print("Embedded shape:", embedded.shape)
```

## 注意事項

- 量子化モデルは読み取り専用（TensorLogicから保存不可）
- 学習には非量子化モデル（F16/F32）を使用
- Q4/Q8は推論専用に最適化
- 全ての量子化形式は自動的にf16に逆量子化されてGPUにロード

## 参考リンク

- [GGUF仕様](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF models](https://huggingface.co/TheBloke)
- [モデルローディングガイド](model_loading.md)
