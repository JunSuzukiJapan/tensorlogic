# GGUF 量化模型指南

本文档说明如何在 TensorLogic 中加载和使用 GGUF 格式的量化模型（llama.cpp 兼容）。

## 关于 GGUF 格式

GGUF (GGML Universal Format) 是由 llama.cpp 项目开发的用于大型语言模型的高效量化格式。

### 主要特点

- 通过 4-bit/8-bit 量化提高内存效率（最高 8 倍压缩）
- 基于块的量化保持精度
- 与 llama.cpp、Ollama、LM Studio 等兼容

### TensorLogic 支持的量化格式

- ✅ **Q4_0**: 4-bit 量化（最高压缩率）
- ✅ **Q8_0**: 8-bit 量化（精度和压缩平衡）
- ✅ **F16**: 16-bit 浮点（高精度）
- ✅ **F32**: 32-bit 浮点（最高精度）

## 基本用法

### 1. 加载量化模型

自动反量化为 f16 并加载到 Metal GPU：

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. 从模型获取张量

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## 选择量化格式

### Q4_0 (4-bit)

- **内存**: 最小使用量（原始模型的 ~1/8）
- **速度**: 最快推理
- **精度**: 轻微下降（通常可接受）
- **使用场景**: 聊天机器人、通用文本生成

### Q8_0 (8-bit)

- **内存**: 中等使用量（原始模型的 ~1/4）
- **速度**: 快速
- **精度**: 高（几乎等同于 F16）
- **使用场景**: 高质量生成、编码助手

### F16 (16-bit)

- **内存**: 原始模型的 ~1/2
- **速度**: 标准
- **精度**: TensorLogic 原生格式，Metal GPU 优化
- **使用场景**: 需要最高质量时

## 实用示例：Token 嵌入

```tensorlogic
// 从 LLama 模型获取 token 嵌入
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// 从 token ID 获取嵌入向量
function get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## 量化的内存节省

示例：LLama-7B 模型（70 亿参数）：

| 格式       | 内存使用 | 压缩率 |
|-----------|---------|--------|
| F32 (原始) | ~28 GB  | 1x     |
| F16       | ~14 GB  | 2x     |
| Q8_0      | ~7 GB   | 4x     |
| Q4_0      | ~3.5 GB | 8x     |

TensorLogic 在加载时将所有格式转换为 f16，并在 Metal GPU 上高效执行。

## 下载和安装模型

### 1. 从 HuggingFace 下载 GGUF 模型

示例：https://huggingface.co/TheBloke

### 2. 推荐模型（适合初学者）

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. 在 TensorLogic 中加载

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## 重要注意事项

- 量化模型为只读（无法从 TensorLogic 保存）
- 训练使用非量化模型（F16/F32）
- Q4/Q8 仅针对推理进行优化
- 所有量化格式自动反量化为 f16 并加载到 GPU

## 参考资料

- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF 模型](https://huggingface.co/TheBloke)
- [模型加载指南](model_loading.md)
