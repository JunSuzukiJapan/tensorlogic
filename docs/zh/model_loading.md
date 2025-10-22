# 模型加载指南

本文档说明如何在 TensorLogic 中加载和使用 PyTorch 和 HuggingFace 模型。支持 SafeTensors 格式（PyTorch 兼容）和 GGUF 格式（量化 LLM）。

## 基本用法

### 1. 加载 SafeTensors 模型（从 PyTorch 保存）

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. 加载 GGUF 模型（量化 LLM）

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. 从模型获取张量

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## 实用示例：线性层推理

使用模型权重和偏置执行推理：

```tensorlogic
function forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // 线性变换: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## 准备 PyTorch 模型

使用 Python 将模型保存为 SafeTensors 格式：

```python
import torch
from safetensors.torch import save_file

# 创建 PyTorch 模型
model = MyModel()

# 将模型权重获取为字典
tensors = {name: param for name, param in model.named_parameters()}

# 保存为 SafeTensors 格式
save_file(tensors, "model.safetensors")
```

然后在 TensorLogic 中加载：

```tensorlogic
model = load_model("model.safetensors")
```

## 支持的格式

### 1. SafeTensors (.safetensors)

- 与 PyTorch 和 HuggingFace 兼容
- 支持 F32、F64、F16、BF16 数据类型
- 所有数据自动转换为 f16
- 直接加载到 Metal GPU

### 2. GGUF (.gguf)

- llama.cpp 格式的量化模型
- 支持 Q4_0、Q8_0、F32、F16
- 直接加载到 Metal GPU

### 3. CoreML (.mlmodel, .mlpackage)

- 为 Apple Neural Engine 优化的模型
- 仅限 iOS/macOS

## 完整线性模型示例

```tensorlogic
// 输入数据（批大小 4，特征维度 3）
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// 权重矩阵（3 x 2）
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// 偏置（2 维）
let b = tensor<float16>([0.01, 0.02], device: gpu)

// 执行推理
let output = forward(X, W, b)

// 打印结果
print("Output shape:", output.shape)
print("Output:", output)
```

## 保存模型

您可以将 TensorLogic 模型保存为 SafeTensors 格式：

```tensorlogic
save_model(model, "output.safetensors")
```

这实现了与 PyTorch 和 HuggingFace 的互操作性。

## 重要注意事项

- TensorLogic 以 f16 执行所有操作（Metal GPU 优化）
- 其他数据类型在加载时自动转换为 f16
- 不支持整数类型（i8、i32 等）（仅浮点）
- 大型模型自动加载到 Metal GPU 内存

## 相关文档

- [GGUF 量化模型](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [入门指南](../claudedocs/getting_started.md)
