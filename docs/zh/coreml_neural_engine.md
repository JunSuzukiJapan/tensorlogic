# CoreML & Neural Engine 集成指南

本指南说明如何在 TensorLogic 中使用 CoreML 模型并在 Apple Neural Engine 上执行高速推理。

## 关于 CoreML 和 Neural Engine

### CoreML

- Apple 专有的机器学习框架
- 专为 iOS/macOS 优化
- 自动利用 Neural Engine、GPU 和 CPU
- .mlmodel / .mlmodelc 格式

### Neural Engine

- Apple Silicon 专用的 AI 专用芯片
- 高达 15.8 TOPS (M1 Pro/Max)
- 超低功耗（与 GPU 相比为 1/10 或更少）
- 针对 f16 运算进行优化

### 与 TensorLogic 的集成

- 所有 f16 操作（Neural Engine 优化）
- 与 Metal GPU 无缝集成
- 自动模型格式检测

## 创建 CoreML 模型

CoreML 模型通常使用 Python 的 coreMLtools 创建：

```python
import coremltools as ct
import torch

# 创建 PyTorch 模型
model = MyModel()
model.eval()

# 创建追踪模型
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 转换为 CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Neural Engine 优化
    compute_precision=ct.precision.FLOAT16  # f16 精度
)

# 保存
mlmodel.save("model.mlpackage")
```

## 在 TensorLogic 中使用

### 1. 加载 CoreML 模型（仅限 macOS）

```tensorlogic
model = load_model("model.mlpackage")
// 或
model = load_model("model.mlmodelc")
```

### 2. 检查元数据

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Neural Engine 优化最佳实践

### 1. 数据类型：使用 f16

✅ 推荐：`compute_precision=ct.precision.FLOAT16`
❌ 不推荐：FLOAT32（在 GPU 上执行）

### 2. 模型格式：使用 mlprogram 格式

✅ 推荐：`convert_to="mlprogram"`
❌ 不推荐：`convert_to="neuralnetwork"`（旧格式）

### 3. 批量大小：1 最佳

✅ 推荐：`batch_size=1`
⚠️ 注意：`batch_size>1` 可能在 GPU 上执行

### 4. 输入大小：固定大小最佳

✅ 推荐：`shape=[1, 3, 224, 224]`
⚠️ 注意：可变大小的优化受限

## 支持的操作

### Neural Engine 上快速执行的操作

- ✅ 卷积 (conv2d, depthwise_conv)
- ✅ 全连接层 (linear, matmul)
- ✅ 池化 (max_pool, avg_pool)
- ✅ 归一化 (batch_norm, layer_norm)
- ✅ 激活函数 (relu, gelu, sigmoid, tanh)
- ✅ 逐元素操作 (add, mul, sub, div)

## 性能比较

ResNet-50 推理（224x224 图像）：

| 设备               | 延迟    | 功耗   | 效率   |
|-------------------|--------|-------|--------|
| Neural Engine     | ~3ms   | ~0.5W | 最高   |
| Metal GPU (M1)    | ~8ms   | ~5W   | 中等   |
| CPU (M1)          | ~50ms  | ~2W   | 低     |

## 模型格式选择

### 按使用场景推荐的格式

**训练**：SafeTensors
- PyTorch 兼容
- 权重保存/加载
- 在 Metal GPU 上训练

**推理 (iOS/macOS)**：CoreML
- Neural Engine 优化
- 超低功耗
- 应用集成

**推理 (通用)**：GGUF
- 量化支持
- 跨平台
- 内存高效

## 参考资料

- [CoreML 官方文档](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Neural Engine 指南](https://machinelearning.apple.com/research/neural-engine-transformers)
- [模型加载指南](model_loading.md)
