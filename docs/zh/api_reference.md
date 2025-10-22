# TensorLogic API 参考

TensorLogic 中所有可用操作的完整 API 参考。

## 目录

1. [张量创建](#张量创建)
2. [形状操作](#形状操作)
3. [数学函数](#数学函数)
4. [聚合操作](#聚合操作)
5. [激活函数](#激活函数)
6. [矩阵运算](#矩阵运算)
7. [归一化](#归一化)
8. [掩码操作](#掩码操作)
9. [索引操作](#索引操作)
10. [嵌入](#嵌入)
11. [采样](#采样)
12. [融合操作](#融合操作)
13. [优化](#优化)
14. [其他操作](#其他操作)
15. [运算符](#运算符)
16. [类型定义](#类型定义)

---

## 张量创建

### `zeros(shape: Array<Int>) -> Tensor`

创建一个填充零的张量。

**参数:**
- `shape`: 指定张量维度的数组

**返回:** 填充 0 的张量

**示例:**
```tensorlogic
let z = zeros([2, 3])  // 2x3 的零张量
```

---

### `ones(shape: Array<Int>) -> Tensor`

创建一个填充一的张量。

**参数:**
- `shape`: 指定张量维度的数组

**返回:** 填充 1 的张量

**示例:**
```tensorlogic
let o = ones([2, 3])  // 2x3 的一张量
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

为 Transformer 生成正弦位置编码。

**参数:**
- `seq_len`: 序列长度
- `d_model`: 模型维度

**返回:** 形状为 `[seq_len, d_model]` 的张量

**数学定义:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**示例:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**用例:**
- Transformer 模型
- 序列到序列模型
- 注意力机制

**参考:**
- arXiv:2510.12269 (表 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## 形状操作

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

在保留数据的同时改变张量形状。

**参数:**
- `tensor`: 输入张量
- `new_shape`: 目标形状

**返回:** 重塑后的张量

**示例:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**约束:**
- 元素总数必须保持不变

---

### `flatten(tensor: Tensor) -> Tensor`

将张量展平为 1D。

**参数:**
- `tensor`: 输入张量

**返回:** 1D 张量

**示例:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

转置 2D 张量(交换轴)。

**参数:**
- `tensor`: 输入 2D 张量

**返回:** 转置后的张量

**示例:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

重新排列张量维度。

**参数:**
- `tensor`: 输入张量
- `dims`: 新的维度顺序

**返回:** 排列后的张量

**示例:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

在指定位置添加大小为 1 的维度。

**参数:**
- `tensor`: 输入张量
- `dim`: 插入新维度的位置

**返回:** 添加维度后的张量

**示例:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

移除所有大小为 1 的维度。

**参数:**
- `tensor`: 输入张量

**返回:** 移除大小为 1 的维度后的张量

**示例:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

沿指定维度将张量分割为多个张量。

**参数:**
- `tensor`: 输入张量
- `sizes`: 每个分割部分的大小
- `dim`: 要分割的维度

**返回:** 张量数组

**示例:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 个张量: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

将张量分割为指定数量的块。

**参数:**
- `tensor`: 输入张量
- `chunks`: 块的数量
- `dim`: 要分割的维度

**返回:** 张量数组

**示例:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 个 [4,4] 的张量
```

---

## 数学函数

### `exp(tensor: Tensor) -> Tensor`

逐元素应用指数函数。

**数学定义:** `exp(x) = e^x`

**示例:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

逐元素应用自然对数。

**数学定义:** `log(x) = ln(x)`

**示例:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

逐元素应用平方根。

**数学定义:** `sqrt(x) = √x`

**示例:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

将张量元素提升到指定幂次。

**数学定义:** `pow(x, n) = x^n`

**示例:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

逐元素应用正弦函数。

**示例:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

逐元素应用余弦函数。

**示例:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

逐元素应用正切函数。

**示例:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## 聚合操作

### `sum(tensor: Tensor) -> Number`

计算所有元素的总和。

**示例:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

计算所有元素的平均值。

**示例:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

返回张量中的最大值。

**示例:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

返回张量中的最小值。

**示例:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

返回沿指定维度的最大值索引。

**参数:**
- `tensor`: 输入张量
- `dim`: 查找最大值的维度

**返回:** 索引张量

**示例:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // 沿维度 1 的最大值索引
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

返回沿指定维度的最小值索引。

**参数:**
- `tensor`: 输入张量
- `dim`: 查找最小值的维度

**返回:** 索引张量

**示例:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // 沿维度 1 的最小值索引
```

---

## 激活函数

### `relu(tensor: Tensor) -> Tensor`

整流线性单元激活。

**数学定义:** `relu(x) = max(0, x)`

**示例:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Sigmoid 激活函数。

**数学定义:** `sigmoid(x) = 1 / (1 + e^(-x))`

**示例:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

高斯误差线性单元激活(用于 BERT、GPT)。

**数学定义:** 
```
gelu(x) = x * Φ(x)
其中 Φ(x) 是标准正态分布的累积分布函数
```

**示例:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**用例:**
- BERT、GPT 模型
- 现代 Transformer 架构

---

### `tanh(tensor: Tensor) -> Tensor`

双曲正切激活。

**数学定义:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**示例:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

沿指定维度应用 softmax 归一化。

**数学定义:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**参数:**
- `tensor`: 输入张量
- `dim`: 应用 softmax 的维度

**返回:** 概率分布张量

**示例:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**用例:**
- 注意力机制
- 分类输出层
- 概率分布

---

## 矩阵运算

### `matmul(a: Tensor, b: Tensor) -> Tensor`

矩阵乘法。

**参数:**
- `a`: 左矩阵
- `b`: 右矩阵

**返回:** 矩阵乘法的结果

**示例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## 归一化

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

应用层归一化。

**数学定义:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**参数:**
- `tensor`: 输入张量
- `normalized_shape`: 要归一化的形状
- `eps`: 数值稳定性的小值(默认: 1e-5)

**示例:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**用例:**
- Transformer 层
- 循环神经网络
- 深度神经网络

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

应用批归一化。

**参数:**
- `tensor`: 输入张量
- `running_mean`: 运行均值
- `running_var`: 运行方差
- `eps`: 数值稳定性的小值

**示例:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

应用 dropout(以概率 p 随机将元素清零)。

**参数:**
- `tensor`: 输入张量
- `p`: Dropout 概率(0.0 到 1.0)

**返回:** 随机元素清零后的张量

**示例:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**用例:**
- 训练正则化
- 防止过拟合
- 集成学习

---

## 掩码操作

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

应用注意力掩码(将掩码位置设置为 -inf)。

**参数:**
- `tensor`: 注意力分数张量
- `mask`: 二进制掩码(1 = 保留, 0 = 掩码)

**返回:** 掩码后的张量

**示例:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**用例:**
- Transformer 注意力机制
- 序列掩码
- 因果(自回归)注意力

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

为可变长度序列创建填充掩码。

**参数:**
- `lengths`: 实际序列长度数组
- `max_len`: 最大序列长度

**返回:** 二进制掩码张量

**示例:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// 结果: [batch_size, max_len] 其中 1 = 真实标记, 0 = 填充
```

**用例:**
- 可变长度序列处理
- 批处理
- 注意力掩码

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

使用逻辑与组合两个掩码。

**参数:**
- `mask1`: 第一个掩码
- `mask2`: 第二个掩码

**返回:** 组合后的掩码

**示例:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**用例:**
- 组合填充和注意力掩码
- 多约束掩码
- 复杂的注意力模式

---

## 索引操作

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

使用索引沿维度收集值。

**参数:**
- `tensor`: 输入张量
- `dim`: 要收集的维度
- `indices`: 索引张量

**返回:** 收集后的张量

**示例:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**用例:**
- 标记选择
- 束搜索
- 高级索引

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

沿维度选择指定索引处的元素。

**参数:**
- `tensor`: 输入张量
- `dim`: 要选择的维度
- `indices`: 索引数组

**返回:** 选择后的张量

**示例:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // 选择第 0, 2, 5 行
```

---

## 嵌入

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

将标记 ID 转换为嵌入。

**参数:**
- `indices`: 标记 ID 序列
- `vocab_size`: 词汇表大小
- `embed_dim`: 嵌入维度

**返回:** 形状为 `[seq_len, embed_dim]` 的嵌入张量

**示例:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**用例:**
- 词嵌入
- 标记表示
- 语言模型

**参考:**
- arXiv:2510.12269 (表 1)

---

## 采样

### `top_k(logits: Tensor, k: Int) -> Tensor`

使用 top-k 采样标记。

**参数:**
- `logits`: 模型输出 logits
- `k`: 要考虑的顶部标记数量

**返回:** 采样的标记 ID

**示例:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**用例:**
- 文本生成
- 受控采样
- 多样化输出

**参考:**
- arXiv:2510.12269 (表 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

使用核采样(top-p)标记。

**参数:**
- `logits`: 模型输出 logits
- `p`: 累积概率阈值(0.0 到 1.0)

**返回:** 采样的标记 ID

**示例:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**用例:**
- 文本生成
- 动态词汇选择
- 质量控制采样

**参考:**
- arXiv:2510.12269 (表 2)

---

## 融合操作

融合操作通过减少内存开销和内核启动来组合多个操作以获得更好的性能。

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

融合加法和 ReLU 激活。

**数学定义:** `fused_add_relu(x, y) = relu(x + y)`

**示例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**性能:** 比单独操作快约 1.5 倍

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

融合乘法和 ReLU 激活。

**数学定义:** `fused_mul_relu(x, y) = relu(x * y)`

**示例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

融合仿射变换(缩放和偏移)。

**数学定义:** `fused_affine(x, s, b) = x * s + b`

**示例:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**用例:**
- 批归一化
- 层归一化
- 自定义线性变换

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

融合 GELU 激活和线性变换。

**数学定义:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**示例:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**用例:**
- Transformer FFN 层
- BERT/GPT 架构
- 性能关键路径

---

## 优化

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

执行 SGD 优化器步骤。

**数学定义:** `params_new = params - lr * gradients`

**参数:**
- `params`: 当前参数
- `gradients`: 计算的梯度
- `lr`: 学习率

**返回:** 更新后的参数

**示例:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

执行 Adam 优化器步骤。

**参数:**
- `params`: 当前参数
- `gradients`: 计算的梯度
- `m`: 一阶矩估计
- `v`: 二阶矩估计
- `lr`: 学习率
- `beta1`: 一阶矩衰减率(默认: 0.9)
- `beta2`: 二阶矩衰减率(默认: 0.999)
- `eps`: 数值稳定性常数(默认: 1e-8)

**返回:** 更新后的参数

**示例:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## 其他操作

### `tokenize(text: String) -> TokenIDs`

将文本转换为标记 ID 序列。

**参数:**
- `text`: 输入文本字符串

**返回:** TokenIDs (Vec<u32>)

**示例:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**用例:**
- 文本预处理
- 语言模型输入
- NLP 流水线

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

将张量广播到指定形状。

**参数:**
- `tensor`: 输入张量
- `shape`: 目标形状

**返回:** 广播后的张量

**示例:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**用例:**
- 形状对齐
- 批处理操作
- 不同形状的逐元素操作

---

## 运算符

TensorLogic 支持标准数学运算符:

### 算术运算符
- `+` : 加法
- `-` : 减法
- `*` : 逐元素乘法
- `/` : 逐元素除法

**示例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### 比较运算符
- `==` : 等于
- `!=` : 不等于
- `<`  : 小于
- `<=` : 小于或等于
- `>`  : 大于
- `>=` : 大于或等于

### 逻辑运算符
- `&&` : 逻辑与
- `||` : 逻辑或
- `!`  : 逻辑非

---

## 类型定义

### Tensor
通过 Metal Performance Shaders 进行 GPU 加速的多维数组。

**属性:**
- 形状: 维度数组
- 数据: Float32 元素
- 设备: Metal GPU 设备

### TokenIDs
标记 ID 序列的特殊类型。

**定义:** `Vec<u32>`

**用例:**
- 分词结果
- 嵌入查找
- 序列处理

### Number
数值(Int 或 Float)。

**变体:**
- `Integer`: 64 位有符号整数
- `Float`: 64 位浮点数

---

## 参考

- **TensorLogic 论文**: arXiv:2510.12269
- **Transformer 架构**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## 相关文档

- [入门指南](./getting_started.md)
- [语言参考](./language_reference.md)
- [示例](../../examples/)
- [2025 年新增操作](../added_operations_2025.md)
- [TODO 列表](../TODO.md)

---

**最后更新:** 2025-01-22

**TensorLogic 版本:** 0.1.1+

**总操作数:** 48 个函数 + 4 个运算符
