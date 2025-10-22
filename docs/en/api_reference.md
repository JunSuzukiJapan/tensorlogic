# TensorLogic API Reference

Complete API reference for all available operations in TensorLogic.

## Table of Contents

1. [Tensor Creation](#tensor-creation)
2. [Shape Operations](#shape-operations)
3. [Mathematical Functions](#mathematical-functions)
4. [Aggregation Operations](#aggregation-operations)
5. [Activation Functions](#activation-functions)
6. [Matrix Operations](#matrix-operations)
7. [Normalization](#normalization)
8. [Masking Operations](#masking-operations)
9. [Indexing Operations](#indexing-operations)
10. [Embeddings](#embeddings)
11. [Sampling](#sampling)
12. [Fused Operations](#fused-operations)
13. [Optimization](#optimization)
14. [Other Operations](#other-operations)
15. [Operators](#operators)
16. [Type Definitions](#type-definitions)

---

## Tensor Creation

### `zeros(shape: Array<Int>) -> Tensor`

Creates a tensor filled with zeros.

**Parameters:**
- `shape`: Array specifying tensor dimensions

**Returns:** Tensor filled with 0

**Example:**
```tensorlogic
let z = zeros([2, 3])  // 2x3 tensor of zeros
```

---

### `ones(shape: Array<Int>) -> Tensor`

Creates a tensor filled with ones.

**Parameters:**
- `shape`: Array specifying tensor dimensions

**Returns:** Tensor filled with 1

**Example:**
```tensorlogic
let o = ones([2, 3])  // 2x3 tensor of ones
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Generates sinusoidal positional encoding for Transformers.

**Parameters:**
- `seq_len`: Sequence length
- `d_model`: Model dimension

**Returns:** Tensor of shape `[seq_len, d_model]`

**Mathematical Definition:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Example:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Use Cases:**
- Transformer models
- Sequence-to-sequence models
- Attention mechanisms

**References:**
- arXiv:2510.12269 (Table 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Shape Operations

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Changes the tensor shape while preserving data.

**Parameters:**
- `tensor`: Input tensor
- `new_shape`: Target shape

**Returns:** Reshaped tensor

**Example:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Constraints:**
- Total number of elements must remain the same

---

### `flatten(tensor: Tensor) -> Tensor`

Flattens tensor to 1D.

**Parameters:**
- `tensor`: Input tensor

**Returns:** 1D tensor

**Example:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Transposes a 2D tensor (swaps axes).

**Parameters:**
- `tensor`: Input 2D tensor

**Returns:** Transposed tensor

**Example:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Reorders tensor dimensions.

**Parameters:**
- `tensor`: Input tensor
- `dims`: New dimension order

**Returns:** Permuted tensor

**Example:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Adds a dimension of size 1 at the specified position.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Position to insert new dimension

**Returns:** Tensor with added dimension

**Example:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Removes all dimensions of size 1.

**Parameters:**
- `tensor`: Input tensor

**Returns:** Tensor with size-1 dimensions removed

**Example:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Splits tensor into multiple tensors along specified dimension.

**Parameters:**
- `tensor`: Input tensor
- `sizes`: Size of each split section
- `dim`: Dimension to split along

**Returns:** Array of tensors

**Example:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 tensors: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Splits tensor into specified number of chunks.

**Parameters:**
- `tensor`: Input tensor
- `chunks`: Number of chunks
- `dim`: Dimension to split along

**Returns:** Array of tensors

**Example:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 tensors of [4,4] each
```

---

## Mathematical Functions

### `exp(tensor: Tensor) -> Tensor`

Applies exponential function element-wise.

**Mathematical Definition:** `exp(x) = e^x`

**Example:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Applies natural logarithm element-wise.

**Mathematical Definition:** `log(x) = ln(x)`

**Example:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Applies square root element-wise.

**Mathematical Definition:** `sqrt(x) = √x`

**Example:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Raises tensor elements to specified power.

**Mathematical Definition:** `pow(x, n) = x^n`

**Example:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Applies sine function element-wise.

**Example:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Applies cosine function element-wise.

**Example:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Applies tangent function element-wise.

**Example:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Aggregation Operations

### `sum(tensor: Tensor) -> Number`

Computes sum of all elements.

**Example:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Computes mean of all elements.

**Example:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Returns maximum value in tensor.

**Example:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Returns minimum value in tensor.

**Example:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Returns indices of maximum values along specified dimension.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension to find maximum along

**Returns:** Tensor of indices

**Example:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Maximum indices along dimension 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Returns indices of minimum values along specified dimension.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension to find minimum along

**Returns:** Tensor of indices

**Example:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Minimum indices along dimension 1
```

---

## Activation Functions

### `relu(tensor: Tensor) -> Tensor`

Rectified Linear Unit activation.

**Mathematical Definition:** `relu(x) = max(0, x)`

**Example:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Sigmoid activation function.

**Mathematical Definition:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Example:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Gaussian Error Linear Unit activation (used in BERT, GPT).

**Mathematical Definition:** 
```
gelu(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard normal distribution
```

**Example:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Use Cases:**
- BERT, GPT models
- Modern transformer architectures

---

### `tanh(tensor: Tensor) -> Tensor`

Hyperbolic tangent activation.

**Mathematical Definition:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Example:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Applies softmax normalization along specified dimension.

**Mathematical Definition:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension to apply softmax

**Returns:** Probability distribution tensor

**Example:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Use Cases:**
- Attention mechanisms
- Classification output layers
- Probability distributions

---

## Matrix Operations

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Matrix multiplication.

**Parameters:**
- `a`: Left matrix
- `b`: Right matrix

**Returns:** Result of matrix multiplication

**Example:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalization

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Applies layer normalization.

**Mathematical Definition:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Parameters:**
- `tensor`: Input tensor
- `normalized_shape`: Shape to normalize over
- `eps`: Small value for numerical stability (default: 1e-5)

**Example:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Use Cases:**
- Transformer layers
- Recurrent networks
- Deep neural networks

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Applies batch normalization.

**Parameters:**
- `tensor`: Input tensor
- `running_mean`: Running mean
- `running_var`: Running variance
- `eps`: Small value for numerical stability

**Example:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Applies dropout (randomly zeros elements with probability p).

**Parameters:**
- `tensor`: Input tensor
- `p`: Dropout probability (0.0 to 1.0)

**Returns:** Tensor with random elements zeroed

**Example:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Use Cases:**
- Training regularization
- Preventing overfitting
- Ensemble learning

---

## Masking Operations

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Applies attention mask (sets masked positions to -inf).

**Parameters:**
- `tensor`: Attention scores tensor
- `mask`: Binary mask (1 = keep, 0 = mask)

**Returns:** Masked tensor

**Example:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Use Cases:**
- Transformer attention mechanisms
- Sequence masking
- Causal (autoregressive) attention

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Creates padding mask for variable-length sequences.

**Parameters:**
- `lengths`: Array of actual sequence lengths
- `max_len`: Maximum sequence length

**Returns:** Binary mask tensor

**Example:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Result: [batch_size, max_len] where 1 = real token, 0 = padding
```

**Use Cases:**
- Variable-length sequence handling
- Batch processing
- Attention masking

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Combines two masks using logical AND.

**Parameters:**
- `mask1`: First mask
- `mask2`: Second mask

**Returns:** Combined mask

**Example:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Use Cases:**
- Combining padding and attention masks
- Multi-constraint masking
- Complex attention patterns

---

## Indexing Operations

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Gathers values along dimension using indices.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension to gather along
- `indices`: Index tensor

**Returns:** Gathered tensor

**Example:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Use Cases:**
- Token selection
- Beam search
- Advanced indexing

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Selects elements at specified indices along dimension.

**Parameters:**
- `tensor`: Input tensor
- `dim`: Dimension to select from
- `indices`: Array of indices

**Returns:** Selected tensor

**Example:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Select rows 0, 2, 5
```

---

## Embeddings

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Converts token IDs to embeddings.

**Parameters:**
- `indices`: Token ID sequence
- `vocab_size`: Vocabulary size
- `embed_dim`: Embedding dimension

**Returns:** Embedding tensor of shape `[seq_len, embed_dim]`

**Example:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Use Cases:**
- Word embeddings
- Token representations
- Language models

**References:**
- arXiv:2510.12269 (Table 1)

---

## Sampling

### `top_k(logits: Tensor, k: Int) -> Tensor`

Samples token using top-k sampling.

**Parameters:**
- `logits`: Model output logits
- `k`: Number of top tokens to consider

**Returns:** Sampled token ID

**Example:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Use Cases:**
- Text generation
- Controlled sampling
- Diverse outputs

**References:**
- arXiv:2510.12269 (Table 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Samples token using nucleus (top-p) sampling.

**Parameters:**
- `logits`: Model output logits
- `p`: Cumulative probability threshold (0.0 to 1.0)

**Returns:** Sampled token ID

**Example:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Use Cases:**
- Text generation
- Dynamic vocabulary selection
- Quality-controlled sampling

**References:**
- arXiv:2510.12269 (Table 2)

---

## Fused Operations

Fused operations combine multiple operations for better performance by reducing memory overhead and kernel launches.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fuses addition and ReLU activation.

**Mathematical Definition:** `fused_add_relu(x, y) = relu(x + y)`

**Example:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Performance:** ~1.5x faster than separate operations

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fuses multiplication and ReLU activation.

**Mathematical Definition:** `fused_mul_relu(x, y) = relu(x * y)`

**Example:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Fuses affine transformation (scale and shift).

**Mathematical Definition:** `fused_affine(x, s, b) = x * s + b`

**Example:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Use Cases:**
- Batch normalization
- Layer normalization
- Custom linear transformations

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Fuses GELU activation and linear transformation.

**Mathematical Definition:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Example:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Use Cases:**
- Transformer FFN layers
- BERT/GPT architectures
- Performance-critical paths

---

## Optimization

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Performs SGD optimizer step.

**Mathematical Definition:** `params_new = params - lr * gradients`

**Parameters:**
- `params`: Current parameters
- `gradients`: Computed gradients
- `lr`: Learning rate

**Returns:** Updated parameters

**Example:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Performs Adam optimizer step.

**Parameters:**
- `params`: Current parameters
- `gradients`: Computed gradients
- `m`: First moment estimate
- `v`: Second moment estimate
- `lr`: Learning rate
- `beta1`: First moment decay rate (default: 0.9)
- `beta2`: Second moment decay rate (default: 0.999)
- `eps`: Numerical stability constant (default: 1e-8)

**Returns:** Updated parameters

**Example:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Other Operations

### `tokenize(text: String) -> TokenIDs`

Converts text to token ID sequence.

**Parameters:**
- `text`: Input text string

**Returns:** TokenIDs (Vec<u32>)

**Example:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Use Cases:**
- Text preprocessing
- Language model input
- NLP pipelines

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Broadcasts tensor to specified shape.

**Parameters:**
- `tensor`: Input tensor
- `shape`: Target shape

**Returns:** Broadcasted tensor

**Example:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Use Cases:**
- Shape alignment
- Batch operations
- Element-wise operations with different shapes

---

## Operators

TensorLogic supports standard mathematical operators:

### Arithmetic Operators
- `+` : Addition
- `-` : Subtraction
- `*` : Element-wise multiplication
- `/` : Element-wise division

**Example:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Comparison Operators
- `==` : Equal
- `!=` : Not equal
- `<`  : Less than
- `<=` : Less than or equal
- `>`  : Greater than
- `>=` : Greater than or equal

### Logical Operators
- `&&` : Logical AND
- `||` : Logical OR
- `!`  : Logical NOT

---

## Type Definitions

### Tensor
Multi-dimensional array with GPU acceleration via Metal Performance Shaders.

**Properties:**
- Shape: Array of dimensions
- Data: Float32 elements
- Device: Metal GPU device

### TokenIDs
Special type for token ID sequences.

**Definition:** `Vec<u32>`

**Use Cases:**
- Tokenization results
- Embedding lookups
- Sequence processing

### Number
Numeric values (Int or Float).

**Variants:**
- `Integer`: 64-bit signed integer
- `Float`: 64-bit floating point

---

## References

- **TensorLogic Paper**: arXiv:2510.12269
- **Transformer Architecture**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Related Documentation

- [Getting Started Guide](./getting_started.md)
- [Language Reference](./language_reference.md)
- [Examples](../../examples/)
- [2025 Added Operations](../added_operations_2025.md)
- [TODO List](../TODO.md)

---

**Last Updated:** 2025-01-22

**TensorLogic Version:** 0.1.1+

**Total Operations:** 48 functions + 4 operators
