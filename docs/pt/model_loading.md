# Guia de Carregamento de Modelos

Este documento explica como carregar e usar modelos PyTorch e HuggingFace no TensorLogic. Suporta o formato SafeTensors (compatível com PyTorch) e o formato GGUF (LLMs quantizados).

## Uso Básico

### 1. Carregar Modelo SafeTensors (salvo do PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Carregar Modelo GGUF (LLM quantizado)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Obter Tensores do Modelo

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Exemplo Prático: Inferência de Camada Linear

Realizar inferência usando pesos e vieses do modelo:

```tensorlogic
fn forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Transformação linear: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Preparar Modelos PyTorch

Salve seu modelo no formato SafeTensors usando Python:

```python
import torch
from safetensors.torch import save_file

# Criar modelo PyTorch
model = MyModel()

# Obter pesos do modelo como dicionário
tensors = {name: param for name, param in model.named_parameters()}

# Salvar no formato SafeTensors
save_file(tensors, "model.safetensors")
```

Depois carregue no TensorLogic:

```tensorlogic
model = load_model("model.safetensors")
```

## Formatos Suportados

### 1. SafeTensors (.safetensors)

- Compatível com PyTorch e HuggingFace
- Suporta tipos de dados F32, F64, F16, BF16
- Todos os dados são automaticamente convertidos para f16
- Carregado diretamente na GPU Metal

### 2. GGUF (.gguf)

- Modelos quantizados no formato llama.cpp
- Suporta Q4_0, Q8_0, F32, F16
- Carregado diretamente na GPU Metal

### 3. CoreML (.mlmodel, .mlpackage)

- Modelos otimizados para Apple Neural Engine
- Apenas iOS/macOS

## Exemplo Completo de Modelo Linear

```tensorlogic
// Dados de entrada (tamanho de lote 4, dimensão de características 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Matriz de pesos (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Viés (2 dimensões)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Executar inferência
let output = forward(X, W, b)

// Imprimir resultados
print("Output shape:", output.shape)
print("Output:", output)
```

## Salvar Modelos

Você pode salvar modelos TensorLogic no formato SafeTensors:

```tensorlogic
save_model(model, "output.safetensors")
```

Isso permite a interoperabilidade com PyTorch e HuggingFace.

## Notas Importantes

- TensorLogic executa todas as operações em f16 (otimizado para GPU Metal)
- Outros tipos de dados são automaticamente convertidos para f16 durante o carregamento
- Tipos inteiros (i8, i32, etc.) não são suportados (apenas ponto flutuante)
- Modelos grandes são automaticamente carregados na memória da GPU Metal

## Documentação Relacionada

- [Modelos Quantizados GGUF](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Guia de Introdução](../claudedocs/getting_started.md)
