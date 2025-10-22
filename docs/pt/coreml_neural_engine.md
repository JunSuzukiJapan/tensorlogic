# Guia de Integração CoreML & Neural Engine

Este guia explica como usar modelos CoreML no TensorLogic e realizar inferência de alta velocidade no Apple Neural Engine.

## Sobre CoreML e Neural Engine

### CoreML

- Framework de aprendizado de máquina proprietário da Apple
- Otimizado exclusivamente para iOS/macOS
- Aproveita automaticamente Neural Engine, GPU e CPU
- Formato .mlmodel / .mlmodelc

### Neural Engine

- Chip dedicado a IA exclusivo do Apple Silicon
- Até 15,8 TOPS (M1 Pro/Max)
- Consumo de energia ultra baixo (1/10 ou menos comparado à GPU)
- Otimizado para operações f16

### Integração com TensorLogic

- Todas as operações f16 (otimizado para Neural Engine)
- Integração perfeita com Metal GPU
- Detecção automática de formato de modelo

## Criar Modelos CoreML

Modelos CoreML são tipicamente criados com coreMLtools do Python:

```python
import coremltools as ct
import torch

# Criar modelo PyTorch
model = MyModel()
model.eval()

# Criar modelo rastreado
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Converter para CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Otimização Neural Engine
    compute_precision=ct.precision.FLOAT16  # precisão f16
)

# Salvar
mlmodel.save("model.mlpackage")
```

## Usar no TensorLogic

### 1. Carregar Modelo CoreML (apenas macOS)

```tensorlogic
model = load_model("model.mlpackage")
// ou
model = load_model("model.mlmodelc")
```

### 2. Verificar Metadados

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Melhores Práticas de Otimização Neural Engine

### 1. Tipo de Dados: Usar f16

✅ Recomendado: `compute_precision=ct.precision.FLOAT16`
❌ Não recomendado: FLOAT32 (executado na GPU)

### 2. Formato de Modelo: Usar formato mlprogram

✅ Recomendado: `convert_to="mlprogram"`
❌ Não recomendado: `convert_to="neuralnetwork"` (formato legado)

### 3. Tamanho de Lote: 1 é ótimo

✅ Recomendado: `batch_size=1`
⚠️ Nota: `batch_size>1` pode executar na GPU

### 4. Tamanho de Entrada: Tamanho fixo é ótimo

✅ Recomendado: `shape=[1, 3, 224, 224]`
⚠️ Nota: Tamanhos variáveis têm otimização limitada

## Operações Suportadas

### Operações executadas rapidamente no Neural Engine

- ✅ Convoluções (conv2d, depthwise_conv)
- ✅ Camadas totalmente conectadas (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalização (batch_norm, layer_norm)
- ✅ Funções de ativação (relu, gelu, sigmoid, tanh)
- ✅ Operações elemento a elemento (add, mul, sub, div)

## Comparação de Desempenho

Inferência ResNet-50 (imagem 224x224):

| Dispositivo        | Latência | Potência | Eficiência |
|-------------------|----------|----------|------------|
| Neural Engine     | ~3ms     | ~0.5W    | Máxima     |
| Metal GPU (M1)    | ~8ms     | ~5W      | Média      |
| CPU (M1)          | ~50ms    | ~2W      | Baixa      |

## Seleção de Formato de Modelo

### Formatos Recomendados por Caso de Uso

**Treinamento**: SafeTensors
- Compatível com PyTorch
- Salvar/carregar pesos
- Treinar na Metal GPU

**Inferência (iOS/macOS)**: CoreML
- Otimização Neural Engine
- Consumo de energia ultra baixo
- Integração de aplicativo

**Inferência (Geral)**: GGUF
- Suporte de quantização
- Multi-plataforma
- Eficiente em memória

## Referências

- [Documentação Oficial CoreML](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Guia Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Guia de Carregamento de Modelos](model_loading.md)
