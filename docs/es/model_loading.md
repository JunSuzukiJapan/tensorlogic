# Guía de Carga de Modelos

Este documento explica cómo cargar y usar modelos de PyTorch y HuggingFace en TensorLogic. Soporta el formato SafeTensors (compatible con PyTorch) y el formato GGUF (LLMs cuantizados).

## Uso Básico

### 1. Cargar Modelo SafeTensors (guardado desde PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Cargar Modelo GGUF (LLM cuantizado)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Obtener Tensores del Modelo

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Ejemplo Práctico: Inferencia de Capa Lineal

Realizar inferencia usando pesos y sesgos del modelo:

```tensorlogic
function forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Transformación lineal: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Preparar Modelos de PyTorch

Guarde su modelo en formato SafeTensors usando Python:

```python
import torch
from safetensors.torch import save_file

# Crear modelo PyTorch
model = MyModel()

# Obtener pesos del modelo como diccionario
tensors = {name: param for name, param in model.named_parameters()}

# Guardar en formato SafeTensors
save_file(tensors, "model.safetensors")
```

Luego cargar en TensorLogic:

```tensorlogic
model = load_model("model.safetensors")
```

## Formatos Soportados

### 1. SafeTensors (.safetensors)

- Compatible con PyTorch y HuggingFace
- Soporta tipos de datos F32, F64, F16, BF16
- Todos los datos se convierten automáticamente a f16
- Cargado directamente en GPU Metal

### 2. GGUF (.gguf)

- Modelos cuantizados en formato llama.cpp
- Soporta Q4_0, Q8_0, F32, F16
- Cargado directamente en GPU Metal

### 3. CoreML (.mlmodel, .mlpackage)

- Modelos optimizados para Apple Neural Engine
- Solo iOS/macOS

## Ejemplo Completo de Modelo Lineal

```tensorlogic
// Datos de entrada (tamaño de lote 4, dimensión de características 3)
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

// Sesgo (2 dimensiones)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Ejecutar inferencia
let output = forward(X, W, b)

// Imprimir resultados
print("Output shape:", output.shape)
print("Output:", output)
```

## Guardar Modelos

Puede guardar modelos de TensorLogic en formato SafeTensors:

```tensorlogic
save_model(model, "output.safetensors")
```

Esto permite la interoperabilidad con PyTorch y HuggingFace.

## Notas Importantes

- TensorLogic ejecuta todas las operaciones en f16 (optimizado para GPU Metal)
- Otros tipos de datos se convierten automáticamente a f16 durante la carga
- Los tipos enteros (i8, i32, etc.) no están soportados (solo punto flotante)
- Los modelos grandes se cargan automáticamente en la memoria GPU Metal

## Documentación Relacionada

- [Modelos Cuantizados GGUF](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Guía de Inicio](../claudedocs/getting_started.md)
