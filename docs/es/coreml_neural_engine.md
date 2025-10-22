# Guía de Integración CoreML & Neural Engine

Esta guía explica cómo usar modelos CoreML en TensorLogic y realizar inferencia de alta velocidad en Apple Neural Engine.

## Sobre CoreML y Neural Engine

### CoreML

- Framework de aprendizaje automático propietario de Apple
- Optimizado exclusivamente para iOS/macOS
- Aprovecha automáticamente Neural Engine, GPU y CPU
- Formato .mlmodel / .mlmodelc

### Neural Engine

- Chip dedicado a IA exclusivo de Apple Silicon
- Hasta 15.8 TOPS (M1 Pro/Max)
- Consumo de energía ultra bajo (1/10 o menos comparado con GPU)
- Optimizado para operaciones f16

### Integración con TensorLogic

- Todas las operaciones f16 (optimizado para Neural Engine)
- Integración fluida con Metal GPU
- Detección automática de formato de modelo

## Crear Modelos CoreML

Los modelos CoreML se crean típicamente con coreMLtools de Python:

```python
import coremltools as ct
import torch

# Crear modelo PyTorch
model = MyModel()
model.eval()

# Crear modelo trazado
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convertir a CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Optimización Neural Engine
    compute_precision=ct.precision.FLOAT16  # precisión f16
)

# Guardar
mlmodel.save("model.mlpackage")
```

## Usar en TensorLogic

### 1. Cargar Modelo CoreML (solo macOS)

```tensorlogic
model = load_model("model.mlpackage")
// o
model = load_model("model.mlmodelc")
```

### 2. Verificar Metadatos

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Mejores Prácticas de Optimización Neural Engine

### 1. Tipo de Datos: Usar f16

✅ Recomendado: `compute_precision=ct.precision.FLOAT16`
❌ No recomendado: FLOAT32 (ejecutado en GPU)

### 2. Formato de Modelo: Usar formato mlprogram

✅ Recomendado: `convert_to="mlprogram"`
❌ No recomendado: `convert_to="neuralnetwork"` (formato antiguo)

### 3. Tamaño de Lote: 1 es óptimo

✅ Recomendado: `batch_size=1`
⚠️ Nota: `batch_size>1` puede ejecutarse en GPU

### 4. Tamaño de Entrada: Tamaño fijo es óptimo

✅ Recomendado: `shape=[1, 3, 224, 224]`
⚠️ Nota: Tamaños variables tienen optimización limitada

## Operaciones Soportadas

### Operaciones ejecutadas rápidamente en Neural Engine

- ✅ Convoluciones (conv2d, depthwise_conv)
- ✅ Capas totalmente conectadas (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalización (batch_norm, layer_norm)
- ✅ Funciones de activación (relu, gelu, sigmoid, tanh)
- ✅ Operaciones elemento a elemento (add, mul, sub, div)

## Comparación de Rendimiento

Inferencia ResNet-50 (imagen 224x224):

| Dispositivo        | Latencia | Potencia | Eficiencia |
|-------------------|----------|----------|------------|
| Neural Engine     | ~3ms     | ~0.5W    | Máxima     |
| Metal GPU (M1)    | ~8ms     | ~5W      | Media      |
| CPU (M1)          | ~50ms    | ~2W      | Baja       |

## Selección de Formato de Modelo

### Formatos Recomendados por Caso de Uso

**Entrenamiento**: SafeTensors
- Compatible con PyTorch
- Guardar/cargar pesos
- Entrenar en Metal GPU

**Inferencia (iOS/macOS)**: CoreML
- Optimización Neural Engine
- Consumo de energía ultra bajo
- Integración de aplicaciones

**Inferencia (General)**: GGUF
- Soporte de cuantización
- Multiplataforma
- Eficiente en memoria

## Referencias

- [Documentación Oficial de CoreML](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Guía Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Guía de Carga de Modelos](model_loading.md)
