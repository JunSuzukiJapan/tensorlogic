# Guía de Modelos Cuantizados GGUF

Este documento explica cómo cargar y usar modelos cuantizados en formato GGUF (compatible con llama.cpp) en TensorLogic.

## Sobre el Formato GGUF

GGUF (GGML Universal Format) es un formato de cuantización eficiente para modelos de lenguaje grandes desarrollado por el proyecto llama.cpp.

### Características Principales

- Eficiencia de memoria mediante cuantización de 4-bit/8-bit (hasta 8x de compresión)
- Cuantización basada en bloques mantiene la precisión
- Compatible con llama.cpp, Ollama, LM Studio y más

### Formatos de Cuantización Soportados por TensorLogic

- ✅ **Q4_0**: Cuantización de 4-bit (máxima compresión)
- ✅ **Q8_0**: Cuantización de 8-bit (precisión y compresión equilibradas)
- ✅ **F16**: Punto flotante de 16-bit (alta precisión)
- ✅ **F32**: Punto flotante de 32-bit (máxima precisión)

## Uso Básico

### 1. Cargar Modelo Cuantizado

Automáticamente descuantizado a f16 y cargado en GPU Metal:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Obtener Tensores del Modelo

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Elegir Formato de Cuantización

### Q4_0 (4-bit)

- **Memoria**: Uso mínimo (~1/8 del modelo original)
- **Velocidad**: Inferencia más rápida
- **Precisión**: Ligera degradación (usualmente aceptable)
- **Casos de Uso**: Chatbots, generación de texto general

### Q8_0 (8-bit)

- **Memoria**: Uso moderado (~1/4 del modelo original)
- **Velocidad**: Rápido
- **Precisión**: Alta (casi equivalente a F16)
- **Casos de Uso**: Generación de alta calidad, asistentes de codificación

### F16 (16-bit)

- **Memoria**: ~1/2 del modelo original
- **Velocidad**: Estándar
- **Precisión**: Formato nativo de TensorLogic, optimizado para GPU Metal
- **Casos de Uso**: Cuando se requiere la máxima calidad

## Ejemplo Práctico: Embeddings de Tokens

```tensorlogic
// Obtener embeddings de tokens del modelo LLama
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Obtener vector de embedding desde ID de token
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Ahorro de Memoria por Cuantización

Ejemplo: Modelo LLama-7B (7 mil millones de parámetros):

| Formato    | Uso de Memoria | Compresión |
|------------|----------------|------------|
| F32 (orig) | ~28 GB         | 1x         |
| F16        | ~14 GB         | 2x         |
| Q8_0       | ~7 GB          | 4x         |
| Q4_0       | ~3.5 GB        | 8x         |

TensorLogic convierte todos los formatos a f16 al cargar y los ejecuta eficientemente en GPU Metal.

## Descargar e Instalar Modelos

### 1. Descargar Modelos GGUF de HuggingFace

Ejemplo: https://huggingface.co/TheBloke

### 2. Modelos Recomendados (para principiantes)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. Cargar en TensorLogic

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Notas Importantes

- Los modelos cuantizados son de solo lectura (no se pueden guardar desde TensorLogic)
- Use modelos no cuantizados (F16/F32) para entrenamiento
- Q4/Q8 están optimizados solo para inferencia
- Todos los formatos de cuantización se descuantizan automáticamente a f16 y se cargan en GPU

## Referencias

- [Especificación GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Modelos GGUF de HuggingFace](https://huggingface.co/TheBloke)
- [Guía de Carga de Modelos](model_loading.md)
