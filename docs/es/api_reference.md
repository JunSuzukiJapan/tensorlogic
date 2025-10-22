# Referencia de API de TensorLogic

Referencia completa de API para todas las operaciones disponibles en TensorLogic.

## Tabla de Contenidos

1. [Creación de Tensores](#creación-de-tensores)
2. [Operaciones de Forma](#operaciones-de-forma)
3. [Funciones Matemáticas](#funciones-matemáticas)
4. [Operaciones de Agregación](#operaciones-de-agregación)
5. [Funciones de Activación](#funciones-de-activación)
6. [Operaciones de Matriz](#operaciones-de-matriz)
7. [Normalización](#normalización)
8. [Operaciones de Enmascaramiento](#operaciones-de-enmascaramiento)
9. [Operaciones de Indexación](#operaciones-de-indexación)
10. [Incrustaciones](#incrustaciones)
11. [Muestreo](#muestreo)
12. [Operaciones Fusionadas](#operaciones-fusionadas)
13. [Optimización](#optimización)
14. [Otras Operaciones](#otras-operaciones)
15. [Operadores](#operadores)
16. [Definiciones de Tipos](#definiciones-de-tipos)

---

## Creación de Tensores

### `zeros(shape: Array<Int>) -> Tensor`

Crea un tensor lleno de ceros.

**Parámetros:**
- `shape`: Array que especifica las dimensiones del tensor

**Retorna:** Tensor lleno de 0

**Ejemplo:**
```tensorlogic
let z = zeros([2, 3])  // Tensor 2x3 de ceros
```

---

### `ones(shape: Array<Int>) -> Tensor`

Crea un tensor lleno de unos.

**Parámetros:**
- `shape`: Array que especifica las dimensiones del tensor

**Retorna:** Tensor lleno de 1

**Ejemplo:**
```tensorlogic
let o = ones([2, 3])  // Tensor 2x3 de unos
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Genera codificación posicional sinusoidal para Transformers.

**Parámetros:**
- `seq_len`: Longitud de secuencia
- `d_model`: Dimensión del modelo

**Retorna:** Tensor de forma `[seq_len, d_model]`

**Definición Matemática:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Ejemplo:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Casos de Uso:**
- Modelos Transformer
- Modelos de secuencia a secuencia
- Mecanismos de atención

**Referencias:**
- arXiv:2510.12269 (Tabla 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Operaciones de Forma

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Cambia la forma del tensor preservando los datos.

**Parámetros:**
- `tensor`: Tensor de entrada
- `new_shape`: Forma objetivo

**Retorna:** Tensor reformado

**Ejemplo:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Restricciones:**
- El número total de elementos debe permanecer igual

---

### `flatten(tensor: Tensor) -> Tensor`

Aplana el tensor a 1D.

**Parámetros:**
- `tensor`: Tensor de entrada

**Retorna:** Tensor 1D

**Ejemplo:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Transpone un tensor 2D (intercambia ejes).

**Parámetros:**
- `tensor`: Tensor 2D de entrada

**Retorna:** Tensor transpuesto

**Ejemplo:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Reordena las dimensiones del tensor.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dims`: Nuevo orden de dimensiones

**Retorna:** Tensor permutado

**Ejemplo:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Añade una dimensión de tamaño 1 en la posición especificada.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Posición para insertar la nueva dimensión

**Retorna:** Tensor con dimensión añadida

**Ejemplo:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Elimina todas las dimensiones de tamaño 1.

**Parámetros:**
- `tensor`: Tensor de entrada

**Retorna:** Tensor con dimensiones de tamaño 1 eliminadas

**Ejemplo:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Divide el tensor en múltiples tensores a lo largo de la dimensión especificada.

**Parámetros:**
- `tensor`: Tensor de entrada
- `sizes`: Tamaño de cada sección dividida
- `dim`: Dimensión para dividir

**Retorna:** Array de tensores

**Ejemplo:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 tensores: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Divide el tensor en el número especificado de fragmentos.

**Parámetros:**
- `tensor`: Tensor de entrada
- `chunks`: Número de fragmentos
- `dim`: Dimensión para dividir

**Retorna:** Array de tensores

**Ejemplo:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 tensores de [4,4] cada uno
```

---

## Funciones Matemáticas

### `exp(tensor: Tensor) -> Tensor`

Aplica la función exponencial elemento por elemento.

**Definición Matemática:** `exp(x) = e^x`

**Ejemplo:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Aplica el logaritmo natural elemento por elemento.

**Definición Matemática:** `log(x) = ln(x)`

**Ejemplo:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Aplica la raíz cuadrada elemento por elemento.

**Definición Matemática:** `sqrt(x) = √x`

**Ejemplo:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Eleva los elementos del tensor a la potencia especificada.

**Definición Matemática:** `pow(x, n) = x^n`

**Ejemplo:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Aplica la función seno elemento por elemento.

**Ejemplo:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Aplica la función coseno elemento por elemento.

**Ejemplo:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Aplica la función tangente elemento por elemento.

**Ejemplo:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Operaciones de Agregación

### `sum(tensor: Tensor) -> Number`

Calcula la suma de todos los elementos.

**Ejemplo:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Calcula la media de todos los elementos.

**Ejemplo:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Devuelve el valor máximo en el tensor.

**Ejemplo:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Devuelve el valor mínimo en el tensor.

**Ejemplo:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Devuelve los índices de los valores máximos a lo largo de la dimensión especificada.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensión para encontrar el máximo

**Retorna:** Tensor de índices

**Ejemplo:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Índices máximos a lo largo de la dimensión 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Devuelve los índices de los valores mínimos a lo largo de la dimensión especificada.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensión para encontrar el mínimo

**Retorna:** Tensor de índices

**Ejemplo:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Índices mínimos a lo largo de la dimensión 1
```

---

## Funciones de Activación

### `relu(tensor: Tensor) -> Tensor`

Activación Rectified Linear Unit.

**Definición Matemática:** `relu(x) = max(0, x)`

**Ejemplo:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Función de activación sigmoide.

**Definición Matemática:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Ejemplo:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Activación Gaussian Error Linear Unit (usado en BERT, GPT).

**Definición Matemática:** 
```
gelu(x) = x * Φ(x)
donde Φ(x) es la función de distribución acumulativa de la distribución normal estándar
```

**Ejemplo:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Casos de Uso:**
- Modelos BERT, GPT
- Arquitecturas Transformer modernas

---

### `tanh(tensor: Tensor) -> Tensor`

Activación tangente hiperbólica.

**Definición Matemática:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Ejemplo:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Aplica normalización softmax a lo largo de la dimensión especificada.

**Definición Matemática:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensión para aplicar softmax

**Retorna:** Tensor de distribución de probabilidad

**Ejemplo:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Casos de Uso:**
- Mecanismos de atención
- Capas de salida de clasificación
- Distribuciones de probabilidad

---

## Operaciones de Matriz

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Multiplicación de matrices.

**Parámetros:**
- `a`: Matriz izquierda
- `b`: Matriz derecha

**Retorna:** Resultado de la multiplicación de matrices

**Ejemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalización

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Aplica normalización de capa.

**Definición Matemática:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Parámetros:**
- `tensor`: Tensor de entrada
- `normalized_shape`: Forma para normalizar
- `eps`: Valor pequeño para estabilidad numérica (predeterminado: 1e-5)

**Ejemplo:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Casos de Uso:**
- Capas Transformer
- Redes recurrentes
- Redes neuronales profundas

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Aplica normalización por lotes.

**Parámetros:**
- `tensor`: Tensor de entrada
- `running_mean`: Media móvil
- `running_var`: Varianza móvil
- `eps`: Valor pequeño para estabilidad numérica

**Ejemplo:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Aplica dropout (pone a cero elementos aleatoriamente con probabilidad p).

**Parámetros:**
- `tensor`: Tensor de entrada
- `p`: Probabilidad de dropout (0.0 a 1.0)

**Retorna:** Tensor con elementos aleatorios puestos a cero

**Ejemplo:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Casos de Uso:**
- Regularización de entrenamiento
- Prevención de sobreajuste
- Aprendizaje en conjunto

---

## Operaciones de Enmascaramiento

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Aplica máscara de atención (establece posiciones enmascaradas a -inf).

**Parámetros:**
- `tensor`: Tensor de puntuaciones de atención
- `mask`: Máscara binaria (1 = mantener, 0 = enmascarar)

**Retorna:** Tensor enmascarado

**Ejemplo:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Casos de Uso:**
- Mecanismos de atención Transformer
- Enmascaramiento de secuencias
- Atención causal (autoregresiva)

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Crea máscara de relleno para secuencias de longitud variable.

**Parámetros:**
- `lengths`: Array de longitudes de secuencia reales
- `max_len`: Longitud máxima de secuencia

**Retorna:** Tensor de máscara binaria

**Ejemplo:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Resultado: [batch_size, max_len] donde 1 = token real, 0 = relleno
```

**Casos de Uso:**
- Manejo de secuencias de longitud variable
- Procesamiento por lotes
- Enmascaramiento de atención

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Combina dos máscaras usando AND lógico.

**Parámetros:**
- `mask1`: Primera máscara
- `mask2`: Segunda máscara

**Retorna:** Máscara combinada

**Ejemplo:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Casos de Uso:**
- Combinación de máscaras de relleno y atención
- Enmascaramiento multi-restricción
- Patrones de atención complejos

---

## Operaciones de Indexación

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Recopila valores a lo largo de la dimensión usando índices.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensión para recopilar
- `indices`: Tensor de índices

**Retorna:** Tensor recopilado

**Ejemplo:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Casos de Uso:**
- Selección de tokens
- Búsqueda en haz
- Indexación avanzada

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Selecciona elementos en índices especificados a lo largo de la dimensión.

**Parámetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensión para seleccionar
- `indices`: Array de índices

**Retorna:** Tensor seleccionado

**Ejemplo:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Seleccionar filas 0, 2, 5
```

---

## Incrustaciones

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Convierte IDs de tokens en incrustaciones.

**Parámetros:**
- `indices`: Secuencia de IDs de tokens
- `vocab_size`: Tamaño del vocabulario
- `embed_dim`: Dimensión de incrustación

**Retorna:** Tensor de incrustación de forma `[seq_len, embed_dim]`

**Ejemplo:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Casos de Uso:**
- Incrustaciones de palabras
- Representaciones de tokens
- Modelos de lenguaje

**Referencias:**
- arXiv:2510.12269 (Tabla 1)

---

## Muestreo

### `top_k(logits: Tensor, k: Int) -> Tensor`

Muestrea token usando muestreo top-k.

**Parámetros:**
- `logits`: Logits de salida del modelo
- `k`: Número de tokens principales a considerar

**Retorna:** ID de token muestreado

**Ejemplo:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Casos de Uso:**
- Generación de texto
- Muestreo controlado
- Salidas diversas

**Referencias:**
- arXiv:2510.12269 (Tabla 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Muestrea token usando muestreo nucleus (top-p).

**Parámetros:**
- `logits`: Logits de salida del modelo
- `p`: Umbral de probabilidad acumulativa (0.0 a 1.0)

**Retorna:** ID de token muestreado

**Ejemplo:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Casos de Uso:**
- Generación de texto
- Selección dinámica de vocabulario
- Muestreo con control de calidad

**Referencias:**
- arXiv:2510.12269 (Tabla 2)

---

## Operaciones Fusionadas

Las operaciones fusionadas combinan múltiples operaciones para mejor rendimiento reduciendo el overhead de memoria y lanzamientos de kernel.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusiona suma y activación ReLU.

**Definición Matemática:** `fused_add_relu(x, y) = relu(x + y)`

**Ejemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Rendimiento:** ~1.5x más rápido que operaciones separadas

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusiona multiplicación y activación ReLU.

**Definición Matemática:** `fused_mul_relu(x, y) = relu(x * y)`

**Ejemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Fusiona transformación afín (escala y desplazamiento).

**Definición Matemática:** `fused_affine(x, s, b) = x * s + b`

**Ejemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Casos de Uso:**
- Normalización por lotes
- Normalización de capa
- Transformaciones lineales personalizadas

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Fusiona activación GELU y transformación lineal.

**Definición Matemática:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Ejemplo:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Casos de Uso:**
- Capas FFN de Transformer
- Arquitecturas BERT/GPT
- Rutas críticas de rendimiento

---

## Optimización

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Realiza paso de optimizador SGD.

**Definición Matemática:** `params_new = params - lr * gradients`

**Parámetros:**
- `params`: Parámetros actuales
- `gradients`: Gradientes calculados
- `lr`: Tasa de aprendizaje

**Retorna:** Parámetros actualizados

**Ejemplo:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Realiza paso de optimizador Adam.

**Parámetros:**
- `params`: Parámetros actuales
- `gradients`: Gradientes calculados
- `m`: Estimación del primer momento
- `v`: Estimación del segundo momento
- `lr`: Tasa de aprendizaje
- `beta1`: Tasa de decaimiento del primer momento (predeterminado: 0.9)
- `beta2`: Tasa de decaimiento del segundo momento (predeterminado: 0.999)
- `eps`: Constante de estabilidad numérica (predeterminado: 1e-8)

**Retorna:** Parámetros actualizados

**Ejemplo:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Otras Operaciones

### `tokenize(text: String) -> TokenIDs`

Convierte texto en secuencia de IDs de tokens.

**Parámetros:**
- `text`: Cadena de texto de entrada

**Retorna:** TokenIDs (Vec<u32>)

**Ejemplo:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Casos de Uso:**
- Preprocesamiento de texto
- Entrada de modelo de lenguaje
- Pipelines de NLP

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Transmite tensor a la forma especificada.

**Parámetros:**
- `tensor`: Tensor de entrada
- `shape`: Forma objetivo

**Retorna:** Tensor transmitido

**Ejemplo:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Casos de Uso:**
- Alineación de formas
- Operaciones por lotes
- Operaciones elemento por elemento con diferentes formas

---

## Operadores

TensorLogic soporta operadores matemáticos estándar:

### Operadores Aritméticos
- `+` : Suma
- `-` : Resta
- `*` : Multiplicación elemento por elemento
- `/` : División elemento por elemento

**Ejemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Operadores de Comparación
- `==` : Igual
- `!=` : No igual
- `<`  : Menor que
- `<=` : Menor o igual que
- `>`  : Mayor que
- `>=` : Mayor o igual que

### Operadores Lógicos
- `&&` : AND lógico
- `||` : OR lógico
- `!`  : NOT lógico

---

## Definiciones de Tipos

### Tensor
Array multidimensional con aceleración GPU mediante Metal Performance Shaders.

**Propiedades:**
- Forma: Array de dimensiones
- Datos: Elementos Float32
- Dispositivo: Dispositivo GPU Metal

### TokenIDs
Tipo especial para secuencias de IDs de tokens.

**Definición:** `Vec<u32>`

**Casos de Uso:**
- Resultados de tokenización
- Búsquedas de incrustación
- Procesamiento de secuencias

### Number
Valores numéricos (Int o Float).

**Variantes:**
- `Integer`: Entero con signo de 64 bits
- `Float`: Punto flotante de 64 bits

---

## Referencias

- **Paper de TensorLogic**: arXiv:2510.12269
- **Arquitectura Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Documentación Relacionada

- [Guía de Inicio](./getting_started.md)
- [Referencia del Lenguaje](./language_reference.md)
- [Ejemplos](../../examples/)
- [Operaciones Añadidas en 2025](../added_operations_2025.md)
- [Lista de TODOs](../TODO.md)

---

**Última Actualización:** 2025-01-22

**Versión de TensorLogic:** 0.1.1+

**Operaciones Totales:** 48 funciones + 4 operadores
