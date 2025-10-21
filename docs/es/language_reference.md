# Referencia del Lenguaje TensorLogic

**Versión**: 0.1.1
**Última actualización**: 2025-10-21

## Tabla de contenidos

1. [Introducción](#introducción)
2. [Estructura del programa](#estructura-del-programa)
3. [Tipos de datos](#tipos-de-datos)
7. [Operadores](#operadores)
8. [Funciones integradas](#funciones-integradas)
9. [Sistema de aprendizaje](#sistema-de-aprendizaje)

---

## 1. Introducción

TensorLogic es un lenguaje de programación que unifica el álgebra tensorial con la programación lógica, permitiendo IA neuro-simbólica.

### Características principales

- **Operaciones tensoriales**: Cálculos de alto rendimiento acelerados por GPU
- **Diferenciación automática**: Cálculo de gradientes incorporado
- **Sistema de aprendizaje**: Descenso de gradiente con múltiples optimizadores (SGD, Adam, AdamW)
- **Variables locales**: Soporte de variables locales en bloques `learn` con `:=`
- **Números negativos**: Soporte completo de literales numéricos negativos
- **Importación de archivos**: Importar declaraciones desde archivos externos

---

## 2. Estructura del programa

### 2.1 Importar archivos externos

```tensorlogic
import "path/to/module.tl"

main {
    result := imported_tensor * 2
}
```

### 2.2 Estructura básica

```tensorlogic
tensor w: float16[10] learnable = [...]

main {
    learn {
        // Variables locales para cálculos intermedios
        pred := x * w
        loss := pred * pred
        
        objective: loss,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

---

## 3. Tipos de datos

| Tipo | Descripción |
|------|-------------|
| `float16` | 16 bits (optimizado Apple Silicon) |
| `float32` | 32 bits precisión simple |
| `float64` | 64 bits precisión doble |
| `int32` | Entero 32 bits |
| `bool` | Booleano |

**Literales numéricos**: `[3.14]`, `[-2.71]`, `[-42.0]`

---

## 7. Operadores

| Operador | Descripción |
|-----------|-------------|
| `+` `-` `*` `/` | Aritmética |
| `@` | Multiplicación matricial |
| `**` | Potencia |
| `==` `!=` `<` `>` `<=` `>=` | Comparación |

---

## 8. Funciones integradas

```tensorlogic
relu(x), gelu(x), softmax(x)      // Activaciones
sum(x), mean(x), max(x), min(x)   // Reducciones
```

---

## 9. Sistema de aprendizaje

### 9.1 Optimizadores

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### 9.2 Variables locales en learn

```tensorlogic
tensor W: float16[1] learnable = [0.5]

main {
    learn {
        // Variables locales
        pred1 := x1 * W
        pred2 := x2 * W
        loss := (pred1 - y1) * (pred1 - y1) + (pred2 - y2) * (pred2 - y2)
        
        objective: loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }
}
```

**Nota**: Solo los tensores `learnable` son optimizados, no las variables locales.

---

Para más información: https://github.com/JunSuzukiJapan/tensorlogic
