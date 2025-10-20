# Referencia del Lenguaje TensorLogic

**Versión**: 0.2.0-alpha
**Última actualización**: 2025-10-20

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Estructura del Programa](#estructura-del-programa)
3. [Tipos de Datos](#tipos-de-datos)
4. [Declaraciones](#declaraciones)
5. [Expresiones](#expresiones)
6. [Sentencias](#sentencias)
7. [Operadores](#operadores)
8. [Funciones Integradas](#funciones-integradas)
9. [Sistema de Aprendizaje](#sistema-de-aprendizaje)
10. [Programación Lógica](#programación-lógica)

---

## 1. Introducción

TensorLogic es un lenguaje de programación que unifica el álgebra tensorial con la programación lógica, habilitando la IA neuro-simbólica.

### Características Principales

- **Operaciones Tensoriales**: Cálculos de alto rendimiento acelerados por GPU
- **Diferenciación Automática**: Cálculo de gradiente integrado
- **Sistema de Aprendizaje**: Descenso de gradiente con múltiples optimizadores
- **Programación Lógica**: Relaciones, reglas y consultas
- **Integración Neuro-Simbólica**: Embeddings para entidades y relaciones

---

## 2. Estructura del Programa

```tensorlogic
// Declaraciones
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Bloque de ejecución principal
main {
    result := w * w

    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

---

## 3. Tipos de Datos

| Tipo | Descripción | Precisión |
|------|-------------|-----------|
| `float32` | Punto flotante 32-bit | Precisión simple |
| `float64` | Punto flotante 64-bit | Precisión doble |
| `int32` | Entero 32-bit | Entero con signo |
| `int64` | Entero 64-bit | Entero largo con signo |
| `bool` | Booleano | verdadero/falso |

---

## 7. Operadores

| Operador | Nombre | Ejemplo |
|----------|------|---------|
| `+` | Adición | `a + b` |
| `-` | Sustracción | `a - b` |
| `*` | Multiplicación | `a * b` |
| `/` | División | `a / b` |
| `@` | Multiplicación Matricial | `A @ B` |
| `**` | Potencia | `a ** 2` |

---

**Fin de la Referencia del Lenguaje**

Para preguntas o contribuciones, visita: https://github.com/JunSuzukiJapan/tensorlogic
