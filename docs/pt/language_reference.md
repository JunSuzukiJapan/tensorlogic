# Referência da Linguagem TensorLogic

**Versão**: 0.2.0-alpha
**Última atualização**: 2025-10-20

## Índice

1. [Introdução](#introdução)
2. [Estrutura do Programa](#estrutura-do-programa)
3. [Tipos de Dados](#tipos-de-dados)
4. [Declarações](#declarações)
5. [Expressões](#expressões)
6. [Instruções](#instruções)
7. [Operadores](#operadores)
8. [Funções Integradas](#funções-integradas)
9. [Sistema de Aprendizagem](#sistema-de-aprendizagem)
10. [Programação Lógica](#programação-lógica)

---

## 1. Introdução

TensorLogic é uma linguagem de programação que unifica álgebra tensorial com programação lógica, habilitando IA neuro-simbólica.

### Características Principais

- **Operações Tensoriais**: Cálculos de alto desempenho acelerados por GPU
- **Diferenciação Automática**: Cálculo de gradiente integrado
- **Sistema de Aprendizagem**: Descida de gradiente com múltiplos otimizadores
- **Programação Lógica**: Relações, regras e consultas
- **Integração Neuro-Simbólica**: Embeddings para entidades e relações

---

## 2. Estrutura do Programa

```tensorlogic
// Declarações
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Bloco de execução principal
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

## 3. Tipos de Dados

| Tipo | Descrição | Precisão |
|------|-------------|-----------|
| `float32` | Ponto flutuante 32-bit | Precisão simples |
| `float64` | Ponto flutuante 64-bit | Precisão dupla |
| `int32` | Inteiro 32-bit | Inteiro com sinal |
| `int64` | Inteiro 64-bit | Inteiro longo com sinal |
| `bool` | Booleano | verdadeiro/falso |

---

## 7. Operadores

| Operador | Nome | Exemplo |
|----------|------|---------|
| `+` | Adição | `a + b` |
| `-` | Subtração | `a - b` |
| `*` | Multiplicação | `a * b` |
| `/` | Divisão | `a / b` |
| `@` | Multiplicação Matricial | `A @ B` |
| `**` | Potência | `a ** 2` |

---

**Fim da Referência da Linguagem**

Para perguntas ou contribuições, visite: https://github.com/JunSuzukiJapan/tensorlogic
