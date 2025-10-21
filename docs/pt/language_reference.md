# Referência da Linguagem TensorLogic

**Version**: 0.1.1
**Last Updated**: 2025-10-21

## Características principais

- **GPU**: TensorLogic é uma linguagem de programação que unifica álgebra tensorial com programação lógica, permitindo IA neuro-simbólica.
- **Otimizadores**: SGD, Adam, AdamW
- **Variáveis locais em learn**: `:=`
- **Import**: `import "file.tl"`

---

## Operadores

`+` `-` `*` `/` `@` `**` `==` `!=` `<` `>` `<=` `>=`

---

## Funções integradas

```tensorlogic
relu(x), gelu(x), softmax(x)
sum(x), mean(x), max(x), min(x)
```

---

## Sistema de aprendizagem

### Otimizadores

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### Variáveis locais em learn

```tensorlogic
tensor W: float16[1] learnable = [0.5]

main {
    learn {
        pred1 := x1 * W
        pred2 := x2 * W
        loss := (pred1 - y1) ** 2 + (pred2 - y2) ** 2
        
        objective: loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }
}
```

**Nota: Apenas tensores `learnable` são otimizados, não variáveis locais.**

---

GitHub: https://github.com/JunSuzukiJapan/tensorlogic
