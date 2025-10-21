# Riferimento del Linguaggio TensorLogic

**Version**: 0.1.1
**Last Updated**: 2025-10-21

## Caratteristiche principali

- **GPU**: TensorLogic è un linguaggio di programmazione che unifica l'algebra tensoriale con la programmazione logica, abilitando l'IA neuro-simbolica.
- **Ottimizzatori**: SGD, Adam, AdamW
- **Variabili locali in learn**: `:=`
- **Import**: `import "file.tl"`

---

## Operatori

`+` `-` `*` `/` `@` `**` `==` `!=` `<` `>` `<=` `>=`

---

## Funzioni integrate

```tensorlogic
relu(x), gelu(x), softmax(x)
sum(x), mean(x), max(x), min(x)
```

---

## Sistema di apprendimento

### Ottimizzatori

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### Variabili locali in learn

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

**Nota: Solo i tensori `learnable` sono ottimizzati, non le variabili locali.**

---

GitHub: https://github.com/JunSuzukiJapan/tensorlogic
