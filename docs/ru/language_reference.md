# Справочник по языку TensorLogic

**Version**: 0.1.1
**Last Updated**: 2025-10-21

## Основные возможности

- **GPU**: TensorLogic - это язык программирования, объединяющий тензорную алгебру с логическим программированием, обеспечивая нейро-символический ИИ.
- **Оптимизаторы**: SGD, Adam, AdamW
- **Локальные переменные в learn**: `:=`
- **Import**: `import "file.tl"`

---

## Операторы

`+` `-` `*` `/` `@` `**` `==` `!=` `<` `>` `<=` `>=`

---

## Встроенные функции

```tensorlogic
relu(x), gelu(x), softmax(x)
sum(x), mean(x), max(x), min(x)
```

---

## Система обучения

### Оптимизаторы

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### Локальные переменные в learn

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

**Примечание: Оптимизируются только тензоры `learnable`, а не локальные переменные.**

---

GitHub: https://github.com/JunSuzukiJapan/tensorlogic
