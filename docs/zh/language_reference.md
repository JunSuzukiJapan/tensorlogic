# TensorLogic 语言参考

**Version**: 0.1.1
**Last Updated**: 2025-10-21

## 主要特性

- **GPU**: TensorLogic 是一种将张量代数与逻辑编程统一的编程语言，实现神经符号AI。
- **优化器**: SGD, Adam, AdamW
- **learn 块中的局部变量**: `:=`
- **Import**: `import "file.tl"`

---

## 运算符

`+` `-` `*` `/` `@` `**` `==` `!=` `<` `>` `<=` `>=`

---

## 内置函数

```tensorlogic
relu(x), gelu(x), softmax(x)
sum(x), mean(x), max(x), min(x)
```

---

## 学习系统

### 优化器

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### learn 块中的局部变量

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

**注意：只有 `learnable` 张量被优化，局部变量不会被优化。**

---

GitHub: https://github.com/JunSuzukiJapan/tensorlogic
