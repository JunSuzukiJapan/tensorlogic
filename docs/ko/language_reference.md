# TensorLogic 언어 레퍼런스

**Version**: 0.1.1
**Last Updated**: 2025-10-21

## 주요 기능

- **GPU**: TensorLogic은 텐서 대수와 논리 프로그래밍을 통합하여 신경-기호 AI를 가능하게 하는 프로그래밍 언어입니다.
- **최적화기**: SGD, Adam, AdamW
- **learn 블록의 로컬 변수**: `:=`
- **Import**: `import "file.tl"`

---

## 연산자

`+` `-` `*` `/` `@` `**` `==` `!=` `<` `>` `<=` `>=`

---

## 내장 함수

```tensorlogic
relu(x), gelu(x), softmax(x)
sum(x), mean(x), max(x), min(x)
```

---

## 학습 시스템

### 최적화기

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### learn 블록의 로컬 변수

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

**참고: `learnable` 텐서만 최적화되며 로컬 변수는 최적화되지 않습니다.**

---

GitHub: https://github.com/JunSuzukiJapan/tensorlogic
