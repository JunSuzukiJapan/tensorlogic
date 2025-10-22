# Справочник API TensorLogic

Полный справочник API для всех доступных операций в TensorLogic.

## Содержание

1. [Создание тензоров](#создание-тензоров)
2. [Операции с формой](#операции-с-формой)
3. [Математические функции](#математические-функции)
4. [Операции агрегации](#операции-агрегации)
5. [Функции активации](#функции-активации)
6. [Матричные операции](#матричные-операции)
7. [Нормализация](#нормализация)
8. [Операции маскирования](#операции-маскирования)
9. [Операции индексирования](#операции-индексирования)
10. [Встраивания](#встраивания)
11. [Выборка](#выборка)
12. [Объединенные операции](#объединенные-операции)
13. [Оптимизация](#оптимизация)
14. [Другие операции](#другие-операции)
15. [Операторы](#операторы)
16. [Определения типов](#определения-типов)

---

## Создание тензоров

### `zeros(shape: Array<Int>) -> Tensor`

Создает тензор, заполненный нулями.

**Параметры:**
- `shape`: Массив, указывающий размерности тензора

**Возвращает:** Тензор, заполненный 0

**Пример:**
```tensorlogic
let z = zeros([2, 3])  // Тензор 2x3 из нулей
```

---

### `ones(shape: Array<Int>) -> Tensor`

Создает тензор, заполненный единицами.

**Параметры:**
- `shape`: Массив, указывающий размерности тензора

**Возвращает:** Тензор, заполненный 1

**Пример:**
```tensorlogic
let o = ones([2, 3])  // Тензор 2x3 из единиц
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Генерирует синусоидальное позиционное кодирование для Transformers.

**Параметры:**
- `seq_len`: Длина последовательности
- `d_model`: Размерность модели

**Возвращает:** Тензор формы `[seq_len, d_model]`

**Математическое определение:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Пример:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Случаи использования:**
- Модели Transformer
- Модели последовательность-к-последовательности
- Механизмы внимания

**Ссылки:**
- arXiv:2510.12269 (Таблица 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Операции с формой

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Изменяет форму тензора с сохранением данных.

**Параметры:**
- `tensor`: Входной тензор
- `new_shape`: Целевая форма

**Возвращает:** Переформированный тензор

**Пример:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Ограничения:**
- Общее количество элементов должно оставаться неизменным

---

### `flatten(tensor: Tensor) -> Tensor`

Выравнивает тензор до 1D.

**Параметры:**
- `tensor`: Входной тензор

**Возвращает:** 1D тензор

**Пример:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Транспонирует 2D тензор (меняет оси местами).

**Параметры:**
- `tensor`: Входной 2D тензор

**Возвращает:** Транспонированный тензор

**Пример:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Переставляет размерности тензора.

**Параметры:**
- `tensor`: Входной тензор
- `dims`: Новый порядок размерностей

**Возвращает:** Переставленный тензор

**Пример:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Добавляет размерность размера 1 в указанную позицию.

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Позиция для вставки новой размерности

**Возвращает:** Тензор с добавленной размерностью

**Пример:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Удаляет все размерности размера 1.

**Параметры:**
- `tensor`: Входной тензор

**Возвращает:** Тензор с удаленными размерностями размера 1

**Пример:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Разделяет тензор на несколько тензоров вдоль указанной размерности.

**Параметры:**
- `tensor`: Входной тензор
- `sizes`: Размер каждой разделенной секции
- `dim`: Размерность для разделения

**Возвращает:** Массив тензоров

**Пример:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 тензора: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Разделяет тензор на указанное количество частей.

**Параметры:**
- `tensor`: Входной тензор
- `chunks`: Количество частей
- `dim`: Размерность для разделения

**Возвращает:** Массив тензоров

**Пример:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 тензора по [4,4] каждый
```

---

## Математические функции

### `exp(tensor: Tensor) -> Tensor`

Применяет экспоненциальную функцию поэлементно.

**Математическое определение:** `exp(x) = e^x`

**Пример:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Применяет натуральный логарифм поэлементно.

**Математическое определение:** `log(x) = ln(x)`

**Пример:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Применяет квадратный корень поэлементно.

**Математическое определение:** `sqrt(x) = √x`

**Пример:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Возводит элементы тензора в указанную степень.

**Математическое определение:** `pow(x, n) = x^n`

**Пример:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Применяет функцию синуса поэлементно.

**Пример:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Применяет функцию косинуса поэлементно.

**Пример:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Применяет функцию тангенса поэлементно.

**Пример:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Операции агрегации

### `sum(tensor: Tensor) -> Number`

Вычисляет сумму всех элементов.

**Пример:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Вычисляет среднее значение всех элементов.

**Пример:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Возвращает максимальное значение в тензоре.

**Пример:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Возвращает минимальное значение в тензоре.

**Пример:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Возвращает индексы максимальных значений вдоль указанной размерности.

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Размерность для поиска максимума

**Возвращает:** Тензор индексов

**Пример:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Индексы максимумов вдоль размерности 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Возвращает индексы минимальных значений вдоль указанной размерности.

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Размерность для поиска минимума

**Возвращает:** Тензор индексов

**Пример:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Индексы минимумов вдоль размерности 1
```

---

## Функции активации

### `relu(tensor: Tensor) -> Tensor`

Активация Rectified Linear Unit.

**Математическое определение:** `relu(x) = max(0, x)`

**Пример:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Функция активации sigmoid.

**Математическое определение:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Пример:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Активация Gaussian Error Linear Unit (используется в BERT, GPT).

**Математическое определение:** 
```
gelu(x) = x * Φ(x)
где Φ(x) - кумулятивная функция распределения стандартного нормального распределения
```

**Пример:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Случаи использования:**
- Модели BERT, GPT
- Современные архитектуры Transformer

---

### `tanh(tensor: Tensor) -> Tensor`

Активация гиперболического тангенса.

**Математическое определение:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Пример:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Применяет нормализацию softmax вдоль указанной размерности.

**Математическое определение:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Размерность для применения softmax

**Возвращает:** Тензор распределения вероятностей

**Пример:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Случаи использования:**
- Механизмы внимания
- Выходные слои классификации
- Распределения вероятностей

---

## Матричные операции

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Матричное умножение.

**Параметры:**
- `a`: Левая матрица
- `b`: Правая матрица

**Возвращает:** Результат матричного умножения

**Пример:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Нормализация

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Применяет нормализацию слоя.

**Математическое определение:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Параметры:**
- `tensor`: Входной тензор
- `normalized_shape`: Форма для нормализации
- `eps`: Малое значение для численной стабильности (по умолчанию: 1e-5)

**Пример:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Случаи использования:**
- Слои Transformer
- Рекуррентные сети
- Глубокие нейронные сети

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Применяет пакетную нормализацию.

**Параметры:**
- `tensor`: Входной тензор
- `running_mean`: Скользящее среднее
- `running_var`: Скользящая дисперсия
- `eps`: Малое значение для численной стабильности

**Пример:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Применяет dropout (случайно обнуляет элементы с вероятностью p).

**Параметры:**
- `tensor`: Входной тензор
- `p`: Вероятность dropout (от 0.0 до 1.0)

**Возвращает:** Тензор со случайно обнуленными элементами

**Пример:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Случаи использования:**
- Регуляризация обучения
- Предотвращение переобучения
- Ансамблевое обучение

---

## Операции маскирования

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Применяет маску внимания (устанавливает маскированные позиции в -inf).

**Параметры:**
- `tensor`: Тензор оценок внимания
- `mask`: Бинарная маска (1 = сохранить, 0 = замаскировать)

**Возвращает:** Маскированный тензор

**Пример:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Случаи использования:**
- Механизмы внимания Transformer
- Маскирование последовательностей
- Причинное (авторегрессивное) внимание

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Создает маску заполнения для последовательностей переменной длины.

**Параметры:**
- `lengths`: Массив фактических длин последовательностей
- `max_len`: Максимальная длина последовательности

**Возвращает:** Тензор бинарной маски

**Пример:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Результат: [batch_size, max_len] где 1 = реальный токен, 0 = заполнение
```

**Случаи использования:**
- Обработка последовательностей переменной длины
- Пакетная обработка
- Маскирование внимания

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Объединяет две маски с помощью логического И.

**Параметры:**
- `mask1`: Первая маска
- `mask2`: Вторая маска

**Возвращает:** Объединенная маска

**Пример:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Случаи использования:**
- Объединение масок заполнения и внимания
- Многоограничительное маскирование
- Сложные шаблоны внимания

---

## Операции индексирования

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Собирает значения вдоль размерности с помощью индексов.

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Размерность для сбора
- `indices`: Тензор индексов

**Возвращает:** Собранный тензор

**Пример:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Случаи использования:**
- Выбор токенов
- Лучевой поиск
- Расширенное индексирование

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Выбирает элементы по указанным индексам вдоль размерности.

**Параметры:**
- `tensor`: Входной тензор
- `dim`: Размерность для выбора
- `indices`: Массив индексов

**Возвращает:** Выбранный тензор

**Пример:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Выбрать строки 0, 2, 5
```

---

## Встраивания

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Преобразует ID токенов во встраивания.

**Параметры:**
- `indices`: Последовательность ID токенов
- `vocab_size`: Размер словаря
- `embed_dim`: Размерность встраивания

**Возвращает:** Тензор встраиваний формы `[seq_len, embed_dim]`

**Пример:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Случаи использования:**
- Встраивания слов
- Представления токенов
- Языковые модели

**Ссылки:**
- arXiv:2510.12269 (Таблица 1)

---

## Выборка

### `top_k(logits: Tensor, k: Int) -> Tensor`

Выбирает токен с использованием выборки top-k.

**Параметры:**
- `logits`: Выходные логиты модели
- `k`: Количество топовых токенов для рассмотрения

**Возвращает:** ID выбранного токена

**Пример:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Случаи использования:**
- Генерация текста
- Контролируемая выборка
- Разнообразные выходы

**Ссылки:**
- arXiv:2510.12269 (Таблица 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Выбирает токен с использованием nucleus (top-p) выборки.

**Параметры:**
- `logits`: Выходные логиты модели
- `p`: Порог кумулятивной вероятности (от 0.0 до 1.0)

**Возвращает:** ID выбранного токена

**Пример:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Случаи использования:**
- Генерация текста
- Динамический выбор словаря
- Выборка с контролем качества

**Ссылки:**
- arXiv:2510.12269 (Таблица 2)

---

## Объединенные операции

Объединенные операции комбинируют несколько операций для лучшей производительности за счет уменьшения накладных расходов памяти и запусков ядер.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Объединяет сложение и активацию ReLU.

**Математическое определение:** `fused_add_relu(x, y) = relu(x + y)`

**Пример:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Производительность:** ~в 1.5 раза быстрее отдельных операций

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Объединяет умножение и активацию ReLU.

**Математическое определение:** `fused_mul_relu(x, y) = relu(x * y)`

**Пример:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Объединяет аффинное преобразование (масштаб и смещение).

**Математическое определение:** `fused_affine(x, s, b) = x * s + b`

**Пример:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Случаи использования:**
- Пакетная нормализация
- Нормализация слоя
- Пользовательские линейные преобразования

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Объединяет активацию GELU и линейное преобразование.

**Математическое определение:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Пример:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Случаи использования:**
- Слои FFN в Transformer
- Архитектуры BERT/GPT
- Критичные для производительности пути

---

## Оптимизация

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Выполняет шаг оптимизатора SGD.

**Математическое определение:** `params_new = params - lr * gradients`

**Параметры:**
- `params`: Текущие параметры
- `gradients`: Вычисленные градиенты
- `lr`: Скорость обучения

**Возвращает:** Обновленные параметры

**Пример:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Выполняет шаг оптимизатора Adam.

**Параметры:**
- `params`: Текущие параметры
- `gradients`: Вычисленные градиенты
- `m`: Оценка первого момента
- `v`: Оценка второго момента
- `lr`: Скорость обучения
- `beta1`: Скорость затухания первого момента (по умолчанию: 0.9)
- `beta2`: Скорость затухания второго момента (по умолчанию: 0.999)
- `eps`: Константа численной стабильности (по умолчанию: 1e-8)

**Возвращает:** Обновленные параметры

**Пример:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Другие операции

### `tokenize(text: String) -> TokenIDs`

Преобразует текст в последовательность ID токенов.

**Параметры:**
- `text`: Входная текстовая строка

**Возвращает:** TokenIDs (Vec<u32>)

**Пример:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Случаи использования:**
- Предобработка текста
- Вход языковой модели
- NLP пайплайны

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Транслирует тензор в указанную форму.

**Параметры:**
- `tensor`: Входной тензор
- `shape`: Целевая форма

**Возвращает:** Транслированный тензор

**Пример:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Случаи использования:**
- Выравнивание форм
- Пакетные операции
- Поэлементные операции с разными формами

---

## Операторы

TensorLogic поддерживает стандартные математические операторы:

### Арифметические операторы
- `+` : Сложение
- `-` : Вычитание
- `*` : Поэлементное умножение
- `/` : Поэлементное деление

**Пример:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Операторы сравнения
- `==` : Равно
- `!=` : Не равно
- `<`  : Меньше
- `<=` : Меньше или равно
- `>`  : Больше
- `>=` : Больше или равно

### Логические операторы
- `&&` : Логическое И
- `||` : Логическое ИЛИ
- `!`  : Логическое НЕ

---

## Определения типов

### Tensor
Многомерный массив с ускорением GPU через Metal Performance Shaders.

**Свойства:**
- Форма: Массив размерностей
- Данные: Элементы Float32
- Устройство: GPU устройство Metal

### TokenIDs
Специальный тип для последовательностей ID токенов.

**Определение:** `Vec<u32>`

**Случаи использования:**
- Результаты токенизации
- Поиск встраиваний
- Обработка последовательностей

### Number
Числовые значения (Int или Float).

**Варианты:**
- `Integer`: 64-битное целое со знаком
- `Float`: 64-битное число с плавающей точкой

---

## Ссылки

- **Статья TensorLogic**: arXiv:2510.12269
- **Архитектура Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Связанная документация

- [Руководство по началу работы](./getting_started.md)
- [Справочник по языку](./language_reference.md)
- [Примеры](../../examples/)
- [Операции, добавленные в 2025](../added_operations_2025.md)
- [Список TODO](../TODO.md)

---

**Последнее обновление:** 2025-01-22

**Версия TensorLogic:** 0.1.1+

**Всего операций:** 48 функций + 4 оператора
