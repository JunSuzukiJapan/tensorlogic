# Руководство по Загрузке Моделей

Этот документ объясняет, как загружать и использовать модели PyTorch и HuggingFace в TensorLogic. Поддерживает формат SafeTensors (совместимый с PyTorch) и формат GGUF (квантованные LLM).

## Основное Использование

### 1. Загрузить Модель SafeTensors (сохраненную из PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Загрузить Модель GGUF (квантованная LLM)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Получить Тензоры из Модели

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Практический Пример: Вывод Линейного Слоя

Выполнение вывода с использованием весов и смещений модели:

```tensorlogic
fn forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Линейное преобразование: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Подготовка Моделей PyTorch

Сохраните вашу модель в формате SafeTensors используя Python:

```python
import torch
from safetensors.torch import save_file

# Создать модель PyTorch
model = MyModel()

# Получить веса модели как словарь
tensors = {name: param for name, param in model.named_parameters()}

# Сохранить в формате SafeTensors
save_file(tensors, "model.safetensors")
```

Затем загрузить в TensorLogic:

```tensorlogic
model = load_model("model.safetensors")
```

## Поддерживаемые Форматы

### 1. SafeTensors (.safetensors)

- Совместимость с PyTorch и HuggingFace
- Поддержка типов данных F32, F64, F16, BF16
- Все данные автоматически конвертируются в f16
- Загружается напрямую в GPU Metal

### 2. GGUF (.gguf)

- Квантованные модели в формате llama.cpp
- Поддержка Q4_0, Q8_0, F32, F16
- Загружается напрямую в GPU Metal

### 3. CoreML (.mlmodel, .mlpackage)

- Модели, оптимизированные для Apple Neural Engine
- Только iOS/macOS

## Полный Пример Линейной Модели

```tensorlogic
// Входные данные (размер пакета 4, размерность признаков 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Матрица весов (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Смещение (2 размерности)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Выполнить вывод
let output = forward(X, W, b)

// Вывести результаты
print("Output shape:", output.shape)
print("Output:", output)
```

## Сохранение Моделей

Вы можете сохранять модели TensorLogic в формате SafeTensors:

```tensorlogic
save_model(model, "output.safetensors")
```

Это обеспечивает совместимость с PyTorch и HuggingFace.

## Важные Замечания

- TensorLogic выполняет все операции в f16 (оптимизировано для GPU Metal)
- Другие типы данных автоматически конвертируются в f16 при загрузке
- Целочисленные типы (i8, i32 и т.д.) не поддерживаются (только с плавающей точкой)
- Большие модели автоматически загружаются в память GPU Metal

## Связанная Документация

- [Квантованные Модели GGUF](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Руководство по Началу Работы](../claudedocs/getting_started.md)
