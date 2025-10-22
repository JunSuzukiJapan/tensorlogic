# Руководство по Интеграции CoreML & Neural Engine

Это руководство объясняет, как использовать модели CoreML в TensorLogic и выполнять высокоскоростной вывод на Apple Neural Engine.

## О CoreML и Neural Engine

### CoreML

- Проприетарный фреймворк машинного обучения Apple
- Оптимизирован исключительно для iOS/macOS
- Автоматически использует Neural Engine, GPU и CPU
- Формат .mlmodel / .mlmodelc

### Neural Engine

- Специализированный AI-чип эксклюзивно для Apple Silicon
- До 15,8 TOPS (M1 Pro/Max)
- Сверхнизкое энергопотребление (1/10 или меньше по сравнению с GPU)
- Оптимизирован для операций f16

### Интеграция с TensorLogic

- Все операции f16 (оптимизировано для Neural Engine)
- Бесшовная интеграция с Metal GPU
- Автоматическое определение формата модели

## Создание Моделей CoreML

Модели CoreML обычно создаются с помощью coreMLtools из Python:

```python
import coremltools as ct
import torch

# Создать модель PyTorch
model = MyModel()
model.eval()

# Создать трассированную модель
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Преобразовать в CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Оптимизация Neural Engine
    compute_precision=ct.precision.FLOAT16  # точность f16
)

# Сохранить
mlmodel.save("model.mlpackage")
```

## Использование в TensorLogic

### 1. Загрузить Модель CoreML (только macOS)

```tensorlogic
model = load_model("model.mlpackage")
// или
model = load_model("model.mlmodelc")
```

### 2. Проверить Метаданные

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Лучшие Практики Оптимизации Neural Engine

### 1. Тип Данных: Использовать f16

✅ Рекомендуется: `compute_precision=ct.precision.FLOAT16`
❌ Не рекомендуется: FLOAT32 (выполняется на GPU)

### 2. Формат Модели: Использовать формат mlprogram

✅ Рекомендуется: `convert_to="mlprogram"`
❌ Не рекомендуется: `convert_to="neuralnetwork"` (устаревший формат)

### 3. Размер Пакета: 1 оптимален

✅ Рекомендуется: `batch_size=1`
⚠️ Примечание: `batch_size>1` может выполняться на GPU

### 4. Размер Входа: Фиксированный размер оптимален

✅ Рекомендуется: `shape=[1, 3, 224, 224]`
⚠️ Примечание: Переменные размеры имеют ограниченную оптимизацию

## Поддерживаемые Операции

### Операции, быстро выполняемые на Neural Engine

- ✅ Свертки (conv2d, depthwise_conv)
- ✅ Полносвязные слои (linear, matmul)
- ✅ Пулинг (max_pool, avg_pool)
- ✅ Нормализация (batch_norm, layer_norm)
- ✅ Функции активации (relu, gelu, sigmoid, tanh)
- ✅ Поэлементные операции (add, mul, sub, div)

## Сравнение Производительности

Вывод ResNet-50 (изображение 224x224):

| Устройство         | Задержка | Мощность | Эффективность |
|-------------------|----------|----------|---------------|
| Neural Engine     | ~3ms     | ~0.5W    | Максимальная  |
| Metal GPU (M1)    | ~8ms     | ~5W      | Средняя       |
| CPU (M1)          | ~50ms    | ~2W      | Низкая        |

## Выбор Формата Модели

### Рекомендуемые Форматы по Вариантам Использования

**Обучение**: SafeTensors
- Совместимость с PyTorch
- Сохранение/загрузка весов
- Обучение на Metal GPU

**Вывод (iOS/macOS)**: CoreML
- Оптимизация Neural Engine
- Сверхнизкое энергопотребление
- Интеграция в приложения

**Вывод (Общий)**: GGUF
- Поддержка квантования
- Кросс-платформенность
- Эффективность памяти

## Ссылки

- [Официальная Документация CoreML](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Руководство Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Руководство по Загрузке Моделей](model_loading.md)
