# Guida al Caricamento dei Modelli

Questo documento spiega come caricare e utilizzare modelli PyTorch e HuggingFace in TensorLogic. Supporta il formato SafeTensors (compatibile con PyTorch) e il formato GGUF (LLM quantizzati).

## Utilizzo Base

### 1. Caricare Modello SafeTensors (salvato da PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Caricare Modello GGUF (LLM quantizzato)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Ottenere Tensori dal Modello

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Esempio Pratico: Inferenza Layer Lineare

Eseguire inferenza usando pesi e bias del modello:

```tensorlogic
function forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Trasformazione lineare: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Preparare Modelli PyTorch

Salvare il modello in formato SafeTensors usando Python:

```python
import torch
from safetensors.torch import save_file

# Creare modello PyTorch
model = MyModel()

# Ottenere pesi del modello come dizionario
tensors = {name: param for name, param in model.named_parameters()}

# Salvare in formato SafeTensors
save_file(tensors, "model.safetensors")
```

Quindi caricare in TensorLogic:

```tensorlogic
model = load_model("model.safetensors")
```

## Formati Supportati

### 1. SafeTensors (.safetensors)

- Compatibile con PyTorch e HuggingFace
- Supporta tipi di dati F32, F64, F16, BF16
- Tutti i dati vengono automaticamente convertiti in f16
- Caricato direttamente su GPU Metal

### 2. GGUF (.gguf)

- Modelli quantizzati in formato llama.cpp
- Supporta Q4_0, Q8_0, F32, F16
- Caricato direttamente su GPU Metal

### 3. CoreML (.mlmodel, .mlpackage)

- Modelli ottimizzati per Apple Neural Engine
- Solo iOS/macOS

## Esempio Completo di Modello Lineare

```tensorlogic
// Dati di input (dimensione batch 4, dimensione caratteristiche 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Matrice dei pesi (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Bias (2 dimensioni)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Eseguire inferenza
let output = forward(X, W, b)

// Stampare risultati
print("Output shape:", output.shape)
print("Output:", output)
```

## Salvare Modelli

È possibile salvare modelli TensorLogic in formato SafeTensors:

```tensorlogic
save_model(model, "output.safetensors")
```

Questo consente l'interoperabilità con PyTorch e HuggingFace.

## Note Importanti

- TensorLogic esegue tutte le operazioni in f16 (ottimizzato per GPU Metal)
- Altri tipi di dati vengono automaticamente convertiti in f16 durante il caricamento
- I tipi interi (i8, i32, ecc.) non sono supportati (solo virgola mobile)
- I modelli di grandi dimensioni vengono automaticamente caricati nella memoria GPU Metal

## Documentazione Correlata

- [Modelli Quantizzati GGUF](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Guida Introduttiva](../claudedocs/getting_started.md)
