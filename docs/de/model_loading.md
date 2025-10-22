# Modell-Ladeführer

Dieses Dokument erklärt, wie man PyTorch- und HuggingFace-Modelle in TensorLogic lädt und verwendet. Es unterstützt das SafeTensors-Format (PyTorch-kompatibel) und das GGUF-Format (quantisierte LLMs).

## Grundlegende Verwendung

### 1. SafeTensors-Modell laden (aus PyTorch gespeichert)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. GGUF-Modell laden (quantisiertes LLM)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Tensoren aus dem Modell abrufen

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Praktisches Beispiel: Lineare Schicht-Inferenz

Inferenz mit Modellgewichten und Biases durchführen:

```tensorlogic
function forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Lineare Transformation: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## PyTorch-Modelle vorbereiten

Speichern Sie Ihr Modell im SafeTensors-Format mit Python:

```python
import torch
from safetensors.torch import save_file

# PyTorch-Modell erstellen
model = MyModel()

# Modellgewichte als Dictionary abrufen
tensors = {name: param for name, param in model.named_parameters()}

# Im SafeTensors-Format speichern
save_file(tensors, "model.safetensors")
```

Dann in TensorLogic laden:

```tensorlogic
model = load_model("model.safetensors")
```

## Unterstützte Formate

### 1. SafeTensors (.safetensors)

- PyTorch- und HuggingFace-kompatibel
- Unterstützt F32, F64, F16, BF16 Datentypen
- Alle Daten werden automatisch in f16 konvertiert
- Direkt in Metal GPU geladen

### 2. GGUF (.gguf)

- llama.cpp-Format quantisierte Modelle
- Unterstützt Q4_0, Q8_0, F32, F16
- Direkt in Metal GPU geladen

### 3. CoreML (.mlmodel, .mlpackage)

- Apple Neural Engine optimierte Modelle
- Nur iOS/macOS

## Vollständiges lineares Modellbeispiel

```tensorlogic
// Eingabedaten (Batch-Größe 4, Merkmalsdimension 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Gewichtsmatrix (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Bias (2 Dimensionen)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Inferenz ausführen
let output = forward(X, W, b)

// Ergebnisse ausgeben
print("Output shape:", output.shape)
print("Output:", output)
```

## Modelle speichern

Sie können TensorLogic-Modelle im SafeTensors-Format speichern:

```tensorlogic
save_model(model, "output.safetensors")
```

Dies ermöglicht Interoperabilität mit PyTorch und HuggingFace.

## Wichtige Hinweise

- TensorLogic führt alle Operationen in f16 aus (Metal GPU optimiert)
- Andere Datentypen werden beim Laden automatisch in f16 konvertiert
- Integer-Typen (i8, i32, etc.) werden nicht unterstützt (nur Gleitkommazahlen)
- Große Modelle werden automatisch in den Metal GPU-Speicher geladen

## Verwandte Dokumentation

- [GGUF Quantisierte Modelle](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Erste Schritte Anleitung](../claudedocs/getting_started.md)
