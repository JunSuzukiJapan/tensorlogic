# CoreML & Neural Engine Integrationsleitfaden

Dieser Leitfaden erklärt, wie man CoreML-Modelle in TensorLogic verwendet und Hochgeschwindigkeitsinferenz auf der Apple Neural Engine durchführt.

## Über CoreML und Neural Engine

### CoreML

- Apples proprietäres Machine Learning Framework
- Ausschließlich für iOS/macOS optimiert
- Nutzt automatisch Neural Engine, GPU und CPU
- .mlmodel / .mlmodelc Format

### Neural Engine

- KI-dedizierter Chip exklusiv für Apple Silicon
- Bis zu 15,8 TOPS (M1 Pro/Max)
- Ultra-niedriger Stromverbrauch (1/10 oder weniger im Vergleich zur GPU)
- Optimiert für f16-Operationen

### Integration mit TensorLogic

- Alle f16-Operationen (Neural Engine optimiert)
- Nahtlose Integration mit Metal GPU
- Automatische Modelformaterkennung

## CoreML-Modelle erstellen

CoreML-Modelle werden typischerweise mit Pythons coreMLtools erstellt:

```python
import coremltools as ct
import torch

# PyTorch-Modell erstellen
model = MyModel()
model.eval()

# Traced-Modell erstellen
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# In CoreML konvertieren
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Neural Engine-Optimierung
    compute_precision=ct.precision.FLOAT16  # f16-Präzision
)

# Speichern
mlmodel.save("model.mlpackage")
```

## Verwendung in TensorLogic

### 1. CoreML-Modell laden (nur macOS)

```tensorlogic
model = load_model("model.mlpackage")
// oder
model = load_model("model.mlmodelc")
```

### 2. Metadaten prüfen

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Best Practices für Neural Engine-Optimierung

### 1. Datentyp: f16 verwenden

✅ Empfohlen: `compute_precision=ct.precision.FLOAT16`
❌ Nicht empfohlen: FLOAT32 (wird auf GPU ausgeführt)

### 2. Modellformat: mlprogram-Format verwenden

✅ Empfohlen: `convert_to="mlprogram"`
❌ Nicht empfohlen: `convert_to="neuralnetwork"` (Legacy-Format)

### 3. Batch-Größe: 1 ist optimal

✅ Empfohlen: `batch_size=1`
⚠️ Hinweis: `batch_size>1` wird möglicherweise auf GPU ausgeführt

### 4. Eingabegröße: Feste Größe ist optimal

✅ Empfohlen: `shape=[1, 3, 224, 224]`
⚠️ Hinweis: Variable Größen haben eingeschränkte Optimierung

## Unterstützte Operationen

### Auf Neural Engine schnell ausgeführte Operationen

- ✅ Faltungen (conv2d, depthwise_conv)
- ✅ Vollständig verbundene Schichten (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalisierung (batch_norm, layer_norm)
- ✅ Aktivierungsfunktionen (relu, gelu, sigmoid, tanh)
- ✅ Elementweise Operationen (add, mul, sub, div)

## Leistungsvergleich

ResNet-50 Inferenz (224x224 Bild):

| Gerät              | Latenz  | Leistung | Effizienz |
|-------------------|---------|----------|-----------|
| Neural Engine     | ~3ms    | ~0,5W    | Höchste   |
| Metal GPU (M1)    | ~8ms    | ~5W      | Mittel    |
| CPU (M1)          | ~50ms   | ~2W      | Niedrig   |

## Modellformatauswahl

### Empfohlene Formate nach Anwendungsfall

**Training**: SafeTensors
- PyTorch-kompatibel
- Gewichtsspeicherung/-laden
- Training auf Metal GPU

**Inferenz (iOS/macOS)**: CoreML
- Neural Engine-Optimierung
- Ultra-niedriger Stromverbrauch
- App-Integration

**Inferenz (Allgemein)**: GGUF
- Quantisierungsunterstützung
- Plattformübergreifend
- Speichereffizient

## Referenzen

- [CoreML Offizielle Dokumentation](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Neural Engine Leitfaden](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Modell-Ladeführer](model_loading.md)
