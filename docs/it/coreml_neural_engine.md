# Guida all'Integrazione CoreML & Neural Engine

Questa guida spiega come usare modelli CoreML in TensorLogic ed eseguire inferenza ad alta velocità su Apple Neural Engine.

## Su CoreML e Neural Engine

### CoreML

- Framework di machine learning proprietario di Apple
- Ottimizzato esclusivamente per iOS/macOS
- Sfrutta automaticamente Neural Engine, GPU e CPU
- Formato .mlmodel / .mlmodelc

### Neural Engine

- Chip dedicato all'IA esclusivo di Apple Silicon
- Fino a 15,8 TOPS (M1 Pro/Max)
- Consumo energetico ultra-basso (1/10 o meno rispetto alla GPU)
- Ottimizzato per operazioni f16

### Integrazione con TensorLogic

- Tutte le operazioni f16 (ottimizzato per Neural Engine)
- Integrazione fluida con Metal GPU
- Rilevamento automatico del formato del modello

## Creare Modelli CoreML

I modelli CoreML sono tipicamente creati con coreMLtools di Python:

```python
import coremltools as ct
import torch

# Creare modello PyTorch
model = MyModel()
model.eval()

# Creare modello tracciato
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convertire in CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Ottimizzazione Neural Engine
    compute_precision=ct.precision.FLOAT16  # precisione f16
)

# Salvare
mlmodel.save("model.mlpackage")
```

## Usare in TensorLogic

### 1. Caricare Modello CoreML (solo macOS)

```tensorlogic
model = load_model("model.mlpackage")
// o
model = load_model("model.mlmodelc")
```

### 2. Verificare Metadati

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Migliori Pratiche di Ottimizzazione Neural Engine

### 1. Tipo di Dati: Usare f16

✅ Raccomandato: `compute_precision=ct.precision.FLOAT16`
❌ Non raccomandato: FLOAT32 (eseguito su GPU)

### 2. Formato Modello: Usare formato mlprogram

✅ Raccomandato: `convert_to="mlprogram"`
❌ Non raccomandato: `convert_to="neuralnetwork"` (formato legacy)

### 3. Dimensione Batch: 1 è ottimale

✅ Raccomandato: `batch_size=1`
⚠️ Nota: `batch_size>1` può essere eseguito su GPU

### 4. Dimensione Input: Dimensione fissa è ottimale

✅ Raccomandato: `shape=[1, 3, 224, 224]`
⚠️ Nota: Dimensioni variabili hanno ottimizzazione limitata

## Operazioni Supportate

### Operazioni eseguite velocemente su Neural Engine

- ✅ Convoluzioni (conv2d, depthwise_conv)
- ✅ Layer completamente connessi (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalizzazione (batch_norm, layer_norm)
- ✅ Funzioni di attivazione (relu, gelu, sigmoid, tanh)
- ✅ Operazioni elemento per elemento (add, mul, sub, div)

## Confronto delle Prestazioni

Inferenza ResNet-50 (immagine 224x224):

| Dispositivo        | Latenza  | Potenza  | Efficienza |
|-------------------|----------|----------|------------|
| Neural Engine     | ~3ms     | ~0,5W    | Massima    |
| Metal GPU (M1)    | ~8ms     | ~5W      | Media      |
| CPU (M1)          | ~50ms    | ~2W      | Bassa      |

## Selezione del Formato del Modello

### Formati Consigliati per Caso d'Uso

**Addestramento**: SafeTensors
- Compatibile con PyTorch
- Salvataggio/caricamento pesi
- Addestramento su Metal GPU

**Inferenza (iOS/macOS)**: CoreML
- Ottimizzazione Neural Engine
- Consumo energetico ultra-basso
- Integrazione app

**Inferenza (Generale)**: GGUF
- Supporto quantizzazione
- Multi-piattaforma
- Efficiente in memoria

## Riferimenti

- [Documentazione Ufficiale CoreML](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Guida Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Guida al Caricamento dei Modelli](model_loading.md)
