# GGUF Quantisierte Modelle Leitfaden

Dieses Dokument erklärt, wie man GGUF-Format quantisierte Modelle (llama.cpp-kompatibel) in TensorLogic lädt und verwendet.

## Über das GGUF-Format

GGUF (GGML Universal Format) ist ein effizientes Quantisierungsformat für große Sprachmodelle, das vom llama.cpp-Projekt entwickelt wurde.

### Hauptmerkmale

- Speichereffizienz durch 4-Bit/8-Bit-Quantisierung (bis zu 8-fache Kompression)
- Blockbasierte Quantisierung erhält Genauigkeit
- Kompatibel mit llama.cpp, Ollama, LM Studio und mehr

### Von TensorLogic unterstützte Quantisierungsformate

- ✅ **Q4_0**: 4-Bit-Quantisierung (höchste Kompression)
- ✅ **Q8_0**: 8-Bit-Quantisierung (ausgewogene Präzision und Kompression)
- ✅ **F16**: 16-Bit-Gleitkomma (hohe Präzision)
- ✅ **F32**: 32-Bit-Gleitkomma (höchste Präzision)

## Grundlegende Verwendung

### 1. Quantisiertes Modell laden

Automatisch zu f16 dequantisiert und in Metal GPU geladen:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Tensoren aus dem Modell abrufen

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Auswahl des Quantisierungsformats

### Q4_0 (4-Bit)

- **Speicher**: Minimale Nutzung (~1/8 des Originalmodells)
- **Geschwindigkeit**: Schnellste Inferenz
- **Genauigkeit**: Leichte Verschlechterung (normalerweise akzeptabel)
- **Anwendungsfälle**: Chatbots, allgemeine Textgenerierung

### Q8_0 (8-Bit)

- **Speicher**: Moderate Nutzung (~1/4 des Originalmodells)
- **Geschwindigkeit**: Schnell
- **Genauigkeit**: Hoch (nahezu äquivalent zu F16)
- **Anwendungsfälle**: Hochwertige Generierung, Coding-Assistenten

### F16 (16-Bit)

- **Speicher**: ~1/2 des Originalmodells
- **Geschwindigkeit**: Standard
- **Genauigkeit**: TensorLogic natives Format, Metal GPU optimiert
- **Anwendungsfälle**: Wenn höchste Qualität erforderlich ist

## Praktisches Beispiel: Token-Einbettungen

```tensorlogic
// Token-Einbettungen aus LLama-Modell abrufen
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Einbettungsvektor aus Token-ID abrufen
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Speichereinsparungen durch Quantisierung

Beispiel: LLama-7B-Modell (7 Milliarden Parameter):

| Format      | Speichernutzung | Kompression |
|-------------|-----------------|-------------|
| F32 (orig)  | ~28 GB          | 1x          |
| F16         | ~14 GB          | 2x          |
| Q8_0        | ~7 GB           | 4x          |
| Q4_0        | ~3.5 GB         | 8x          |

TensorLogic konvertiert alle Formate beim Laden zu f16 und führt sie effizient auf Metal GPU aus.

## Modelle herunterladen und installieren

### 1. GGUF-Modelle von HuggingFace herunterladen

Beispiel: https://huggingface.co/TheBloke

### 2. Empfohlene Modelle (für Anfänger)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. In TensorLogic laden

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Wichtige Hinweise

- Quantisierte Modelle sind schreibgeschützt (können nicht von TensorLogic gespeichert werden)
- Verwenden Sie nicht-quantisierte Modelle (F16/F32) für das Training
- Q4/Q8 sind nur für Inferenz optimiert
- Alle Quantisierungsformate werden automatisch zu f16 dequantisiert und auf GPU geladen

## Referenzen

- [GGUF-Spezifikation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF-Modelle](https://huggingface.co/TheBloke)
- [Modell-Ladeführer](model_loading.md)
