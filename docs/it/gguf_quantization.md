# Guida ai Modelli Quantizzati GGUF

Questo documento spiega come caricare e utilizzare modelli quantizzati in formato GGUF (compatibile con llama.cpp) in TensorLogic.

## Sul Formato GGUF

GGUF (GGML Universal Format) è un formato di quantizzazione efficiente per modelli di linguaggio di grandi dimensioni sviluppato dal progetto llama.cpp.

### Caratteristiche Principali

- Efficienza di memoria tramite quantizzazione a 4-bit/8-bit (fino a 8x di compressione)
- La quantizzazione basata su blocchi mantiene la precisione
- Compatibile con llama.cpp, Ollama, LM Studio e altro

### Formati di Quantizzazione Supportati da TensorLogic

- ✅ **Q4_0**: Quantizzazione a 4-bit (massima compressione)
- ✅ **Q8_0**: Quantizzazione a 8-bit (precisione e compressione bilanciate)
- ✅ **F16**: Virgola mobile a 16-bit (alta precisione)
- ✅ **F32**: Virgola mobile a 32-bit (massima precisione)

## Utilizzo Base

### 1. Caricare Modello Quantizzato

Automaticamente dequantizzato in f16 e caricato su GPU Metal:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Ottenere Tensori dal Modello

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Scegliere il Formato di Quantizzazione

### Q4_0 (4-bit)

- **Memoria**: Uso minimo (~1/8 del modello originale)
- **Velocità**: Inferenza più veloce
- **Precisione**: Leggera degradazione (solitamente accettabile)
- **Casi d'Uso**: Chatbot, generazione di testo generale

### Q8_0 (8-bit)

- **Memoria**: Uso moderato (~1/4 del modello originale)
- **Velocità**: Veloce
- **Precisione**: Alta (quasi equivalente a F16)
- **Casi d'Uso**: Generazione di alta qualità, assistenti di codifica

### F16 (16-bit)

- **Memoria**: ~1/2 del modello originale
- **Velocità**: Standard
- **Precisione**: Formato nativo di TensorLogic, ottimizzato per GPU Metal
- **Casi d'Uso**: Quando è richiesta la massima qualità

## Esempio Pratico: Embeddings di Token

```tensorlogic
// Ottenere embeddings di token dal modello LLama
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Ottenere vettore di embedding da ID token
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Risparmi di Memoria dalla Quantizzazione

Esempio: Modello LLama-7B (7 miliardi di parametri):

| Formato    | Uso Memoria | Compressione |
|------------|-------------|--------------|
| F32 (orig) | ~28 GB      | 1x           |
| F16        | ~14 GB      | 2x           |
| Q8_0       | ~7 GB       | 4x           |
| Q4_0       | ~3.5 GB     | 8x           |

TensorLogic converte tutti i formati in f16 al caricamento e li esegue efficientemente su GPU Metal.

## Scaricare e Installare Modelli

### 1. Scaricare Modelli GGUF da HuggingFace

Esempio: https://huggingface.co/TheBloke

### 2. Modelli Consigliati (per principianti)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. Caricare in TensorLogic

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Note Importanti

- I modelli quantizzati sono di sola lettura (non possono essere salvati da TensorLogic)
- Utilizzare modelli non quantizzati (F16/F32) per l'addestramento
- Q4/Q8 sono ottimizzati solo per l'inferenza
- Tutti i formati di quantizzazione vengono automaticamente dequantizzati in f16 e caricati su GPU

## Riferimenti

- [Specifica GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Modelli GGUF HuggingFace](https://huggingface.co/TheBloke)
- [Guida al Caricamento dei Modelli](model_loading.md)
