# Guia de Modelos Quantizados GGUF

Este documento explica como carregar e usar modelos quantizados no formato GGUF (compatível com llama.cpp) no TensorLogic.

## Sobre o Formato GGUF

GGUF (GGML Universal Format) é um formato de quantização eficiente para modelos de linguagem de grande porte desenvolvido pelo projeto llama.cpp.

### Características Principais

- Eficiência de memória através de quantização de 4-bit/8-bit (até 8x de compressão)
- Quantização baseada em blocos mantém a precisão
- Compatível com llama.cpp, Ollama, LM Studio e mais

### Formatos de Quantização Suportados pelo TensorLogic

- ✅ **Q4_0**: Quantização de 4-bit (máxima compressão)
- ✅ **Q8_0**: Quantização de 8-bit (precisão e compressão equilibradas)
- ✅ **F16**: Ponto flutuante de 16-bit (alta precisão)
- ✅ **F32**: Ponto flutuante de 32-bit (máxima precisão)

## Uso Básico

### 1. Carregar Modelo Quantizado

Automaticamente desquantizado para f16 e carregado na GPU Metal:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Obter Tensores do Modelo

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Escolher Formato de Quantização

### Q4_0 (4-bit)

- **Memória**: Uso mínimo (~1/8 do modelo original)
- **Velocidade**: Inferência mais rápida
- **Precisão**: Leve degradação (geralmente aceitável)
- **Casos de Uso**: Chatbots, geração de texto geral

### Q8_0 (8-bit)

- **Memória**: Uso moderado (~1/4 do modelo original)
- **Velocidade**: Rápido
- **Precisão**: Alta (quase equivalente a F16)
- **Casos de Uso**: Geração de alta qualidade, assistentes de codificação

### F16 (16-bit)

- **Memória**: ~1/2 do modelo original
- **Velocidade**: Padrão
- **Precisão**: Formato nativo do TensorLogic, otimizado para GPU Metal
- **Casos de Uso**: Quando a máxima qualidade é necessária

## Exemplo Prático: Embeddings de Tokens

```tensorlogic
// Obter embeddings de tokens do modelo LLama
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Obter vetor de embedding do ID de token
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Economia de Memória com Quantização

Exemplo: Modelo LLama-7B (7 bilhões de parâmetros):

| Formato    | Uso de Memória | Compressão |
|------------|----------------|------------|
| F32 (orig) | ~28 GB         | 1x         |
| F16        | ~14 GB         | 2x         |
| Q8_0       | ~7 GB          | 4x         |
| Q4_0       | ~3.5 GB        | 8x         |

TensorLogic converte todos os formatos para f16 ao carregar e os executa eficientemente na GPU Metal.

## Baixar e Instalar Modelos

### 1. Baixar Modelos GGUF do HuggingFace

Exemplo: https://huggingface.co/TheBloke

### 2. Modelos Recomendados (para iniciantes)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. Carregar no TensorLogic

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Notas Importantes

- Modelos quantizados são somente leitura (não podem ser salvos do TensorLogic)
- Use modelos não quantizados (F16/F32) para treinamento
- Q4/Q8 são otimizados apenas para inferência
- Todos os formatos de quantização são automaticamente desquantizados para f16 e carregados na GPU

## Referências

- [Especificação GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Modelos GGUF do HuggingFace](https://huggingface.co/TheBloke)
- [Guia de Carregamento de Modelos](model_loading.md)
