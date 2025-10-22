# Referência da API TensorLogic

Referência completa da API para todas as operações disponíveis no TensorLogic.

## Índice

1. [Criação de Tensores](#criação-de-tensores)
2. [Operações de Forma](#operações-de-forma)
3. [Funções Matemáticas](#funções-matemáticas)
4. [Operações de Agregação](#operações-de-agregação)
5. [Funções de Ativação](#funções-de-ativação)
6. [Operações de Matriz](#operações-de-matriz)
7. [Normalização](#normalização)
8. [Operações de Mascaramento](#operações-de-mascaramento)
9. [Operações de Indexação](#operações-de-indexação)
10. [Embeddings](#embeddings)
11. [Amostragem](#amostragem)
12. [Operações Fundidas](#operações-fundidas)
13. [Otimização](#otimização)
14. [Outras Operações](#outras-operações)
15. [Operadores](#operadores)
16. [Definições de Tipos](#definições-de-tipos)

---

## Criação de Tensores

### `zeros(shape: Array<Int>) -> Tensor`

Cria um tensor preenchido com zeros.

**Parâmetros:**
- `shape`: Array especificando as dimensões do tensor

**Retorna:** Tensor preenchido com 0

**Exemplo:**
```tensorlogic
let z = zeros([2, 3])  // Tensor 2x3 de zeros
```

---

### `ones(shape: Array<Int>) -> Tensor`

Cria um tensor preenchido com uns.

**Parâmetros:**
- `shape`: Array especificando as dimensões do tensor

**Retorna:** Tensor preenchido com 1

**Exemplo:**
```tensorlogic
let o = ones([2, 3])  // Tensor 2x3 de uns
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Gera codificação posicional sinusoidal para Transformers.

**Parâmetros:**
- `seq_len`: Comprimento da sequência
- `d_model`: Dimensão do modelo

**Retorna:** Tensor de forma `[seq_len, d_model]`

**Definição Matemática:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Exemplo:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Casos de Uso:**
- Modelos Transformer
- Modelos sequência-para-sequência
- Mecanismos de atenção

**Referências:**
- arXiv:2510.12269 (Tabela 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Operações de Forma

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Altera a forma do tensor preservando os dados.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `new_shape`: Forma alvo

**Retorna:** Tensor remodelado

**Exemplo:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Restrições:**
- O número total de elementos deve permanecer o mesmo

---

### `flatten(tensor: Tensor) -> Tensor`

Achata o tensor para 1D.

**Parâmetros:**
- `tensor`: Tensor de entrada

**Retorna:** Tensor 1D

**Exemplo:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Transpõe um tensor 2D (troca eixos).

**Parâmetros:**
- `tensor`: Tensor 2D de entrada

**Retorna:** Tensor transposto

**Exemplo:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Reordena as dimensões do tensor.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dims`: Nova ordem das dimensões

**Retorna:** Tensor permutado

**Exemplo:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Adiciona uma dimensão de tamanho 1 na posição especificada.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Posição para inserir a nova dimensão

**Retorna:** Tensor com dimensão adicionada

**Exemplo:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Remove todas as dimensões de tamanho 1.

**Parâmetros:**
- `tensor`: Tensor de entrada

**Retorna:** Tensor com dimensões de tamanho 1 removidas

**Exemplo:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Divide o tensor em múltiplos tensores ao longo da dimensão especificada.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `sizes`: Tamanho de cada seção dividida
- `dim`: Dimensão para dividir

**Retorna:** Array de tensores

**Exemplo:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 tensores: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Divide o tensor em número especificado de pedaços.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `chunks`: Número de pedaços
- `dim`: Dimensão para dividir

**Retorna:** Array de tensores

**Exemplo:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 tensores de [4,4] cada
```

---

## Funções Matemáticas

### `exp(tensor: Tensor) -> Tensor`

Aplica a função exponencial elemento por elemento.

**Definição Matemática:** `exp(x) = e^x`

**Exemplo:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Aplica o logaritmo natural elemento por elemento.

**Definição Matemática:** `log(x) = ln(x)`

**Exemplo:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Aplica a raiz quadrada elemento por elemento.

**Definição Matemática:** `sqrt(x) = √x`

**Exemplo:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Eleva os elementos do tensor à potência especificada.

**Definição Matemática:** `pow(x, n) = x^n`

**Exemplo:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Aplica a função seno elemento por elemento.

**Exemplo:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Aplica a função cosseno elemento por elemento.

**Exemplo:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Aplica a função tangente elemento por elemento.

**Exemplo:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Operações de Agregação

### `sum(tensor: Tensor) -> Number`

Calcula a soma de todos os elementos.

**Exemplo:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Calcula a média de todos os elementos.

**Exemplo:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Retorna o valor máximo no tensor.

**Exemplo:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Retorna o valor mínimo no tensor.

**Exemplo:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Retorna os índices dos valores máximos ao longo da dimensão especificada.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensão para encontrar o máximo

**Retorna:** Tensor de índices

**Exemplo:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Índices máximos ao longo da dimensão 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Retorna os índices dos valores mínimos ao longo da dimensão especificada.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensão para encontrar o mínimo

**Retorna:** Tensor de índices

**Exemplo:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Índices mínimos ao longo da dimensão 1
```

---

## Funções de Ativação

### `relu(tensor: Tensor) -> Tensor`

Ativação Rectified Linear Unit.

**Definição Matemática:** `relu(x) = max(0, x)`

**Exemplo:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Função de ativação sigmoide.

**Definição Matemática:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Exemplo:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Ativação Gaussian Error Linear Unit (usada em BERT, GPT).

**Definição Matemática:** 
```
gelu(x) = x * Φ(x)
onde Φ(x) é a função de distribuição cumulativa da distribuição normal padrão
```

**Exemplo:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Casos de Uso:**
- Modelos BERT, GPT
- Arquiteturas Transformer modernas

---

### `tanh(tensor: Tensor) -> Tensor`

Ativação tangente hiperbólica.

**Definição Matemática:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Exemplo:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Aplica normalização softmax ao longo da dimensão especificada.

**Definição Matemática:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensão para aplicar softmax

**Retorna:** Tensor de distribuição de probabilidade

**Exemplo:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Casos de Uso:**
- Mecanismos de atenção
- Camadas de saída de classificação
- Distribuições de probabilidade

---

## Operações de Matriz

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Multiplicação de matrizes.

**Parâmetros:**
- `a`: Matriz da esquerda
- `b`: Matriz da direita

**Retorna:** Resultado da multiplicação de matrizes

**Exemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalização

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Aplica normalização de camada.

**Definição Matemática:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Parâmetros:**
- `tensor`: Tensor de entrada
- `normalized_shape`: Forma para normalizar
- `eps`: Pequeno valor para estabilidade numérica (padrão: 1e-5)

**Exemplo:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Casos de Uso:**
- Camadas Transformer
- Redes recorrentes
- Redes neurais profundas

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Aplica normalização em lote.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `running_mean`: Média móvel
- `running_var`: Variância móvel
- `eps`: Pequeno valor para estabilidade numérica

**Exemplo:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Aplica dropout (zera elementos aleatoriamente com probabilidade p).

**Parâmetros:**
- `tensor`: Tensor de entrada
- `p`: Probabilidade de dropout (0.0 a 1.0)

**Retorna:** Tensor com elementos aleatórios zerados

**Exemplo:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Casos de Uso:**
- Regularização de treinamento
- Prevenção de overfitting
- Aprendizado em conjunto

---

## Operações de Mascaramento

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Aplica máscara de atenção (define posições mascaradas como -inf).

**Parâmetros:**
- `tensor`: Tensor de pontuações de atenção
- `mask`: Máscara binária (1 = manter, 0 = mascarar)

**Retorna:** Tensor mascarado

**Exemplo:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Casos de Uso:**
- Mecanismos de atenção Transformer
- Mascaramento de sequências
- Atenção causal (autoregressiva)

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Cria máscara de preenchimento para sequências de comprimento variável.

**Parâmetros:**
- `lengths`: Array de comprimentos de sequência reais
- `max_len`: Comprimento máximo de sequência

**Retorna:** Tensor de máscara binária

**Exemplo:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Resultado: [batch_size, max_len] onde 1 = token real, 0 = preenchimento
```

**Casos de Uso:**
- Manipulação de sequências de comprimento variável
- Processamento em lote
- Mascaramento de atenção

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Combina duas máscaras usando AND lógico.

**Parâmetros:**
- `mask1`: Primeira máscara
- `mask2`: Segunda máscara

**Retorna:** Máscara combinada

**Exemplo:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Casos de Uso:**
- Combinação de máscaras de preenchimento e atenção
- Mascaramento multi-restrição
- Padrões de atenção complexos

---

## Operações de Indexação

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Coleta valores ao longo da dimensão usando índices.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensão para coletar
- `indices`: Tensor de índices

**Retorna:** Tensor coletado

**Exemplo:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Casos de Uso:**
- Seleção de tokens
- Busca em feixe
- Indexação avançada

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Seleciona elementos em índices especificados ao longo da dimensão.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `dim`: Dimensão para selecionar
- `indices`: Array de índices

**Retorna:** Tensor selecionado

**Exemplo:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Selecionar linhas 0, 2, 5
```

---

## Embeddings

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Converte IDs de tokens em embeddings.

**Parâmetros:**
- `indices`: Sequência de IDs de tokens
- `vocab_size`: Tamanho do vocabulário
- `embed_dim`: Dimensão do embedding

**Retorna:** Tensor de embedding de forma `[seq_len, embed_dim]`

**Exemplo:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Casos de Uso:**
- Embeddings de palavras
- Representações de tokens
- Modelos de linguagem

**Referências:**
- arXiv:2510.12269 (Tabela 1)

---

## Amostragem

### `top_k(logits: Tensor, k: Int) -> Tensor`

Amostra token usando amostragem top-k.

**Parâmetros:**
- `logits`: Logits de saída do modelo
- `k`: Número de tokens principais a considerar

**Retorna:** ID de token amostrado

**Exemplo:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Casos de Uso:**
- Geração de texto
- Amostragem controlada
- Saídas diversas

**Referências:**
- arXiv:2510.12269 (Tabela 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Amostra token usando amostragem nucleus (top-p).

**Parâmetros:**
- `logits`: Logits de saída do modelo
- `p`: Limiar de probabilidade cumulativa (0.0 a 1.0)

**Retorna:** ID de token amostrado

**Exemplo:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Casos de Uso:**
- Geração de texto
- Seleção dinâmica de vocabulário
- Amostragem com controle de qualidade

**Referências:**
- arXiv:2510.12269 (Tabela 2)

---

## Operações Fundidas

Operações fundidas combinam múltiplas operações para melhor desempenho reduzindo overhead de memória e lançamentos de kernel.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Funde adição e ativação ReLU.

**Definição Matemática:** `fused_add_relu(x, y) = relu(x + y)`

**Exemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Desempenho:** ~1.5x mais rápido que operações separadas

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Funde multiplicação e ativação ReLU.

**Definição Matemática:** `fused_mul_relu(x, y) = relu(x * y)`

**Exemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Funde transformação afim (escala e deslocamento).

**Definição Matemática:** `fused_affine(x, s, b) = x * s + b`

**Exemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Casos de Uso:**
- Normalização em lote
- Normalização de camada
- Transformações lineares personalizadas

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Funde ativação GELU e transformação linear.

**Definição Matemática:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Exemplo:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Casos de Uso:**
- Camadas FFN de Transformer
- Arquiteturas BERT/GPT
- Caminhos críticos de desempenho

---

## Otimização

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Executa passo do otimizador SGD.

**Definição Matemática:** `params_new = params - lr * gradients`

**Parâmetros:**
- `params`: Parâmetros atuais
- `gradients`: Gradientes calculados
- `lr`: Taxa de aprendizado

**Retorna:** Parâmetros atualizados

**Exemplo:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Executa passo do otimizador Adam.

**Parâmetros:**
- `params`: Parâmetros atuais
- `gradients`: Gradientes calculados
- `m`: Estimativa do primeiro momento
- `v`: Estimativa do segundo momento
- `lr`: Taxa de aprendizado
- `beta1`: Taxa de decaimento do primeiro momento (padrão: 0.9)
- `beta2`: Taxa de decaimento do segundo momento (padrão: 0.999)
- `eps`: Constante de estabilidade numérica (padrão: 1e-8)

**Retorna:** Parâmetros atualizados

**Exemplo:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Outras Operações

### `tokenize(text: String) -> TokenIDs`

Converte texto em sequência de IDs de tokens.

**Parâmetros:**
- `text`: String de texto de entrada

**Retorna:** TokenIDs (Vec<u32>)

**Exemplo:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Casos de Uso:**
- Pré-processamento de texto
- Entrada de modelo de linguagem
- Pipelines NLP

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Transmite tensor para a forma especificada.

**Parâmetros:**
- `tensor`: Tensor de entrada
- `shape`: Forma alvo

**Retorna:** Tensor transmitido

**Exemplo:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Casos de Uso:**
- Alinhamento de formas
- Operações em lote
- Operações elemento por elemento com formas diferentes

---

## Operadores

TensorLogic suporta operadores matemáticos padrão:

### Operadores Aritméticos
- `+` : Adição
- `-` : Subtração
- `*` : Multiplicação elemento por elemento
- `/` : Divisão elemento por elemento

**Exemplo:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Operadores de Comparação
- `==` : Igual
- `!=` : Diferente
- `<`  : Menor que
- `<=` : Menor ou igual a
- `>`  : Maior que
- `>=` : Maior ou igual a

### Operadores Lógicos
- `&&` : E lógico
- `||` : OU lógico
- `!`  : NÃO lógico

---

## Definições de Tipos

### Tensor
Array multidimensional com aceleração GPU via Metal Performance Shaders.

**Propriedades:**
- Forma: Array de dimensões
- Dados: Elementos Float32
- Dispositivo: Dispositivo GPU Metal

### TokenIDs
Tipo especial para sequências de IDs de tokens.

**Definição:** `Vec<u32>`

**Casos de Uso:**
- Resultados de tokenização
- Buscas de embedding
- Processamento de sequências

### Number
Valores numéricos (Int ou Float).

**Variantes:**
- `Integer`: Inteiro com sinal de 64 bits
- `Float`: Ponto flutuante de 64 bits

---

## Referências

- **Artigo TensorLogic**: arXiv:2510.12269
- **Arquitetura Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Documentação Relacionada

- [Guia de Introdução](./getting_started.md)
- [Referência da Linguagem](./language_reference.md)
- [Exemplos](../../examples/)
- [Operações Adicionadas em 2025](../added_operations_2025.md)
- [Lista de TODOs](../TODO.md)

---

**Última Atualização:** 2025-01-22

**Versão do TensorLogic:** 0.1.1+

**Operações Totais:** 48 funções + 4 operadores
