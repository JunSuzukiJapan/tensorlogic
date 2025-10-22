# Riferimento API TensorLogic

Riferimento API completo per tutte le operazioni disponibili in TensorLogic.

## Indice

1. [Creazione di Tensori](#creazione-di-tensori)
2. [Operazioni di Forma](#operazioni-di-forma)
3. [Funzioni Matematiche](#funzioni-matematiche)
4. [Operazioni di Aggregazione](#operazioni-di-aggregazione)
5. [Funzioni di Attivazione](#funzioni-di-attivazione)
6. [Operazioni Matriciali](#operazioni-matriciali)
7. [Normalizzazione](#normalizzazione)
8. [Operazioni di Mascheramento](#operazioni-di-mascheramento)
9. [Operazioni di Indicizzazione](#operazioni-di-indicizzazione)
10. [Embeddings](#embeddings)
11. [Campionamento](#campionamento)
12. [Operazioni Fuse](#operazioni-fuse)
13. [Ottimizzazione](#ottimizzazione)
14. [Altre Operazioni](#altre-operazioni)
15. [Operatori](#operatori)
16. [Definizioni di Tipi](#definizioni-di-tipi)

---

## Creazione di Tensori

### `zeros(shape: Array<Int>) -> Tensor`

Crea un tensore riempito di zeri.

**Parametri:**
- `shape`: Array che specifica le dimensioni del tensore

**Restituisce:** Tensore riempito di 0

**Esempio:**
```tensorlogic
let z = zeros([2, 3])  // Tensore 2x3 di zeri
```

---

### `ones(shape: Array<Int>) -> Tensor`

Crea un tensore riempito di uni.

**Parametri:**
- `shape`: Array che specifica le dimensioni del tensore

**Restituisce:** Tensore riempito di 1

**Esempio:**
```tensorlogic
let o = ones([2, 3])  // Tensore 2x3 di uni
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Genera codifica posizionale sinusoidale per Transformer.

**Parametri:**
- `seq_len`: Lunghezza della sequenza
- `d_model`: Dimensione del modello

**Restituisce:** Tensore di forma `[seq_len, d_model]`

**Definizione Matematica:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Esempio:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Casi d'Uso:**
- Modelli Transformer
- Modelli sequenza-a-sequenza
- Meccanismi di attenzione

**Riferimenti:**
- arXiv:2510.12269 (Tabella 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Operazioni di Forma

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Cambia la forma del tensore preservando i dati.

**Parametri:**
- `tensor`: Tensore di input
- `new_shape`: Forma target

**Restituisce:** Tensore rimodellato

**Esempio:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Vincoli:**
- Il numero totale di elementi deve rimanere uguale

---

### `flatten(tensor: Tensor) -> Tensor`

Appiattisce il tensore a 1D.

**Parametri:**
- `tensor`: Tensore di input

**Restituisce:** Tensore 1D

**Esempio:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Traspone un tensore 2D (scambia gli assi).

**Parametri:**
- `tensor`: Tensore 2D di input

**Restituisce:** Tensore trasposto

**Esempio:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Riordina le dimensioni del tensore.

**Parametri:**
- `tensor`: Tensore di input
- `dims`: Nuovo ordine delle dimensioni

**Restituisce:** Tensore permutato

**Esempio:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Aggiunge una dimensione di dimensione 1 nella posizione specificata.

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Posizione per inserire la nuova dimensione

**Restituisce:** Tensore con dimensione aggiunta

**Esempio:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Rimuove tutte le dimensioni di dimensione 1.

**Parametri:**
- `tensor`: Tensore di input

**Restituisce:** Tensore con dimensioni di dimensione 1 rimosse

**Esempio:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Divide il tensore in più tensori lungo la dimensione specificata.

**Parametri:**
- `tensor`: Tensore di input
- `sizes`: Dimensione di ogni sezione divisa
- `dim`: Dimensione lungo cui dividere

**Restituisce:** Array di tensori

**Esempio:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 tensori: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Divide il tensore nel numero specificato di parti.

**Parametri:**
- `tensor`: Tensore di input
- `chunks`: Numero di parti
- `dim`: Dimensione lungo cui dividere

**Restituisce:** Array di tensori

**Esempio:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 tensori di [4,4] ciascuno
```

---

## Funzioni Matematiche

### `exp(tensor: Tensor) -> Tensor`

Applica la funzione esponenziale elemento per elemento.

**Definizione Matematica:** `exp(x) = e^x`

**Esempio:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Applica il logaritmo naturale elemento per elemento.

**Definizione Matematica:** `log(x) = ln(x)`

**Esempio:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Applica la radice quadrata elemento per elemento.

**Definizione Matematica:** `sqrt(x) = √x`

**Esempio:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Eleva gli elementi del tensore alla potenza specificata.

**Definizione Matematica:** `pow(x, n) = x^n`

**Esempio:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Applica la funzione seno elemento per elemento.

**Esempio:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Applica la funzione coseno elemento per elemento.

**Esempio:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Applica la funzione tangente elemento per elemento.

**Esempio:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Operazioni di Aggregazione

### `sum(tensor: Tensor) -> Number`

Calcola la somma di tutti gli elementi.

**Esempio:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Calcola la media di tutti gli elementi.

**Esempio:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Restituisce il valore massimo nel tensore.

**Esempio:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Restituisce il valore minimo nel tensore.

**Esempio:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Restituisce gli indici dei valori massimi lungo la dimensione specificata.

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Dimensione per trovare il massimo

**Restituisce:** Tensore di indici

**Esempio:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Indici massimi lungo la dimensione 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Restituisce gli indici dei valori minimi lungo la dimensione specificata.

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Dimensione per trovare il minimo

**Restituisce:** Tensore di indici

**Esempio:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Indici minimi lungo la dimensione 1
```

---

## Funzioni di Attivazione

### `relu(tensor: Tensor) -> Tensor`

Attivazione Rectified Linear Unit.

**Definizione Matematica:** `relu(x) = max(0, x)`

**Esempio:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Funzione di attivazione sigmoide.

**Definizione Matematica:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Esempio:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Attivazione Gaussian Error Linear Unit (usata in BERT, GPT).

**Definizione Matematica:** 
```
gelu(x) = x * Φ(x)
dove Φ(x) è la funzione di distribuzione cumulativa della distribuzione normale standard
```

**Esempio:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Casi d'Uso:**
- Modelli BERT, GPT
- Architetture Transformer moderne

---

### `tanh(tensor: Tensor) -> Tensor`

Attivazione tangente iperbolica.

**Definizione Matematica:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Esempio:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Applica la normalizzazione softmax lungo la dimensione specificata.

**Definizione Matematica:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Dimensione per applicare softmax

**Restituisce:** Tensore di distribuzione di probabilità

**Esempio:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Casi d'Uso:**
- Meccanismi di attenzione
- Livelli di output di classificazione
- Distribuzioni di probabilità

---

## Operazioni Matriciali

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Moltiplicazione matriciale.

**Parametri:**
- `a`: Matrice sinistra
- `b`: Matrice destra

**Restituisce:** Risultato della moltiplicazione matriciale

**Esempio:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalizzazione

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Applica la normalizzazione di livello.

**Definizione Matematica:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Parametri:**
- `tensor`: Tensore di input
- `normalized_shape`: Forma per normalizzare
- `eps`: Piccolo valore per la stabilità numerica (predefinito: 1e-5)

**Esempio:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Casi d'Uso:**
- Livelli Transformer
- Reti ricorrenti
- Reti neurali profonde

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Applica la normalizzazione di batch.

**Parametri:**
- `tensor`: Tensore di input
- `running_mean`: Media mobile
- `running_var`: Varianza mobile
- `eps`: Piccolo valore per la stabilità numerica

**Esempio:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Applica il dropout (mette casualmente elementi a zero con probabilità p).

**Parametri:**
- `tensor`: Tensore di input
- `p`: Probabilità di dropout (0.0 a 1.0)

**Restituisce:** Tensore con elementi casuali messi a zero

**Esempio:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Casi d'Uso:**
- Regolarizzazione del training
- Prevenzione dell'overfitting
- Apprendimento d'insieme

---

## Operazioni di Mascheramento

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Applica maschera di attenzione (imposta posizioni mascherate a -inf).

**Parametri:**
- `tensor`: Tensore di punteggi di attenzione
- `mask`: Maschera binaria (1 = mantieni, 0 = maschera)

**Restituisce:** Tensore mascherato

**Esempio:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Casi d'Uso:**
- Meccanismi di attenzione Transformer
- Mascheramento di sequenze
- Attenzione causale (autoregressiva)

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Crea maschera di riempimento per sequenze di lunghezza variabile.

**Parametri:**
- `lengths`: Array di lunghezze di sequenza reali
- `max_len`: Lunghezza massima di sequenza

**Restituisce:** Tensore di maschera binaria

**Esempio:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Risultato: [batch_size, max_len] dove 1 = token reale, 0 = riempimento
```

**Casi d'Uso:**
- Gestione di sequenze di lunghezza variabile
- Elaborazione in batch
- Mascheramento di attenzione

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Combina due maschere usando AND logico.

**Parametri:**
- `mask1`: Prima maschera
- `mask2`: Seconda maschera

**Restituisce:** Maschera combinata

**Esempio:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Casi d'Uso:**
- Combinazione di maschere di riempimento e attenzione
- Mascheramento multi-vincolo
- Pattern di attenzione complessi

---

## Operazioni di Indicizzazione

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Raccoglie valori lungo la dimensione usando gli indici.

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Dimensione per raccogliere
- `indices`: Tensore di indici

**Restituisce:** Tensore raccolto

**Esempio:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Casi d'Uso:**
- Selezione di token
- Ricerca a fascio
- Indicizzazione avanzata

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Seleziona elementi agli indici specificati lungo la dimensione.

**Parametri:**
- `tensor`: Tensore di input
- `dim`: Dimensione per selezionare
- `indices`: Array di indici

**Restituisce:** Tensore selezionato

**Esempio:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Seleziona righe 0, 2, 5
```

---

## Embeddings

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Converte ID di token in embeddings.

**Parametri:**
- `indices`: Sequenza di ID di token
- `vocab_size`: Dimensione del vocabolario
- `embed_dim`: Dimensione dell'embedding

**Restituisce:** Tensore di embedding di forma `[seq_len, embed_dim]`

**Esempio:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Casi d'Uso:**
- Embeddings di parole
- Rappresentazioni di token
- Modelli di linguaggio

**Riferimenti:**
- arXiv:2510.12269 (Tabella 1)

---

## Campionamento

### `top_k(logits: Tensor, k: Int) -> Tensor`

Campiona token usando campionamento top-k.

**Parametri:**
- `logits`: Logits di output del modello
- `k`: Numero di token principali da considerare

**Restituisce:** ID di token campionato

**Esempio:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Casi d'Uso:**
- Generazione di testo
- Campionamento controllato
- Output diversi

**Riferimenti:**
- arXiv:2510.12269 (Tabella 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Campiona token usando campionamento nucleus (top-p).

**Parametri:**
- `logits`: Logits di output del modello
- `p`: Soglia di probabilità cumulativa (0.0 a 1.0)

**Restituisce:** ID di token campionato

**Esempio:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Casi d'Uso:**
- Generazione di testo
- Selezione dinamica del vocabolario
- Campionamento con controllo di qualità

**Riferimenti:**
- arXiv:2510.12269 (Tabella 2)

---

## Operazioni Fuse

Le operazioni fuse combinano più operazioni per prestazioni migliori riducendo l'overhead di memoria e i lanci di kernel.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fonde addizione e attivazione ReLU.

**Definizione Matematica:** `fused_add_relu(x, y) = relu(x + y)`

**Esempio:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Prestazioni:** ~1.5x più veloce delle operazioni separate

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fonde moltiplicazione e attivazione ReLU.

**Definizione Matematica:** `fused_mul_relu(x, y) = relu(x * y)`

**Esempio:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Fonde trasformazione affine (scala e spostamento).

**Definizione Matematica:** `fused_affine(x, s, b) = x * s + b`

**Esempio:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Casi d'Uso:**
- Normalizzazione di batch
- Normalizzazione di livello
- Trasformazioni lineari personalizzate

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Fonde attivazione GELU e trasformazione lineare.

**Definizione Matematica:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Esempio:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Casi d'Uso:**
- Livelli FFN di Transformer
- Architetture BERT/GPT
- Percorsi critici per le prestazioni

---

## Ottimizzazione

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Esegue passo di ottimizzatore SGD.

**Definizione Matematica:** `params_new = params - lr * gradients`

**Parametri:**
- `params`: Parametri attuali
- `gradients`: Gradienti calcolati
- `lr`: Tasso di apprendimento

**Restituisce:** Parametri aggiornati

**Esempio:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Esegue passo di ottimizzatore Adam.

**Parametri:**
- `params`: Parametri attuali
- `gradients`: Gradienti calcolati
- `m`: Stima del primo momento
- `v`: Stima del secondo momento
- `lr`: Tasso di apprendimento
- `beta1`: Tasso di decadimento del primo momento (predefinito: 0.9)
- `beta2`: Tasso di decadimento del secondo momento (predefinito: 0.999)
- `eps`: Costante di stabilità numerica (predefinito: 1e-8)

**Restituisce:** Parametri aggiornati

**Esempio:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Altre Operazioni

### `tokenize(text: String) -> TokenIDs`

Converte testo in sequenza di ID di token.

**Parametri:**
- `text`: Stringa di testo di input

**Restituisce:** TokenIDs (Vec<u32>)

**Esempio:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Casi d'Uso:**
- Pre-elaborazione del testo
- Input del modello di linguaggio
- Pipeline NLP

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Trasmette tensore alla forma specificata.

**Parametri:**
- `tensor`: Tensore di input
- `shape`: Forma target

**Restituisce:** Tensore trasmesso

**Esempio:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Casi d'Uso:**
- Allineamento delle forme
- Operazioni in batch
- Operazioni elemento per elemento con forme diverse

---

## Operatori

TensorLogic supporta operatori matematici standard:

### Operatori Aritmetici
- `+` : Addizione
- `-` : Sottrazione
- `*` : Moltiplicazione elemento per elemento
- `/` : Divisione elemento per elemento

**Esempio:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Operatori di Confronto
- `==` : Uguale
- `!=` : Diverso
- `<`  : Minore di
- `<=` : Minore o uguale a
- `>`  : Maggiore di
- `>=` : Maggiore o uguale a

### Operatori Logici
- `&&` : AND logico
- `||` : OR logico
- `!`  : NOT logico

---

## Definizioni di Tipi

### Tensor
Array multidimensionale con accelerazione GPU tramite Metal Performance Shaders.

**Proprietà:**
- Forma: Array di dimensioni
- Dati: Elementi Float32
- Dispositivo: Dispositivo GPU Metal

### TokenIDs
Tipo speciale per sequenze di ID di token.

**Definizione:** `Vec<u32>`

**Casi d'Uso:**
- Risultati di tokenizzazione
- Ricerche di embedding
- Elaborazione di sequenze

### Number
Valori numerici (Int o Float).

**Varianti:**
- `Integer`: Intero con segno a 64 bit
- `Float`: Virgola mobile a 64 bit

---

## Riferimenti

- **Paper TensorLogic**: arXiv:2510.12269
- **Architettura Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Documentazione Correlata

- [Guida Introduttiva](./getting_started.md)
- [Riferimento del Linguaggio](./language_reference.md)
- [Esempi](../../examples/)
- [Operazioni Aggiunte nel 2025](../added_operations_2025.md)
- [Elenco dei TODO](../TODO.md)

---

**Ultimo Aggiornamento:** 2025-01-22

**Versione TensorLogic:** 0.1.1+

**Operazioni Totali:** 48 funzioni + 4 operatori
