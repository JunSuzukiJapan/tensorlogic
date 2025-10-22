# TensorLogic API-Referenz

Vollständige API-Referenz für alle verfügbaren Operationen in TensorLogic.

## Inhaltsverzeichnis

1. [Tensor-Erstellung](#tensor-erstellung)
2. [Form-Operationen](#form-operationen)
3. [Mathematische Funktionen](#mathematische-funktionen)
4. [Aggregationsoperationen](#aggregationsoperationen)
5. [Aktivierungsfunktionen](#aktivierungsfunktionen)
6. [Matrixoperationen](#matrixoperationen)
7. [Normalisierung](#normalisierung)
8. [Maskierungsoperationen](#maskierungsoperationen)
9. [Indizierungsoperationen](#indizierungsoperationen)
10. [Einbettungen](#einbettungen)
11. [Sampling](#sampling)
12. [Fusionierte Operationen](#fusionierte-operationen)
13. [Optimierung](#optimierung)
14. [Weitere Operationen](#weitere-operationen)
15. [Operatoren](#operatoren)
16. [Typdefinitionen](#typdefinitionen)

---

## Tensor-Erstellung

### `zeros(shape: Array<Int>) -> Tensor`

Erstellt einen mit Nullen gefüllten Tensor.

**Parameter:**
- `shape`: Array, das die Tensor-Dimensionen angibt

**Rückgabe:** Mit 0 gefüllter Tensor

**Beispiel:**
```tensorlogic
let z = zeros([2, 3])  // 2x3-Tensor mit Nullen
```

---

### `ones(shape: Array<Int>) -> Tensor`

Erstellt einen mit Einsen gefüllten Tensor.

**Parameter:**
- `shape`: Array, das die Tensor-Dimensionen angibt

**Rückgabe:** Mit 1 gefüllter Tensor

**Beispiel:**
```tensorlogic
let o = ones([2, 3])  // 2x3-Tensor mit Einsen
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Erzeugt sinusförmige Positionskodierung für Transformer.

**Parameter:**
- `seq_len`: Sequenzlänge
- `d_model`: Modelldimension

**Rückgabe:** Tensor der Form `[seq_len, d_model]`

**Mathematische Definition:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Beispiel:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Anwendungsfälle:**
- Transformer-Modelle
- Sequenz-zu-Sequenz-Modelle
- Attention-Mechanismen

**Referenzen:**
- arXiv:2510.12269 (Tabelle 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Form-Operationen

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Ändert die Tensor-Form unter Beibehaltung der Daten.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `new_shape`: Zielform

**Rückgabe:** Umgeformter Tensor

**Beispiel:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Einschränkungen:**
- Gesamtanzahl der Elemente muss gleich bleiben

---

### `flatten(tensor: Tensor) -> Tensor`

Flacht Tensor zu 1D ab.

**Parameter:**
- `tensor`: Eingabe-Tensor

**Rückgabe:** 1D-Tensor

**Beispiel:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Transponiert einen 2D-Tensor (vertauscht Achsen).

**Parameter:**
- `tensor`: Eingabe-2D-Tensor

**Rückgabe:** Transponierter Tensor

**Beispiel:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Ordnet Tensor-Dimensionen neu an.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dims`: Neue Dimensionsreihenfolge

**Rückgabe:** Permutierter Tensor

**Beispiel:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Fügt eine Dimension der Größe 1 an der angegebenen Position hinzu.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Position zum Einfügen der neuen Dimension

**Rückgabe:** Tensor mit hinzugefügter Dimension

**Beispiel:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Entfernt alle Dimensionen der Größe 1.

**Parameter:**
- `tensor`: Eingabe-Tensor

**Rückgabe:** Tensor mit entfernten Dimensionen der Größe 1

**Beispiel:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Teilt Tensor in mehrere Tensoren entlang der angegebenen Dimension.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `sizes`: Größe jedes Teilabschnitts
- `dim`: Dimension zum Teilen

**Rückgabe:** Array von Tensoren

**Beispiel:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 Tensoren: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Teilt Tensor in angegebene Anzahl von Chunks.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `chunks`: Anzahl der Chunks
- `dim`: Dimension zum Teilen

**Rückgabe:** Array von Tensoren

**Beispiel:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 Tensoren von je [4,4]
```

---

## Mathematische Funktionen

### `exp(tensor: Tensor) -> Tensor`

Wendet Exponentialfunktion elementweise an.

**Mathematische Definition:** `exp(x) = e^x`

**Beispiel:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Wendet natürlichen Logarithmus elementweise an.

**Mathematische Definition:** `log(x) = ln(x)`

**Beispiel:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Wendet Quadratwurzel elementweise an.

**Mathematische Definition:** `sqrt(x) = √x`

**Beispiel:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Erhebt Tensor-Elemente zur angegebenen Potenz.

**Mathematische Definition:** `pow(x, n) = x^n`

**Beispiel:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Wendet Sinusfunktion elementweise an.

**Beispiel:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Wendet Kosinusfunktion elementweise an.

**Beispiel:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Wendet Tangensfunktion elementweise an.

**Beispiel:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Aggregationsoperationen

### `sum(tensor: Tensor) -> Number`

Berechnet Summe aller Elemente.

**Beispiel:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Berechnet Mittelwert aller Elemente.

**Beispiel:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Gibt Maximalwert im Tensor zurück.

**Beispiel:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Gibt Minimalwert im Tensor zurück.

**Beispiel:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Gibt Indizes der Maximalwerte entlang der angegebenen Dimension zurück.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Dimension zum Finden des Maximums

**Rückgabe:** Tensor von Indizes

**Beispiel:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Maximale Indizes entlang Dimension 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Gibt Indizes der Minimalwerte entlang der angegebenen Dimension zurück.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Dimension zum Finden des Minimums

**Rückgabe:** Tensor von Indizes

**Beispiel:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Minimale Indizes entlang Dimension 1
```

---

## Aktivierungsfunktionen

### `relu(tensor: Tensor) -> Tensor`

Rectified Linear Unit Aktivierung.

**Mathematische Definition:** `relu(x) = max(0, x)`

**Beispiel:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Sigmoid-Aktivierungsfunktion.

**Mathematische Definition:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Beispiel:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Gaussian Error Linear Unit Aktivierung (verwendet in BERT, GPT).

**Mathematische Definition:** 
```
gelu(x) = x * Φ(x)
wobei Φ(x) die kumulative Verteilungsfunktion der Standardnormalverteilung ist
```

**Beispiel:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Anwendungsfälle:**
- BERT, GPT-Modelle
- Moderne Transformer-Architekturen

---

### `tanh(tensor: Tensor) -> Tensor`

Hyperbolischer Tangens-Aktivierung.

**Mathematische Definition:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Beispiel:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Wendet Softmax-Normalisierung entlang der angegebenen Dimension an.

**Mathematische Definition:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Dimension für Softmax

**Rückgabe:** Wahrscheinlichkeitsverteilungs-Tensor

**Beispiel:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Anwendungsfälle:**
- Attention-Mechanismen
- Klassifikations-Ausgabeschichten
- Wahrscheinlichkeitsverteilungen

---

## Matrixoperationen

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Matrixmultiplikation.

**Parameter:**
- `a`: Linke Matrix
- `b`: Rechte Matrix

**Rückgabe:** Ergebnis der Matrixmultiplikation

**Beispiel:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalisierung

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Wendet Layer-Normalisierung an.

**Mathematische Definition:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Parameter:**
- `tensor`: Eingabe-Tensor
- `normalized_shape`: Form zur Normalisierung
- `eps`: Kleiner Wert für numerische Stabilität (Standard: 1e-5)

**Beispiel:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Anwendungsfälle:**
- Transformer-Schichten
- Rekurrente Netzwerke
- Tiefe neuronale Netzwerke

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Wendet Batch-Normalisierung an.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `running_mean`: Laufender Mittelwert
- `running_var`: Laufende Varianz
- `eps`: Kleiner Wert für numerische Stabilität

**Beispiel:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Wendet Dropout an (setzt zufällig Elemente mit Wahrscheinlichkeit p auf Null).

**Parameter:**
- `tensor`: Eingabe-Tensor
- `p`: Dropout-Wahrscheinlichkeit (0.0 bis 1.0)

**Rückgabe:** Tensor mit zufällig auf Null gesetzten Elementen

**Beispiel:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Anwendungsfälle:**
- Trainingsregularisierung
- Verhinderung von Overfitting
- Ensemble-Learning

---

## Maskierungsoperationen

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Wendet Attention-Maske an (setzt maskierte Positionen auf -inf).

**Parameter:**
- `tensor`: Attention-Scores-Tensor
- `mask`: Binäre Maske (1 = behalten, 0 = maskieren)

**Rückgabe:** Maskierter Tensor

**Beispiel:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Anwendungsfälle:**
- Transformer-Attention-Mechanismen
- Sequenzmaskierung
- Kausale (autoregressive) Attention

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Erstellt Padding-Maske für Sequenzen variabler Länge.

**Parameter:**
- `lengths`: Array der tatsächlichen Sequenzlängen
- `max_len`: Maximale Sequenzlänge

**Rückgabe:** Binärer Masken-Tensor

**Beispiel:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Ergebnis: [batch_size, max_len] wobei 1 = echtes Token, 0 = Padding
```

**Anwendungsfälle:**
- Handhabung von Sequenzen variabler Länge
- Batch-Verarbeitung
- Attention-Maskierung

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Kombiniert zwei Masken mit logischem UND.

**Parameter:**
- `mask1`: Erste Maske
- `mask2`: Zweite Maske

**Rückgabe:** Kombinierte Maske

**Beispiel:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Anwendungsfälle:**
- Kombinieren von Padding- und Attention-Masken
- Multi-Constraint-Maskierung
- Komplexe Attention-Muster

---

## Indizierungsoperationen

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Sammelt Werte entlang der Dimension mit Indizes.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Dimension zum Sammeln
- `indices`: Index-Tensor

**Rückgabe:** Gesammelter Tensor

**Beispiel:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Anwendungsfälle:**
- Token-Auswahl
- Beam-Search
- Erweiterte Indizierung

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Wählt Elemente an angegebenen Indizes entlang der Dimension aus.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `dim`: Dimension zur Auswahl
- `indices`: Array von Indizes

**Rückgabe:** Ausgewählter Tensor

**Beispiel:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Wähle Zeilen 0, 2, 5
```

---

## Einbettungen

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Konvertiert Token-IDs in Einbettungen.

**Parameter:**
- `indices`: Token-ID-Sequenz
- `vocab_size`: Vokabulargröße
- `embed_dim`: Einbettungsdimension

**Rückgabe:** Einbettungs-Tensor der Form `[seq_len, embed_dim]`

**Beispiel:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Anwendungsfälle:**
- Worteinbettungen
- Token-Darstellungen
- Sprachmodelle

**Referenzen:**
- arXiv:2510.12269 (Tabelle 1)

---

## Sampling

### `top_k(logits: Tensor, k: Int) -> Tensor`

Sampelt Token mit Top-k-Sampling.

**Parameter:**
- `logits`: Modell-Ausgabe-Logits
- `k`: Anzahl der zu berücksichtigenden Top-Tokens

**Rückgabe:** Gesampelte Token-ID

**Beispiel:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Anwendungsfälle:**
- Textgenerierung
- Kontrolliertes Sampling
- Vielfältige Ausgaben

**Referenzen:**
- arXiv:2510.12269 (Tabelle 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Sampelt Token mit Nucleus (Top-p) Sampling.

**Parameter:**
- `logits`: Modell-Ausgabe-Logits
- `p`: Kumulative Wahrscheinlichkeitsschwelle (0.0 bis 1.0)

**Rückgabe:** Gesampelte Token-ID

**Beispiel:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Anwendungsfälle:**
- Textgenerierung
- Dynamische Vokabularauswahl
- Qualitätskontrolliertes Sampling

**Referenzen:**
- arXiv:2510.12269 (Tabelle 2)

---

## Fusionierte Operationen

Fusionierte Operationen kombinieren mehrere Operationen für bessere Leistung durch Reduzierung von Speicher-Overhead und Kernel-Starts.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusioniert Addition und ReLU-Aktivierung.

**Mathematische Definition:** `fused_add_relu(x, y) = relu(x + y)`

**Beispiel:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Leistung:** ~1.5x schneller als separate Operationen

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusioniert Multiplikation und ReLU-Aktivierung.

**Mathematische Definition:** `fused_mul_relu(x, y) = relu(x * y)`

**Beispiel:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Fusioniert affine Transformation (Skalierung und Verschiebung).

**Mathematische Definition:** `fused_affine(x, s, b) = x * s + b`

**Beispiel:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Anwendungsfälle:**
- Batch-Normalisierung
- Layer-Normalisierung
- Benutzerdefinierte lineare Transformationen

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Fusioniert GELU-Aktivierung und lineare Transformation.

**Mathematische Definition:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Beispiel:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Anwendungsfälle:**
- Transformer-FFN-Schichten
- BERT/GPT-Architekturen
- Leistungskritische Pfade

---

## Optimierung

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Führt SGD-Optimizer-Schritt aus.

**Mathematische Definition:** `params_new = params - lr * gradients`

**Parameter:**
- `params`: Aktuelle Parameter
- `gradients`: Berechnete Gradienten
- `lr`: Lernrate

**Rückgabe:** Aktualisierte Parameter

**Beispiel:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Führt Adam-Optimizer-Schritt aus.

**Parameter:**
- `params`: Aktuelle Parameter
- `gradients`: Berechnete Gradienten
- `m`: Erste Momentenschätzung
- `v`: Zweite Momentenschätzung
- `lr`: Lernrate
- `beta1`: Erste Moment-Zerfallsrate (Standard: 0.9)
- `beta2`: Zweite Moment-Zerfallsrate (Standard: 0.999)
- `eps`: Numerische Stabilitätskonstante (Standard: 1e-8)

**Rückgabe:** Aktualisierte Parameter

**Beispiel:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Weitere Operationen

### `tokenize(text: String) -> TokenIDs`

Konvertiert Text in Token-ID-Sequenz.

**Parameter:**
- `text`: Eingabetextzeichenfolge

**Rückgabe:** TokenIDs (Vec<u32>)

**Beispiel:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Anwendungsfälle:**
- Text-Vorverarbeitung
- Sprachmodell-Eingabe
- NLP-Pipelines

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Broadcastet Tensor auf angegebene Form.

**Parameter:**
- `tensor`: Eingabe-Tensor
- `shape`: Zielform

**Rückgabe:** Gebroadcasteter Tensor

**Beispiel:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Anwendungsfälle:**
- Formausrichtung
- Batch-Operationen
- Elementweise Operationen mit unterschiedlichen Formen

---

## Operatoren

TensorLogic unterstützt standardmäßige mathematische Operatoren:

### Arithmetische Operatoren
- `+` : Addition
- `-` : Subtraktion
- `*` : Elementweise Multiplikation
- `/` : Elementweise Division

**Beispiel:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Vergleichsoperatoren
- `==` : Gleich
- `!=` : Ungleich
- `<`  : Kleiner als
- `<=` : Kleiner oder gleich
- `>`  : Größer als
- `>=` : Größer oder gleich

### Logische Operatoren
- `&&` : Logisches UND
- `||` : Logisches ODER
- `!`  : Logisches NICHT

---

## Typdefinitionen

### Tensor
Mehrdimensionales Array mit GPU-Beschleunigung über Metal Performance Shaders.

**Eigenschaften:**
- Form: Array von Dimensionen
- Daten: Float32-Elemente
- Gerät: Metal-GPU-Gerät

### TokenIDs
Spezieller Typ für Token-ID-Sequenzen.

**Definition:** `Vec<u32>`

**Anwendungsfälle:**
- Tokenisierungsergebnisse
- Einbettungs-Lookups
- Sequenzverarbeitung

### Number
Numerische Werte (Int oder Float).

**Varianten:**
- `Integer`: 64-Bit-Ganzzahl mit Vorzeichen
- `Float`: 64-Bit-Gleitkommazahl

---

## Referenzen

- **TensorLogic-Paper**: arXiv:2510.12269
- **Transformer-Architektur**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Verwandte Dokumentation

- [Erste Schritte](./getting_started.md)
- [Sprachreferenz](./language_reference.md)
- [Beispiele](../../examples/)
- [2025 hinzugefügte Operationen](../added_operations_2025.md)
- [TODO-Liste](../TODO.md)

---

**Zuletzt aktualisiert:** 2025-01-22

**TensorLogic-Version:** 0.1.1+

**Gesamtoperationen:** 48 Funktionen + 4 Operatoren
