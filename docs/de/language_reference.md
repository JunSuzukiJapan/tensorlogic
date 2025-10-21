# TensorLogic Sprachreferenz

**Version**: 0.1.1
**Letzte Aktualisierung**: 2025-10-21

## Inhaltsverzeichnis

1. [Einführung](#einführung)
2. [Programmstruktur](#programmstruktur)
3. [Datentypen](#datentypen)
4. [Deklarationen](#deklarationen)
5. [Ausdrücke](#ausdrücke)
6. [Anweisungen](#anweisungen)
7. [Operatoren](#operatoren)
8. [Eingebaute Funktionen](#eingebaute-funktionen)
9. [Lernsystem](#lernsystem)
10. [Logische Programmierung](#logische-programmierung)

---

## 1. Einführung

TensorLogic ist eine Programmiersprache, die Tensoralgebra mit logischer Programmierung vereint und neuro-symbolische KI ermöglicht. Sie kombiniert differenzierbare Tensoroperationen mit logischem Schließen für KI-Systeme der nächsten Generation.

### Hauptmerkmale

- **Tensoroperationen**: GPU-beschleunigte Hochleistungsberechnungen
- **Automatische Differentiation**: Eingebaute Gradientenberechnung
- **Lernsystem**: Gradientenabstieg mit mehreren Optimierern
- **Logische Programmierung**: Relationen, Regeln und Abfragen
- **Neuro-Symbolische Integration**: Einbettungen für Entitäten und Relationen

---

## 2. Programmstruktur

### 2.1 Grundstruktur

```tensorlogic
// Deklarationen
tensor w: float16[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Hauptausführungsblock
main {
    // Anweisungen
    result := w * w

    // Lernen
    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

### 2.2 Externe Dateien importieren

TensorLogic unterstützt das Importieren von Deklarationen aus externen Dateien:

```tensorlogic
// Deklarationen aus einer anderen Datei importieren
import "path/to/module.tl"
import "../lib/constants.tl"

main {
    // Importierte Tensoren und Funktionen verwenden
    result := imported_tensor * 2
}
```

**Funktionen**:
- Relative Pfadauflösung (relativ zur importierenden Datei)
- Erkennung zirkulärer Abhängigkeiten (verhindert Endlosschleifen)
- Verhinderung doppelter Importe (dieselbe Datei wird nicht zweimal importiert)
- Nur Deklarationen werden importiert (Hauptblöcke werden nicht ausgeführt)

**Beispiel**:

Datei: `lib/constants.tl`
```tensorlogic
tensor pi: float16[1] = [3.14159]
tensor e: float16[1] = [2.71828]
```

Datei: `main.tl`
```tensorlogic
import "lib/constants.tl"

main {
    tensor circumference: float16[1] = [2.0]
    result := circumference * pi  // Verwendet importiertes pi
    print("Ergebnis:", result)
}
```

### 2.3 Kommentare

```tensorlogic
// Einzeiliger Kommentar

/* Mehrzeiliger Kommentar
   Noch nicht implementiert */
```

---

## 3. Datentypen

### 3.1 Basistypen

| Typ | Beschreibung | Genauigkeit |
|------|-------------|-----------|
| `float16` | 16-Bit-Fließkomma (f16) | Halbe Genauigkeit (Apple Silicon optimiert) |
| `float32` | 32-Bit-Fließkomma | Einfache Genauigkeit |
| `float64` | 64-Bit-Fließkomma | Doppelte Genauigkeit |
| `int32` | 32-Bit-Ganzzahl | Vorzeichenbehaftet |
| `int64` | 64-Bit-Ganzzahl | Lange Ganzzahl |
| `bool` | Boolean | true/false |
| `complex64` | 64-Bit-Komplexzahl | Komplex float32 |

**Hinweis**: TensorLogic verwendet hauptsächlich `float16` für optimale Leistung auf Apple Silicon (Metal GPU und Neural Engine).

### 3.1.1 Numerische Literale

TensorLogic unterstützt positive und negative numerische Literale:

```tensorlogic
tensor positive: float16[1] = [3.14]
tensor negative: float16[1] = [-2.71]
tensor zero: float16[1] = [0.0]
tensor neg_int: float16[1] = [-42.0]
```

### 3.2 Tensor-Typen

```tensorlogic
tensor x: float16[10]           // 1D-Tensor
tensor W: float16[3, 4]         // 2D-Matrix
tensor T: float16[2, 3, 4]      // 3D-Tensor
```

---

## 7. Operatoren

### 7.1 Arithmetische Operatoren

| Operator | Name | Beispiel |
|----------|------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraktion | `a - b` |
| `*` | Multiplikation (elementweise) | `a * b` |
| `/` | Division | `a / b` |
| `@` | Matrixmultiplikation | `A @ B` |
| `**` | Potenz | `a ** 2` |

### 7.2 Vergleichsoperatoren

| Operator | Bedeutung |
|----------|-----------|
| `==` | Gleich |
| `!=` | Ungleich |
| `<` | Kleiner als |
| `>` | Größer als |
| `<=` | Kleiner oder gleich |
| `>=` | Größer oder gleich |

---

## 8. Eingebaute Funktionen

### 8.1 Aktivierungsfunktionen

```tensorlogic
relu(x)      // Rectified Linear Unit
gelu(x)      // Gaussian Error Linear Unit
softmax(x)   // Softmax-Normalisierung
```

### 8.2 Reduktionsfunktionen

```tensorlogic
sum(tensor)   // Summe aller Elemente
mean(tensor)  // Mittelwert
max(tensor)   // Maximum
min(tensor)   // Minimum
```

---

## 9. Lernsystem

### 9.1 Lernbare Parameter

```tensorlogic
tensor w: float16[10] learnable = [...]
tensor b: float16[1] learnable = [0.0]
```

### 9.2 Optimierer

#### SGD (Stochastic Gradient Descent)

```tensorlogic
optimizer: sgd(lr: 0.1)
```

**Parameter**:
- `lr`: Lernrate (Standard: 0.01)

#### Adam

```tensorlogic
optimizer: adam(lr: 0.001)
```

**Parameter**:
- `lr`: Lernrate (Standard: 0.001)
- `beta1`: Erste Momentenabnahme (Standard: 0.9)
- `beta2`: Zweite Momentenabnahme (Standard: 0.999)
- `epsilon`: Kleine Konstante (Standard: 1e-8)

#### AdamW

```tensorlogic
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

**Parameter**:
- `lr`: Lernrate (Standard: 0.001)
- `weight_decay`: Gewichtsabnahmekoeffizient (Standard: 0.01)

### 9.3 Lernspezifikation

```tensorlogic
learn {
    // Optional: Lokale Variablendeklarationen für Zwischenberechnungen
    intermediate := some_expression
    another_var := other_expression

    objective: loss_expression,
    optimizer: optimizer_spec,
    epochs: number
}
```

**Anforderungen**:
- `objective` muss ein skalarer Tensorausdruck sein
- Alle mit `learnable` markierten Tensoren werden optimiert
- Gradienten werden durch automatische Differentiation berechnet
- Lokale Variablen (`:=`) können vor `objective` für Zwischenberechnungen verwendet werden
- Lokale Variablen werden bei jeder Epoche neu berechnet

**Beispiel - Grundlegendes Lernen**:

```tensorlogic
tensor w: float16[10] learnable = [...]
tensor x: float16[10] = [...]

main {
    pred := w * x
    loss := pred * pred

    learn {
        objective: loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }
}
```

**Beispiel - Mit lokalen Variablen**:

```tensorlogic
tensor W: float16[1] learnable = [0.5]
tensor x1: float16[1] = [1.0]
tensor y1: float16[1] = [3.0]
tensor x2: float16[1] = [-2.0]  // Negative Zahlen werden unterstützt
tensor y2: float16[1] = [-6.0]

main {
    learn {
        // Lokale Variablen für Zwischenberechnungen
        pred1 := x1 * W
        pred2 := x2 * W

        // Fehler berechnen
        err1 := pred1 - y1
        err2 := pred2 - y2

        // Summe der quadratischen Fehler
        total_loss := err1 * err1 + err2 * err2

        objective: total_loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }

    print("Gelerntes W:", W)  // Sollte nahe 3.0 sein
}
```

**Hinweis**: Nur explizit mit dem Schlüsselwort `learnable` deklarierte Tensoren werden optimiert. Lokale Variablen, die innerhalb des `learn`-Blocks berechnet werden, werden nicht als lernbare Parameter behandelt.

---

## 10. Logische Programmierung

### 10.1 Relationen

```tensorlogic
relation Parent(x: entity, y: entity)
relation Sibling(x: entity, y: entity)
```

### 10.2 Regeln (Zukünftig)

```tensorlogic
rule Grandparent(X, Z) <- Parent(X, Y), Parent(Y, Z)
rule Ancestor(X, Y) <- Parent(X, Y)
rule Ancestor(X, Z) <- Parent(X, Y), Ancestor(Y, Z)
```

### 10.3 Abfragen (Teilweise)

```tensorlogic
query Parent(alice, X)
query Parent(X, Y) where X != Y
```

---

**Ende der Sprachreferenz**

Für Fragen oder Beiträge besuchen Sie: https://github.com/JunSuzukiJapan/tensorlogic
