# TensorLogic Sprachreferenz

**Version**: 0.2.0-alpha
**Letzte Aktualisierung**: 2025-10-20

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

TensorLogic ist eine Programmiersprache, die Tensoralgebra mit logischer Programmierung vereint und neuro-symbolische KI ermöglicht.

### Hauptmerkmale

- **Tensoroperationen**: GPU-beschleunigte Hochleistungsberechnungen
- **Automatische Differentiation**: Eingebaute Gradientenberechnung
- **Lernsystem**: Gradientenabstieg mit mehreren Optimierern
- **Logische Programmierung**: Relationen, Regeln und Abfragen
- **Neuro-Symbolische Integration**: Einbettungen für Entitäten und Relationen

---

## 2. Programmstruktur

```tensorlogic
// Deklarationen
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Hauptausführungsblock
main {
    result := w * w

    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

---

## 3. Datentypen

| Typ | Beschreibung | Genauigkeit |
|------|-------------|-----------|
| `float32` | 32-Bit-Fließkomma | Einfache Genauigkeit |
| `float64` | 64-Bit-Fließkomma | Doppelte Genauigkeit |
| `int32` | 32-Bit-Ganzzahl | Vorzeichenbehaftet |
| `int64` | 64-Bit-Ganzzahl | Lange Ganzzahl |
| `bool` | Boolean | wahr/falsch |

---

## 7. Operatoren

| Operator | Name | Beispiel |
|----------|------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraktion | `a - b` |
| `*` | Multiplikation | `a * b` |
| `/` | Division | `a / b` |
| `@` | Matrixmultiplikation | `A @ B` |
| `**` | Potenz | `a ** 2` |

---

## 9. Lernsystem

### Lernbare Parameter

```tensorlogic
tensor w: float32[10] learnable = [...]
```

### Optimierer

```tensorlogic
optimizer: sgd(lr: 0.1)      // SGD
optimizer: adam(lr: 0.001)   // Adam
optimizer: adamw(lr: 0.001)  // AdamW
```

---

**Ende der Sprachreferenz**

Für Fragen oder Beiträge besuchen Sie: https://github.com/JunSuzukiJapan/tensorlogic
