# Riferimento Linguaggio TensorLogic

**Versione**: 0.2.0-alpha
**Ultimo aggiornamento**: 2025-10-20

## Sommario

1. [Introduzione](#introduzione)
2. [Struttura del Programma](#struttura-del-programma)
3. [Tipi di Dati](#tipi-di-dati)
4. [Dichiarazioni](#dichiarazioni)
5. [Espressioni](#espressioni)
6. [Istruzioni](#istruzioni)
7. [Operatori](#operatori)
8. [Funzioni Integrate](#funzioni-integrate)
9. [Sistema di Apprendimento](#sistema-di-apprendimento)
10. [Programmazione Logica](#programmazione-logica)

---

## 1. Introduzione

TensorLogic è un linguaggio di programmazione che unifica l'algebra tensoriale con la programmazione logica, abilitando l'IA neuro-simbolica.

### Caratteristiche Principali

- **Operazioni Tensoriali**: Calcoli ad alte prestazioni accelerati da GPU
- **Differenziazione Automatica**: Calcolo del gradiente integrato
- **Sistema di Apprendimento**: Discesa del gradiente con multipli ottimizzatori
- **Programmazione Logica**: Relazioni, regole e query
- **Integrazione Neuro-Simbolica**: Embedding per entità e relazioni

---

## 2. Struttura del Programma

```tensorlogic
// Dichiarazioni
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Blocco di esecuzione principale
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

## 3. Tipi di Dati

| Tipo | Descrizione | Precisione |
|------|-------------|-----------|
| `float32` | Virgola mobile 32-bit | Singola precisione |
| `float64` | Virgola mobile 64-bit | Doppia precisione |
| `int32` | Intero 32-bit | Intero con segno |
| `int64` | Intero 64-bit | Intero lungo con segno |
| `bool` | Booleano | vero/falso |

---

## 7. Operatori

| Operatore | Nome | Esempio |
|----------|------|---------|
| `+` | Addizione | `a + b` |
| `-` | Sottrazione | `a - b` |
| `*` | Moltiplicazione | `a * b` |
| `/` | Divisione | `a / b` |
| `@` | Moltiplicazione Matriciale | `A @ B` |
| `**` | Potenza | `a ** 2` |

---

**Fine del Riferimento Linguaggio**

Per domande o contributi, visita: https://github.com/JunSuzukiJapan/tensorlogic
