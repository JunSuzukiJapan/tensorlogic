# Référence du Langage TensorLogic

**Version**: 0.1.1
**Dernière mise à jour**: 2025-10-21

## Table des matières

1. [Introduction](#introduction)
2. [Structure du programme](#structure-du-programme)
3. [Types de données](#types-de-données)
7. [Opérateurs](#opérateurs)
8. [Fonctions intégrées](#fonctions-intégrées)
9. [Système d'apprentissage](#système-dapprentissage)
10. [Programmation logique](#programmation-logique)

---

## 1. Introduction

TensorLogic est un langage de programmation qui unifie l'algèbre tensorielle avec la programmation logique, permettant l'IA neuro-symbolique.

### Caractéristiques principales

- **Opérations tensorielles**: Calculs haute performance accélérés par GPU
- **Différentiation automatique**: Calcul de gradient intégré
- **Système d'apprentissage**: Descente de gradient avec plusieurs optimiseurs (SGD, Adam, AdamW)
- **Variables locales**: Support des variables locales dans les blocs `learn` avec `:=`
- **Nombres négatifs**: Support complet des littéraux numériques négatifs
- **Importation de fichiers**: Import de déclarations depuis des fichiers externes

---

## 2. Structure du programme

### 2.1 Importer des fichiers externes

```tensorlogic
import "path/to/module.tl"

main {
    result := imported_tensor * 2
}
```

### 2.2 Structure de base

```tensorlogic
tensor w: float16[10] learnable = [...]

main {
    learn {
        // Variables locales pour calculs intermédiaires
        pred := x * w
        loss := pred * pred
        
        objective: loss,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

---

## 3. Types de données

| Type | Description |
|------|-------------|
| `float16` | 16 bits (optimisé Apple Silicon) |
| `float32` | 32 bits simple précision |
| `float64` | 64 bits double précision |
| `int32` | Entier 32 bits |
| `bool` | Booléen |

**Littéraux numériques**: `[3.14]`, `[-2.71]`, `[-42.0]`

---

## 7. Opérateurs

| Opérateur | Description |
|-----------|-------------|
| `+` `-` `*` `/` | Arithmétique |
| `@` | Multiplication matricielle |
| `**` | Puissance |
| `==` `!=` `<` `>` `<=` `>=` | Comparaison |

---

## 8. Fonctions intégrées

```tensorlogic
relu(x), gelu(x), softmax(x)      // Activations
sum(x), mean(x), max(x), min(x)   // Réductions
```

---

## 9. Système d'apprentissage

### 9.1 Optimiseurs

```tensorlogic
optimizer: sgd(lr: 0.1)
optimizer: adam(lr: 0.001)
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

### 9.2 Variables locales dans learn

```tensorlogic
tensor W: float16[1] learnable = [0.5]

main {
    learn {
        // Variables locales
        pred1 := x1 * W
        pred2 := x2 * W
        loss := (pred1 - y1) * (pred1 - y1) + (pred2 - y2) * (pred2 - y2)
        
        objective: loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }
}
```

**Note**: Seuls les tenseurs `learnable` sont optimisés, pas les variables locales.

---

Pour plus d'informations: https://github.com/JunSuzukiJapan/tensorlogic
