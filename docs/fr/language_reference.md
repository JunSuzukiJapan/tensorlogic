# Référence du Langage TensorLogic

**Version**: 0.2.0-alpha
**Dernière mise à jour**: 2025-10-20

## Table des Matières

1. [Introduction](#introduction)
2. [Structure du Programme](#structure-du-programme)
3. [Types de Données](#types-de-données)
4. [Déclarations](#déclarations)
5. [Expressions](#expressions)
6. [Instructions](#instructions)
7. [Opérateurs](#opérateurs)
8. [Fonctions Intégrées](#fonctions-intégrées)
9. [Système d'Apprentissage](#système-dapprentissage)
10. [Programmation Logique](#programmation-logique)

---

## 1. Introduction

TensorLogic est un langage de programmation qui unifie l'algèbre tensorielle avec la programmation logique, permettant l'IA neuro-symbolique. Il combine les opérations tensorielles différentiables avec le raisonnement logique pour les systèmes d'IA de nouvelle génération.

### Caractéristiques Principales

- **Opérations Tensorielles**: Calculs accélérés par GPU haute performance
- **Différentiation Automatique**: Calcul de gradient intégré
- **Système d'Apprentissage**: Descente de gradient avec plusieurs optimiseurs
- **Programmation Logique**: Relations, règles et requêtes
- **Intégration Neuro-Symbolique**: Embeddings pour entités et relations

---

## 2. Structure du Programme

### 2.1 Structure de Base

```tensorlogic
// Déclarations
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Bloc d'exécution principal
main {
    // Instructions
    result := w * w

    // Apprentissage
    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

---

## 3. Types de Données

### 3.1 Types de Base

| Type | Description | Précision |
|------|-------------|-----------|
| `float32` | Flottant 32 bits | Simple précision |
| `float64` | Flottant 64 bits | Double précision |
| `int32` | Entier 32 bits | Entier signé |
| `int64` | Entier 64 bits | Long entier signé |
| `bool` | Booléen | vrai/faux |
| `complex64` | Nombre complexe 64 bits | Complexe float32 |

---

## 7. Opérateurs

### 7.1 Opérateurs Arithmétiques

| Opérateur | Nom | Description | Exemple |
|----------|------|-------------|---------|
| `+` | Addition | Addition élément par élément | `a + b` |
| `-` | Soustraction | Soustraction élément par élément | `a - b` |
| `*` | Multiplication | Multiplication élément par élément (Hadamard) | `a * b` |
| `/` | Division | Division élément par élément | `a / b` |
| `@` | Multiplication Matricielle | Contraction tensorielle | `A @ B` |
| `**` | Puissance | Puissance élément par élément | `a ** 2` |

---

## 9. Système d'Apprentissage

### 9.1 Paramètres Apprenables

```tensorlogic
tensor w: float32[10] learnable = [...]
tensor b: float32[1] learnable = [0.0]
```

### 9.2 Optimiseurs

#### SGD (Descente de Gradient Stochastique)

```tensorlogic
optimizer: sgd(lr: 0.1)
```

**Paramètres**:
- `lr`: Taux d'apprentissage (par défaut: 0.01)

#### Adam

```tensorlogic
optimizer: adam(lr: 0.001)
```

**Paramètres**:
- `lr`: Taux d'apprentissage (par défaut: 0.001)
- `beta1`: Décroissance du premier moment (par défaut: 0.9)
- `beta2`: Décroissance du second moment (par défaut: 0.999)
- `epsilon`: Petite constante (par défaut: 1e-8)

---

**Fin de la Référence du Langage**

Pour questions ou contributions, visitez: https://github.com/JunSuzukiJapan/tensorlogic
