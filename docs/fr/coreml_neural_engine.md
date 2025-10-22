# Guide d'Intégration CoreML & Neural Engine

Ce guide explique comment utiliser des modèles CoreML dans TensorLogic et effectuer des inférences à haute vitesse sur l'Apple Neural Engine.

## À propos de CoreML et Neural Engine

### CoreML

- Framework d'apprentissage automatique propriétaire d'Apple
- Optimisé exclusivement pour iOS/macOS
- Exploite automatiquement Neural Engine, GPU et CPU
- Format .mlmodel / .mlmodelc

### Neural Engine

- Puce dédiée à l'IA exclusive à Apple Silicon
- Jusqu'à 15,8 TOPS (M1 Pro/Max)
- Consommation d'énergie ultra-faible (1/10 ou moins par rapport au GPU)
- Optimisé pour les opérations f16

### Intégration avec TensorLogic

- Toutes les opérations f16 (optimisé pour Neural Engine)
- Intégration transparente avec Metal GPU
- Détection automatique du format de modèle

## Créer des Modèles CoreML

Les modèles CoreML sont généralement créés avec coreMLtools de Python :

```python
import coremltools as ct
import torch

# Créer un modèle PyTorch
model = MyModel()
model.eval()

# Créer un modèle tracé
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convertir en CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Optimisation Neural Engine
    compute_precision=ct.precision.FLOAT16  # précision f16
)

# Sauvegarder
mlmodel.save("model.mlpackage")
```

## Utiliser dans TensorLogic

### 1. Charger un Modèle CoreML (macOS uniquement)

```tensorlogic
model = load_model("model.mlpackage")
// ou
model = load_model("model.mlmodelc")
```

### 2. Vérifier les Métadonnées

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Meilleures Pratiques d'Optimisation Neural Engine

### 1. Type de Données : Utiliser f16

✅ Recommandé : `compute_precision=ct.precision.FLOAT16`
❌ Non recommandé : FLOAT32 (exécuté sur GPU)

### 2. Format de Modèle : Utiliser le format mlprogram

✅ Recommandé : `convert_to="mlprogram"`
❌ Non recommandé : `convert_to="neuralnetwork"` (format ancien)

### 3. Taille de Lot : 1 est optimal

✅ Recommandé : `batch_size=1`
⚠️ Note : `batch_size>1` peut s'exécuter sur GPU

### 4. Taille d'Entrée : Taille fixe est optimale

✅ Recommandé : `shape=[1, 3, 224, 224]`
⚠️ Note : Les tailles variables ont une optimisation limitée

## Opérations Supportées

### Opérations exécutées rapidement sur Neural Engine

- ✅ Convolutions (conv2d, depthwise_conv)
- ✅ Couches entièrement connectées (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalisation (batch_norm, layer_norm)
- ✅ Fonctions d'activation (relu, gelu, sigmoid, tanh)
- ✅ Opérations élément par élément (add, mul, sub, div)

## Comparaison de Performance

Inférence ResNet-50 (image 224x224) :

| Appareil           | Latence  | Puissance | Efficacité |
|-------------------|----------|-----------|------------|
| Neural Engine     | ~3ms     | ~0,5W     | Maximale   |
| Metal GPU (M1)    | ~8ms     | ~5W       | Moyenne    |
| CPU (M1)          | ~50ms    | ~2W       | Faible     |

## Sélection du Format de Modèle

### Formats Recommandés par Cas d'Usage

**Entraînement** : SafeTensors
- Compatible PyTorch
- Sauvegarde/chargement des poids
- Entraîner sur Metal GPU

**Inférence (iOS/macOS)** : CoreML
- Optimisation Neural Engine
- Consommation d'énergie ultra-faible
- Intégration d'applications

**Inférence (Général)** : GGUF
- Support de quantification
- Multi-plateforme
- Efficace en mémoire

## Références

- [Documentation Officielle CoreML](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Guide Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Guide de Chargement de Modèles](model_loading.md)
