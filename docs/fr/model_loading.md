# Guide de Chargement de Modèles

Ce document explique comment charger et utiliser des modèles PyTorch et HuggingFace dans TensorLogic. Il prend en charge le format SafeTensors (compatible PyTorch) et le format GGUF (LLMs quantifiés).

## Utilisation de Base

### 1. Charger un Modèle SafeTensors (sauvegardé depuis PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Charger un Modèle GGUF (LLM quantifié)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Obtenir des Tenseurs du Modèle

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Exemple Pratique : Inférence de Couche Linéaire

Effectuer une inférence en utilisant les poids et biais du modèle :

```tensorlogic
fn forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Transformation linéaire : output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Préparer des Modèles PyTorch

Enregistrez votre modèle au format SafeTensors avec Python :

```python
import torch
from safetensors.torch import save_file

# Créer un modèle PyTorch
model = MyModel()

# Obtenir les poids du modèle sous forme de dictionnaire
tensors = {name: param for name, param in model.named_parameters()}

# Enregistrer au format SafeTensors
save_file(tensors, "model.safetensors")
```

Puis charger dans TensorLogic :

```tensorlogic
model = load_model("model.safetensors")
```

## Formats Pris en Charge

### 1. SafeTensors (.safetensors)

- Compatible avec PyTorch et HuggingFace
- Prend en charge les types de données F32, F64, F16, BF16
- Toutes les données sont automatiquement converties en f16
- Chargé directement sur le GPU Metal

### 2. GGUF (.gguf)

- Modèles quantifiés au format llama.cpp
- Prend en charge Q4_0, Q8_0, F32, F16
- Chargé directement sur le GPU Metal

### 3. CoreML (.mlmodel, .mlpackage)

- Modèles optimisés pour Apple Neural Engine
- iOS/macOS uniquement

## Exemple Complet de Modèle Linéaire

```tensorlogic
// Données d'entrée (taille de lot 4, dimension de caractéristiques 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Matrice de poids (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Biais (2 dimensions)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Exécuter l'inférence
let output = forward(X, W, b)

// Afficher les résultats
print("Output shape:", output.shape)
print("Output:", output)
```

## Sauvegarder des Modèles

Vous pouvez sauvegarder des modèles TensorLogic au format SafeTensors :

```tensorlogic
save_model(model, "output.safetensors")
```

Cela permet l'interopérabilité avec PyTorch et HuggingFace.

## Notes Importantes

- TensorLogic exécute toutes les opérations en f16 (optimisé GPU Metal)
- Les autres types de données sont automatiquement convertis en f16 lors du chargement
- Les types entiers (i8, i32, etc.) ne sont pas pris en charge (virgule flottante uniquement)
- Les grands modèles sont automatiquement chargés dans la mémoire GPU Metal

## Documentation Connexe

- [Modèles Quantifiés GGUF](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Guide de Démarrage](../claudedocs/getting_started.md)
