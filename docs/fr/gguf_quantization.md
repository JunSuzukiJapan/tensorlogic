# Guide des Modèles Quantifiés GGUF

Ce document explique comment charger et utiliser des modèles quantifiés au format GGUF (compatible llama.cpp) dans TensorLogic.

## À propos du Format GGUF

GGUF (GGML Universal Format) est un format de quantification efficace pour les grands modèles de langage développé par le projet llama.cpp.

### Caractéristiques Principales

- Efficacité mémoire grâce à la quantification 4-bit/8-bit (jusqu'à 8x de compression)
- La quantification par blocs maintient la précision
- Compatible avec llama.cpp, Ollama, LM Studio et plus

### Formats de Quantification Supportés par TensorLogic

- ✅ **Q4_0** : Quantification 4-bit (compression maximale)
- ✅ **Q8_0** : Quantification 8-bit (précision et compression équilibrées)
- ✅ **F16** : Virgule flottante 16-bit (haute précision)
- ✅ **F32** : Virgule flottante 32-bit (précision maximale)

## Utilisation de Base

### 1. Charger un Modèle Quantifié

Automatiquement déquantifié en f16 et chargé sur le GPU Metal :

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Obtenir des Tenseurs du Modèle

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Choisir le Format de Quantification

### Q4_0 (4-bit)

- **Mémoire** : Utilisation minimale (~1/8 du modèle original)
- **Vitesse** : Inférence la plus rapide
- **Précision** : Légère dégradation (généralement acceptable)
- **Cas d'Usage** : Chatbots, génération de texte générale

### Q8_0 (8-bit)

- **Mémoire** : Utilisation modérée (~1/4 du modèle original)
- **Vitesse** : Rapide
- **Précision** : Élevée (presque équivalente à F16)
- **Cas d'Usage** : Génération de haute qualité, assistants de codage

### F16 (16-bit)

- **Mémoire** : ~1/2 du modèle original
- **Vitesse** : Standard
- **Précision** : Format natif de TensorLogic, optimisé GPU Metal
- **Cas d'Usage** : Lorsque la qualité maximale est requise

## Exemple Pratique : Embeddings de Tokens

```tensorlogic
// Obtenir les embeddings de tokens du modèle LLama
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Obtenir le vecteur d'embedding depuis l'ID de token
function get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Économies de Mémoire par Quantification

Exemple : Modèle LLama-7B (7 milliards de paramètres) :

| Format     | Utilisation Mémoire | Compression |
|------------|---------------------|-------------|
| F32 (orig) | ~28 GB              | 1x          |
| F16        | ~14 GB              | 2x          |
| Q8_0       | ~7 GB               | 4x          |
| Q4_0       | ~3.5 GB             | 8x          |

TensorLogic convertit tous les formats en f16 au chargement et les exécute efficacement sur le GPU Metal.

## Télécharger et Installer des Modèles

### 1. Télécharger des Modèles GGUF depuis HuggingFace

Exemple : https://huggingface.co/TheBloke

### 2. Modèles Recommandés (pour débutants)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0 : ~600MB)
- **Phi-2** (Q4_0 : ~1.6GB)
- **Mistral-7B** (Q4_0 : ~3.8GB)

### 3. Charger dans TensorLogic

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Notes Importantes

- Les modèles quantifiés sont en lecture seule (ne peuvent pas être sauvegardés depuis TensorLogic)
- Utilisez des modèles non quantifiés (F16/F32) pour l'entraînement
- Q4/Q8 sont optimisés uniquement pour l'inférence
- Tous les formats de quantification sont automatiquement déquantifiés en f16 et chargés sur GPU

## Références

- [Spécification GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Modèles GGUF HuggingFace](https://huggingface.co/TheBloke)
- [Guide de Chargement de Modèles](model_loading.md)
