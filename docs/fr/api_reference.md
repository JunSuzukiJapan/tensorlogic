# Référence API de TensorLogic

Référence API complète pour toutes les opérations disponibles dans TensorLogic.

## Table des Matières

1. [Création de Tenseurs](#création-de-tenseurs)
2. [Opérations de Forme](#opérations-de-forme)
3. [Fonctions Mathématiques](#fonctions-mathématiques)
4. [Opérations d'Agrégation](#opérations-dagrégation)
5. [Fonctions d'Activation](#fonctions-dactivation)
6. [Opérations Matricielles](#opérations-matricielles)
7. [Normalisation](#normalisation)
8. [Opérations de Masquage](#opérations-de-masquage)
9. [Opérations d'Indexation](#opérations-dindexation)
10. [Embeddings](#embeddings)
11. [Échantillonnage](#échantillonnage)
12. [Opérations Fusionnées](#opérations-fusionnées)
13. [Optimisation](#optimisation)
14. [Autres Opérations](#autres-opérations)
15. [Opérateurs](#opérateurs)
16. [Définitions de Types](#définitions-de-types)

---

## Création de Tenseurs

### `zeros(shape: Array<Int>) -> Tensor`

Crée un tenseur rempli de zéros.

**Paramètres:**
- `shape`: Tableau spécifiant les dimensions du tenseur

**Retourne:** Tenseur rempli de 0

**Exemple:**
```tensorlogic
let z = zeros([2, 3])  // Tenseur 2x3 de zéros
```

---

### `ones(shape: Array<Int>) -> Tensor`

Crée un tenseur rempli de uns.

**Paramètres:**
- `shape`: Tableau spécifiant les dimensions du tenseur

**Retourne:** Tenseur rempli de 1

**Exemple:**
```tensorlogic
let o = ones([2, 3])  // Tenseur 2x3 de uns
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Génère un encodage positionnel sinusoïdal pour les Transformers.

**Paramètres:**
- `seq_len`: Longueur de la séquence
- `d_model`: Dimension du modèle

**Retourne:** Tenseur de forme `[seq_len, d_model]`

**Définition Mathématique:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Exemple:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**Cas d'Usage:**
- Modèles Transformer
- Modèles séquence-à-séquence
- Mécanismes d'attention

**Références:**
- arXiv:2510.12269 (Tableau 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## Opérations de Forme

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

Change la forme du tenseur tout en préservant les données.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `new_shape`: Forme cible

**Retourne:** Tenseur reformé

**Exemple:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**Contraintes:**
- Le nombre total d'éléments doit rester identique

---

### `flatten(tensor: Tensor) -> Tensor`

Aplatit le tenseur en 1D.

**Paramètres:**
- `tensor`: Tenseur d'entrée

**Retourne:** Tenseur 1D

**Exemple:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

Transpose un tenseur 2D (échange les axes).

**Paramètres:**
- `tensor`: Tenseur 2D d'entrée

**Retourne:** Tenseur transposé

**Exemple:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

Réordonne les dimensions du tenseur.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dims`: Nouvel ordre des dimensions

**Retourne:** Tenseur permuté

**Exemple:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

Ajoute une dimension de taille 1 à la position spécifiée.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Position pour insérer la nouvelle dimension

**Retourne:** Tenseur avec dimension ajoutée

**Exemple:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

Supprime toutes les dimensions de taille 1.

**Paramètres:**
- `tensor`: Tenseur d'entrée

**Retourne:** Tenseur avec dimensions de taille 1 supprimées

**Exemple:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

Divise le tenseur en plusieurs tenseurs le long de la dimension spécifiée.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `sizes`: Taille de chaque section divisée
- `dim`: Dimension pour diviser

**Retourne:** Tableau de tenseurs

**Exemple:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3 tenseurs: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

Divise le tenseur en nombre spécifié de morceaux.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `chunks`: Nombre de morceaux
- `dim`: Dimension pour diviser

**Retourne:** Tableau de tenseurs

**Exemple:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 3 tenseurs de [4,4] chacun
```

---

## Fonctions Mathématiques

### `exp(tensor: Tensor) -> Tensor`

Applique la fonction exponentielle élément par élément.

**Définition Mathématique:** `exp(x) = e^x`

**Exemple:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

Applique le logarithme naturel élément par élément.

**Définition Mathématique:** `log(x) = ln(x)`

**Exemple:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

Applique la racine carrée élément par élément.

**Définition Mathématique:** `sqrt(x) = √x`

**Exemple:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

Élève les éléments du tenseur à la puissance spécifiée.

**Définition Mathématique:** `pow(x, n) = x^n`

**Exemple:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

Applique la fonction sinus élément par élément.

**Exemple:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

Applique la fonction cosinus élément par élément.

**Exemple:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

Applique la fonction tangente élément par élément.

**Exemple:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## Opérations d'Agrégation

### `sum(tensor: Tensor) -> Number`

Calcule la somme de tous les éléments.

**Exemple:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

Calcule la moyenne de tous les éléments.

**Exemple:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

Retourne la valeur maximale dans le tenseur.

**Exemple:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

Retourne la valeur minimale dans le tenseur.

**Exemple:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

Retourne les indices des valeurs maximales le long de la dimension spécifiée.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Dimension pour trouver le maximum

**Retourne:** Tenseur d'indices

**Exemple:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // Indices maximums le long de la dimension 1
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

Retourne les indices des valeurs minimales le long de la dimension spécifiée.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Dimension pour trouver le minimum

**Retourne:** Tenseur d'indices

**Exemple:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // Indices minimums le long de la dimension 1
```

---

## Fonctions d'Activation

### `relu(tensor: Tensor) -> Tensor`

Activation Rectified Linear Unit.

**Définition Mathématique:** `relu(x) = max(0, x)`

**Exemple:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

Fonction d'activation sigmoïde.

**Définition Mathématique:** `sigmoid(x) = 1 / (1 + e^(-x))`

**Exemple:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

Activation Gaussian Error Linear Unit (utilisée dans BERT, GPT).

**Définition Mathématique:** 
```
gelu(x) = x * Φ(x)
où Φ(x) est la fonction de répartition de la distribution normale standard
```

**Exemple:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**Cas d'Usage:**
- Modèles BERT, GPT
- Architectures Transformer modernes

---

### `tanh(tensor: Tensor) -> Tensor`

Activation tangente hyperbolique.

**Définition Mathématique:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Exemple:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

Applique la normalisation softmax le long de la dimension spécifiée.

**Définition Mathématique:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Dimension pour appliquer softmax

**Retourne:** Tenseur de distribution de probabilité

**Exemple:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**Cas d'Usage:**
- Mécanismes d'attention
- Couches de sortie de classification
- Distributions de probabilité

---

## Opérations Matricielles

### `matmul(a: Tensor, b: Tensor) -> Tensor`

Multiplication matricielle.

**Paramètres:**
- `a`: Matrice de gauche
- `b`: Matrice de droite

**Retourne:** Résultat de la multiplication matricielle

**Exemple:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## Normalisation

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

Applique la normalisation de couche.

**Définition Mathématique:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `normalized_shape`: Forme pour normaliser
- `eps`: Petite valeur pour la stabilité numérique (par défaut: 1e-5)

**Exemple:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**Cas d'Usage:**
- Couches Transformer
- Réseaux récurrents
- Réseaux de neurones profonds

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

Applique la normalisation par lots.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `running_mean`: Moyenne mobile
- `running_var`: Variance mobile
- `eps`: Petite valeur pour la stabilité numérique

**Exemple:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

Applique le dropout (met aléatoirement des éléments à zéro avec probabilité p).

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `p`: Probabilité de dropout (0.0 à 1.0)

**Retourne:** Tenseur avec éléments aléatoires mis à zéro

**Exemple:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**Cas d'Usage:**
- Régularisation d'entraînement
- Prévention du surapprentissage
- Apprentissage d'ensemble

---

## Opérations de Masquage

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

Applique un masque d'attention (met les positions masquées à -inf).

**Paramètres:**
- `tensor`: Tenseur de scores d'attention
- `mask`: Masque binaire (1 = garder, 0 = masquer)

**Retourne:** Tenseur masqué

**Exemple:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**Cas d'Usage:**
- Mécanismes d'attention Transformer
- Masquage de séquences
- Attention causale (autorégressive)

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

Crée un masque de remplissage pour les séquences de longueur variable.

**Paramètres:**
- `lengths`: Tableau des longueurs de séquences réelles
- `max_len`: Longueur maximale de séquence

**Retourne:** Tenseur de masque binaire

**Exemple:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// Résultat: [batch_size, max_len] où 1 = token réel, 0 = remplissage
```

**Cas d'Usage:**
- Gestion des séquences de longueur variable
- Traitement par lots
- Masquage d'attention

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

Combine deux masques en utilisant un ET logique.

**Paramètres:**
- `mask1`: Premier masque
- `mask2`: Deuxième masque

**Retourne:** Masque combiné

**Exemple:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**Cas d'Usage:**
- Combinaison de masques de remplissage et d'attention
- Masquage multi-contraintes
- Motifs d'attention complexes

---

## Opérations d'Indexation

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

Recueille les valeurs le long de la dimension en utilisant les indices.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Dimension pour recueillir
- `indices`: Tenseur d'indices

**Retourne:** Tenseur recueilli

**Exemple:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**Cas d'Usage:**
- Sélection de tokens
- Recherche par faisceau
- Indexation avancée

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

Sélectionne les éléments aux indices spécifiés le long de la dimension.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `dim`: Dimension pour sélectionner
- `indices`: Tableau d'indices

**Retourne:** Tenseur sélectionné

**Exemple:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // Sélectionner lignes 0, 2, 5
```

---

## Embeddings

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

Convertit les IDs de tokens en embeddings.

**Paramètres:**
- `indices`: Séquence d'IDs de tokens
- `vocab_size`: Taille du vocabulaire
- `embed_dim`: Dimension d'embedding

**Retourne:** Tenseur d'embedding de forme `[seq_len, embed_dim]`

**Exemple:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**Cas d'Usage:**
- Embeddings de mots
- Représentations de tokens
- Modèles de langage

**Références:**
- arXiv:2510.12269 (Tableau 1)

---

## Échantillonnage

### `top_k(logits: Tensor, k: Int) -> Tensor`

Échantillonne un token en utilisant l'échantillonnage top-k.

**Paramètres:**
- `logits`: Logits de sortie du modèle
- `k`: Nombre de tokens principaux à considérer

**Retourne:** ID de token échantillonné

**Exemple:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**Cas d'Usage:**
- Génération de texte
- Échantillonnage contrôlé
- Sorties diverses

**Références:**
- arXiv:2510.12269 (Tableau 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

Échantillonne un token en utilisant l'échantillonnage nucleus (top-p).

**Paramètres:**
- `logits`: Logits de sortie du modèle
- `p`: Seuil de probabilité cumulée (0.0 à 1.0)

**Retourne:** ID de token échantillonné

**Exemple:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**Cas d'Usage:**
- Génération de texte
- Sélection dynamique de vocabulaire
- Échantillonnage avec contrôle de qualité

**Références:**
- arXiv:2510.12269 (Tableau 2)

---

## Opérations Fusionnées

Les opérations fusionnées combinent plusieurs opérations pour de meilleures performances en réduisant la surcharge mémoire et les lancements de noyaux.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusionne l'addition et l'activation ReLU.

**Définition Mathématique:** `fused_add_relu(x, y) = relu(x + y)`

**Exemple:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**Performance:** ~1.5x plus rapide que les opérations séparées

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

Fusionne la multiplication et l'activation ReLU.

**Définition Mathématique:** `fused_mul_relu(x, y) = relu(x * y)`

**Exemple:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

Fusionne la transformation affine (échelle et décalage).

**Définition Mathématique:** `fused_affine(x, s, b) = x * s + b`

**Exemple:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**Cas d'Usage:**
- Normalisation par lots
- Normalisation de couche
- Transformations linéaires personnalisées

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

Fusionne l'activation GELU et la transformation linéaire.

**Définition Mathématique:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**Exemple:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**Cas d'Usage:**
- Couches FFN de Transformer
- Architectures BERT/GPT
- Chemins critiques de performance

---

## Optimisation

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

Effectue une étape d'optimisation SGD.

**Définition Mathématique:** `params_new = params - lr * gradients`

**Paramètres:**
- `params`: Paramètres actuels
- `gradients`: Gradients calculés
- `lr`: Taux d'apprentissage

**Retourne:** Paramètres mis à jour

**Exemple:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Effectue une étape d'optimisation Adam.

**Paramètres:**
- `params`: Paramètres actuels
- `gradients`: Gradients calculés
- `m`: Estimation du premier moment
- `v`: Estimation du second moment
- `lr`: Taux d'apprentissage
- `beta1`: Taux de décroissance du premier moment (par défaut: 0.9)
- `beta2`: Taux de décroissance du second moment (par défaut: 0.999)
- `eps`: Constante de stabilité numérique (par défaut: 1e-8)

**Retourne:** Paramètres mis à jour

**Exemple:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## Autres Opérations

### `tokenize(text: String) -> TokenIDs`

Convertit le texte en séquence d'IDs de tokens.

**Paramètres:**
- `text`: Chaîne de texte d'entrée

**Retourne:** TokenIDs (Vec<u32>)

**Exemple:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**Cas d'Usage:**
- Prétraitement de texte
- Entrée de modèle de langage
- Pipelines NLP

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

Diffuse le tenseur à la forme spécifiée.

**Paramètres:**
- `tensor`: Tenseur d'entrée
- `shape`: Forme cible

**Retourne:** Tenseur diffusé

**Exemple:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**Cas d'Usage:**
- Alignement de formes
- Opérations par lots
- Opérations élément par élément avec différentes formes

---

## Opérateurs

TensorLogic prend en charge les opérateurs mathématiques standards:

### Opérateurs Arithmétiques
- `+` : Addition
- `-` : Soustraction
- `*` : Multiplication élément par élément
- `/` : Division élément par élément

**Exemple:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### Opérateurs de Comparaison
- `==` : Égal
- `!=` : Différent
- `<`  : Inférieur à
- `<=` : Inférieur ou égal à
- `>`  : Supérieur à
- `>=` : Supérieur ou égal à

### Opérateurs Logiques
- `&&` : ET logique
- `||` : OU logique
- `!`  : NON logique

---

## Définitions de Types

### Tensor
Tableau multidimensionnel avec accélération GPU via Metal Performance Shaders.

**Propriétés:**
- Forme: Tableau de dimensions
- Données: Éléments Float32
- Appareil: Appareil GPU Metal

### TokenIDs
Type spécial pour les séquences d'IDs de tokens.

**Définition:** `Vec<u32>`

**Cas d'Usage:**
- Résultats de tokenisation
- Recherches d'embedding
- Traitement de séquences

### Number
Valeurs numériques (Int ou Float).

**Variantes:**
- `Integer`: Entier signé 64 bits
- `Float`: Virgule flottante 64 bits

---

## Références

- **Article TensorLogic**: arXiv:2510.12269
- **Architecture Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## Documentation Associée

- [Guide de Démarrage](./getting_started.md)
- [Référence du Langage](./language_reference.md)
- [Exemples](../../examples/)
- [Opérations Ajoutées en 2025](../added_operations_2025.md)
- [Liste des TODOs](../TODO.md)

---

**Dernière Mise à Jour:** 2025-01-22

**Version de TensorLogic:** 0.1.1+

**Opérations Totales:** 48 fonctions + 4 opérateurs
