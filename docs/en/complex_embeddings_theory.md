# ComplEx: Complex Embeddings for Knowledge Graphs

## Overview

ComplEx (Complex Embeddings) is a knowledge graph embedding method that uses complex-valued vectors to represent entities and relations. It was introduced in the paper "Complex Embeddings for Simple Link Prediction" (Trouillon et al., 2016).

## Motivation

**Problem with Real-Valued Embeddings:**
- TransE: Good for antisymmetric relations, struggles with symmetric
- DistMult: Good for symmetric relations, cannot model antisymmetric

**ComplEx Solution:**
- Use complex numbers to unify both symmetric and antisymmetric relations
- Complex conjugate operation provides asymmetry
- Generalizes DistMult while adding expressiveness

## Mathematical Formulation

### 1. Representation

Entities and relations are represented as complex vectors:

```
h, r, t ∈ ℂ^d
```

Where each element is a complex number:
```
h_i = h_re[i] + j·h_im[i]
```

In implementation, we split into real and imaginary parts:
```
h = (h_re, h_im)  where h_re, h_im ∈ ℝ^d
```

### 2. Score Function

The score of a triple (h, r, t) is:

```
φ(h, r, t) = Re(⟨h, r, t̄⟩)
           = Re(Σ_i h_i · r_i · conj(t_i))
```

Where:
- `conj(t)` is the complex conjugate of t
- `Re()` extracts the real part
- `⟨·,·,·⟩` is the trilinear dot product

### 3. Complex Conjugate

For a complex number `z = a + bi`:
```
conj(z) = a - bi
```

This operation makes ComplEx asymmetric, unlike DistMult.

### 4. Expanded Score Formula

Expanding the complex multiplication:

```
h_i · r_i · conj(t_i)
= (h_re[i] + j·h_im[i]) · (r_re[i] + j·r_im[i]) · (t_re[i] - j·t_im[i])
```

After algebraic expansion, the real part is:

```
Re(h_i · r_i · conj(t_i)) =
    h_re[i] · r_re[i] · t_re[i]
  + h_re[i] · r_im[i] · t_im[i]
  + h_im[i] · r_re[i] · t_im[i]
  - h_im[i] · r_im[i] · t_re[i]
```

Final score:
```
φ(h, r, t) = Σ_i [
    h_re[i] · r_re[i] · t_re[i]
  + h_re[i] · r_im[i] · t_im[i]
  + h_im[i] · r_re[i] · t_im[i]
  - h_im[i] · r_im[i] · t_re[i]
]
```

## Properties

### Symmetric Relations

If `r_im = 0` (purely real relation):
```
φ(h, r, t) = Σ_i h_re[i] · r_re[i] · t_re[i]
           = DistMult score
```

ComplEx generalizes DistMult!

### Antisymmetric Relations

Using complex conjugate:
```
φ(h, r, t) ≠ φ(t, r, h)  in general
```

This allows modeling of asymmetric relations like "parent_of".

### Composition

ComplEx can model compositional relations:
```
If r = r1 ○ r2, then ComplEx can learn this through r = r1 · r2
```

## Implementation Strategy

### Storage

Instead of native complex numbers, use paired real tensors:

```python
# Entity h represented as:
h_re: Tensor[d]  # Real part
h_im: Tensor[d]  # Imaginary part
```

### Computation

The score computation becomes:

```python
def complex_score(h_re, h_im, r_re, r_im, t_re, t_im):
    # Four trilinear products
    term1 = sum(h_re * r_re * t_re)
    term2 = sum(h_re * r_im * t_im)
    term3 = sum(h_im * r_re * t_im)
    term4 = sum(h_im * r_im * t_re)

    return term1 + term2 + term3 - term4
```

## Training

### Loss Functions

Same as TransE and DistMult:

**Margin Ranking Loss:**
```
L = max(0, γ + φ(h', r, t') - φ(h, r, t))
```

**Binary Cross Entropy:**
```
L = -y·log(σ(φ)) - (1-y)·log(1-σ(φ))
```

### Regularization

L2 regularization on embeddings prevents overfitting:
```
L_reg = λ(||h||² + ||r||² + ||t||²)
```

For complex embeddings:
```
||h||² = ||h_re||² + ||h_im||²
```

## Advantages

1. **Expressiveness**: Models symmetric and antisymmetric relations
2. **Generalization**: Subsumes DistMult as special case
3. **Composition**: Can learn relation composition
4. **Performance**: State-of-the-art on many benchmarks

## Disadvantages

1. **Parameters**: 2x parameters compared to real embeddings
2. **Computation**: More complex score calculation
3. **Interpretation**: Less intuitive than TransE

## Comparison

| Model     | Symmetric | Antisymmetric | Parameters |
|-----------|-----------|---------------|------------|
| TransE    | ❌        | ✅            | d          |
| DistMult  | ✅        | ❌            | d          |
| ComplEx   | ✅        | ✅            | 2d         |

## References

- Trouillon et al. "Complex Embeddings for Simple Link Prediction" (ICML 2016)
- https://arxiv.org/abs/1606.06357
