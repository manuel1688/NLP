# Plan: Reconocer Sentimiento

Comparar tres métodos de representación (BoW, TF-IDF, Word2Vec) para clasificación
binaria de sentimiento en reseñas en español.

---

## Paso 1 — Carga del corpus

- Descargar `amazon_reviews_multi` en español desde HuggingFace.
- Usar la estructura oficial de splits: `train`, `validation`, `test`.

```python
from datasets import load_dataset
ds = load_dataset("amazon_reviews_multi", "es")
```

---

## Paso 2 — Binarización de etiquetas

Convertir las estrellas (1–5) en sentimiento binario y descartar las neutras:

| Estrellas | Etiqueta |
|-----------|----------|
| 1 – 2     | negativo (`0`) |
| 3         | **descartar** |
| 4 – 5     | positivo (`1`) |

Descartar las reseñas de 3 estrellas evita ambigüedad y hace el problema más limpio
para comparar los tres métodos.

---

## Paso 3 — Balanceo de clases

- Recortar la clase mayoritaria para que haya el mismo número de positivos y negativos.
- Sin esto, un clasificador que siempre prediga "positivo" tendría accuracy alta y los
  resultados de BoW / TF-IDF / embeddings serían engañosos.

---

## Paso 4 — Tokenización

Reusar exactamente el mismo preprocesamiento de `word2vec_word2vec.py`:

```python
tokens = nltk.word_tokenize(texto.lower(), language="spanish")
tokens = [t for t in tokens if t.isalpha()]
```

Así los tres métodos parten de los mismos tokens y la comparación es justa.

---

## Paso 5 — Split train / test

- Usar el split oficial de HuggingFace (`train` para entrenar, `test` para evaluar).
- Es reproducible y es la práctica estándar en NLP.
- Opción ligera: tomar una muestra del `train` (p. ej. 5 000 reseñas) y usar el
  `test` oficial tal cual, para experimentar más rápido.

---

## Paso 6 — Guardar en disco

Guardar dos archivos para que los tres scripts carguen exactamente los mismos datos
sin repetir el preprocesamiento:

```
corpus_train.jsonl   →  {"tokens": [...], "label": 0}
corpus_test.jsonl    →  {"tokens": [...], "label": 1}
```

---

## Resumen visual

```
HuggingFace (amazon_reviews_multi, es)
    ↓
Filtrar estrellas 3  →  Binarizar (0 / 1)
    ↓
Balancear clases
    ↓
Tokenizar (NLTK, mismo que word2vec_word2vec.py)
    ↓
Split oficial  →  train / test
    ↓
corpus_train.jsonl  +  corpus_test.jsonl
```

Con esto los tres métodos arrancan desde el mismo punto y cualquier diferencia
en resultados viene del método, no del dato.

---

## Métodos a comparar

| Método | Script |
|--------|--------|
| Bag of Words (BoW) | `bow_sentimiento.py` |
| TF-IDF | `tfidf_sentimiento.py` |
| Word2Vec embeddings | `w2v_sentimiento.py` |
