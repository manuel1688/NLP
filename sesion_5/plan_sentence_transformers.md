# Plan: Clasificación Multi-Clase con Sentence Transformers

Adaptación del notebook `clasificacion_multiclase_Colab.ipynb` (TF-IDF + Logistic Regression)  
para usar embeddings semánticos modernos con `SentenceTransformer`.

---

## ¿Qué cambia respecto al notebook de TF-IDF?

| Aspecto | TF-IDF | SentenceTransformer |
|---|---|---|
| Representación | Frecuencias de términos (sparse) | Embeddings semánticos 384-dim (dense) |
| Vocabulario | Limitado al corpus (`max_features=1000`) | Generaliza a palabras nunca vistas |
| `fit` necesario | Sí (`fit_transform`) | No (modelo pre-entrenado) |
| Velocidad | Muy rápido | Más lento (GPU recomendada en Colab) |
| Multilingüe | No directamente | Sí — `paraphrase-multilingual-MiniLM-L12-v2` soporta español |
| Sensibilidad semántica | Baja (no entiende sinónimos) | Alta (captura significado) |

> El clasificador (`LogisticRegression multi_class='ovr'`) y las métricas **no cambian**.  
> Solo cambia la forma en que se representan los textos.

---

## Estructura del nuevo notebook

### Celda 0 — Instalación (solo Colab)

```python
!pip install sentence-transformers
```

---

### Celda 1 — Subir corpus

```python
from google.colab import files
uploaded = files.upload()
```

---

### Celda 2 — Imports

```python
import json, io
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

CLASES       = ['peliculas', 'restaurantes', 'productos', 'servicios', 'hoteles']
RANDOM_STATE = 42
```

> Se elimina `TfidfVectorizer`. Se agrega `SentenceTransformer`.

---

### Celda 3 — Carga del corpus

Sin cambios respecto al notebook anterior.

```python
data = json.load(io.BytesIO(uploaded['corpus_sentimiento_reviews.json']))

reviews    = [item['texto']     for item in data['reviews']]
categorias = [item['categoria'] for item in data['reviews']]

print(f'Total de reseñas: {len(reviews)}')
for c in CLASES:
    print(f'  {c:<15}: {categorias.count(c):>3}')
```

---

### Celda 4 — Métricas desde cero (sin sklearn)

Sin cambios. Las mismas funciones `precision`, `recall`, `f1`, `accuracy`.

```python
def confusion_matrix_manual(y_true, y_pred, pos_label='positivo'):
    TP = sum(t == pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    TN = sum(t != pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    FP = sum(t != pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    FN = sum(t == pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_manual(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred, pos_label='positivo'):
    TP, _, FP, _ = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred, pos_label='positivo'):
    TP, _, _, FN = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1(y_true, y_pred, pos_label='positivo'):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

---

### Celda 5 — Matriz de confusión N×N

Sin cambios.

```python
def confusion_matrix_multiclase(y_true, y_pred, clases):
    ancho = 14
    cabecera = ' ' * ancho + ''.join(c[:ancho].ljust(ancho) for c in clases)
    print(cabecera)
    print('-' * (ancho * (len(clases) + 1)))
    for real in clases:
        fila = real[:ancho].ljust(ancho)
        for pred in clases:
            count = sum(t == real and p == pred for t, p in zip(y_true, y_pred))
            fila += str(count).ljust(ancho)
        print(fila)
```

---

### Celda 6 — Split Train / Test

Sin cambios.

```python
reviews_train, reviews_test, y_train, y_test = train_test_split(
    reviews, categorias, test_size=0.3, random_state=RANDOM_STATE,
    stratify=categorias
)

print(f'Train : {len(reviews_train)} reseñas')
print(f'Test  : {len(reviews_test)} reseñas')
```

---

### Celda 7 — Embeddings y entrenamiento ⬅ CAMBIO PRINCIPAL

```python
# Cargar el modelo multilingüe pre-entrenado
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Codificar los textos como vectores densos de 384 dimensiones
X_train = encoder.encode(reviews_train, show_progress_bar=True)
X_test  = encoder.encode(reviews_test,  show_progress_bar=True)

# El mismo clasificador OvR — solo cambia la entrada
modelo = LogisticRegression(multi_class='ovr', random_state=RANDOM_STATE)
modelo.fit(X_train, y_train)

print('Modelo entrenado.')
print(f'Clases aprendidas: {modelo.classes_.tolist()}')
print(f'Dimensión de embeddings: {X_train.shape[1]}')
```

> **Nota pedagógica:** no existe `fit_transform` porque el encoder es pre-entrenado.  
> Se llama `encode()` por separado en train y test.

---

### Celda 8 — Evaluación con sklearn

Sin cambios.

```python
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

### Celda 9 — Métricas manuales por clase

Sin cambios.

```python
print(f"{'Categoría':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print('-' * 48)

f1_por_clase = []
for c in CLASES:
    p = precision(y_test, y_pred, pos_label=c)
    r = recall(y_test, y_pred, pos_label=c)
    f = f1(y_test, y_pred, pos_label=c)
    f1_por_clase.append(f)
    print(f'{c:<15} {p:>10.2f} {r:>10.2f} {f:>10.2f}')

print('-' * 48)
macro_f1 = sum(f1_por_clase) / len(f1_por_clase)
print(f"{'Macro F1':<15} {'':>10} {'':>10} {macro_f1:>10.2f}")
print(f'Accuracy global: {accuracy(y_test, y_pred):.2f}')
```

---

### Celda 10 — Matriz de confusión 5×5

Sin cambios.

```python
confusion_matrix_multiclase(y_test, y_pred, CLASES)
```

---

### Celda 11 — Predicción sobre textos nuevos ⬅ CAMBIO

```python
nuevas_reviews = [
    'La actuación fue brillante y la historia muy emotiva',
    'El sushi estaba fresco y el servicio impecable',
    'El producto llegó en perfectas condiciones y funciona genial',
    'Tardaron semanas en responder y no solucionaron el problema',
    'Habitación limpia, cama cómoda y muy buena ubicación',
]

# encoder en lugar de vectorizador.transform()
preds = modelo.predict(encoder.encode(nuevas_reviews))

for texto, pred in zip(nuevas_reviews, preds):
    print(f'  [{pred.upper():<13}]  {texto}')
```

---

## Resumen de cambios celda por celda

| Celda | Acción |
|---|---|
| 0 — Instalación | **Nueva** — `!pip install sentence-transformers` |
| 2 — Imports | **Modificada** — quitar `TfidfVectorizer`, agregar `SentenceTransformer` |
| 7 — Vectorización + entrenamiento | **Modificada** — reemplazar TF-IDF por `encoder.encode()` |
| 11 — Predicción nuevos textos | **Modificada** — reemplazar `vectorizador.transform()` por `encoder.encode()` |
| Resto | Sin cambios |
