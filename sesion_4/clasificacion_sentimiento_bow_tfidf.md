# Clasificación de Sentimiento con BoW/TF-IDF

## ✅ Pipeline completo: BoW/TF-IDF → Clasificador de Sentimiento

### 🔄 El proceso paso a paso

```
Texto crudo → TF-IDF → Vector numérico → Modelo ML → Predicción
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. DATOS: Reviews con etiquetas
reviews = [
    "Excelente película, me encantó la actuación",
    "Muy buena, totalmente recomendada",
    "Obra maestra, brillante dirección",
    "Horrible, aburrida y mal actuada",
    "Pérdida de tiempo, no la vean",
    "Malísima, me dormí a los 10 minutos"
]

sentimientos = [
    'positivo', 'positivo', 'positivo',
    'negativo', 'negativo', 'negativo'
]

# 2. REPRESENTACIÓN: Texto → Vectores TF-IDF
vectorizador = TfidfVectorizer(
    max_features=1000,      # Top 1000 palabras
    ngram_range=(1, 2),     # Unigramas + bigramas
    min_df=2                # Palabra debe aparecer 2+ veces
)

X = vectorizador.fit_transform(reviews)
# X ahora es una matriz sparse de (6 reviews × ~50 features)

# 3. DIVISIÓN: Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, sentimientos, test_size=0.3, random_state=42
)

# 4. MODELO: Entrenar clasificador
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 5. PREDICCIÓN: Nuevas reviews
nuevas_reviews = [
    "Me gustó mucho, muy entretenida",
    "Qué porquería, no la terminen de ver"
]

X_nuevas = vectorizador.transform(nuevas_reviews)
predicciones = modelo.predict(X_nuevas)

print(predicciones)  # ['positivo', 'negativo']
```

---

## 🎯 ¿Por qué funciona?

### El modelo aprende patrones como:

```python
# Palabras que predicen sentimiento positivo:
"excelente" → peso alto para 'positivo'
"brillante" → peso alto para 'positivo'
"recomendada" → peso alto para 'positivo'

# Palabras que predicen sentimiento negativo:
"horrible" → peso alto para 'negativo'
"aburrida" → peso alto para 'negativo'
"pérdida" → peso alto para 'negativo'
```

**TF-IDF ayuda porque:**
- Palabras comunes ("la", "es") tienen peso bajo
- Palabras discriminativas ("excelente", "horrible") tienen peso alto

---

## 📊 Modelos recomendados para BoW/TF-IDF

| Modelo | Rendimiento | Velocidad | Cuándo usarlo |
|--------|-------------|-----------|---------------|
| **Naive Bayes** | ⭐⭐⭐ | ⚡⚡⚡ | Baseline rápido, datasets pequeños |
| **Regresión Logística** | ⭐⭐⭐⭐ | ⚡⚡⚡ | Mejor opción general, interpretable |
| **SVM (Linear)** | ⭐⭐⭐⭐ | ⚡⚡ | Alta dimensionalidad, datasets medianos |
| **Random Forest** | ⭐⭐⭐ | ⚡ | Features ruidosas, no linealidad |

---

## 💪 Ventajas del enfoque TF-IDF + ML clásico

✅ **Funciona muy bien** (80-85% accuracy típicamente)  
✅ **Rápido de entrenar** (segundos, no horas)  
✅ **Interpretable** (puedes ver qué palabras importan)  
✅ **Pocos recursos** (no necesitas GPU)  
✅ **Producción simple** (modelos pequeños, <10 MB)

---

## ⚠️ Limitaciones vs Embeddings

```python
# TF-IDF NO captura similitud semántica:
"La película es excelente" → [0, 1, 0, 0, 0.8, ...]
"La película es magnífica" → [0, 0, 1, 0, 0, ...]
                               ↑
                  "excelente" y "magnífica" son ortogonales

# Word2Vec SÍ captura similitud:
embedding("excelente") ≈ embedding("magnífica")
```

---

## 📈 Ejemplo real con métricas

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset de ejemplo (simulando reviews)
textos_train = [
    "Gran película, actuaciones increíbles",
    "Me encantó, muy recomendable",
    # ... más ejemplos
    "Aburrida, no la terminen",
    "Malísima, horrible guion"
]
y_train = ['pos', 'pos', ..., 'neg', 'neg']

# Pipeline completo
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(textos_train)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Evaluar
X_test = vectorizer.transform(textos_test)
y_pred = modelo.predict(X_test)

print(classification_report(y_test, y_pred))
# Typical output:
#               precision    recall  f1-score
# negativo        0.82      0.85      0.83
# positivo        0.84      0.81      0.82
```

---

## 🏆 Casos de uso reales (2024)

Empresas **aún usan** TF-IDF + ML clásico para:

| Aplicación | Por qué TF-IDF |
|------------|----------------|
| **Filtros de spam** | Rápido, interpretable, efectivo |
| **Clasificación de tickets** | No necesita GPU, fácil de mantener |
| **Análisis de encuestas** | Datasets pequeños, necesita explicabilidad |
| **Moderación de contenido (primer filtro)** | Baja latencia, bajo costo |

---

## ✅ Respuesta directa

### ¿Se puede hacer clasificador de sentimiento con BoW/TF-IDF?

**SÍ, absolutamente:**

```
Reviews → TF-IDF → Logistic Regression → Positivo/Negativo
          ↑                ↑
    Representación      Modelo ML
```

### Rendimiento esperado:
- Datasets limpios: **80-85% accuracy**
- Con tuning: **85-90% accuracy**
- Embeddings (BERT): **90-95% accuracy**

### Cuándo usarlo:
- ✅ Necesitas rapidez
- ✅ Dataset pequeño/mediano (<100k reviews)
- ✅ Quieres interpretabilidad
- ✅ No tienes GPU

### Cuándo NO usarlo:
- ❌ Necesitas capturar sarcasmo/ironía
- ❌ Textos muy cortos (tweets)
- ❌ Múltiples idiomas
- ❌ Estado del arte absoluto

---

## 🎯 Conclusión

> **BoW/TF-IDF NO predicen por sí solos, PERO con un modelo ML encima funcionan muy bien para clasificación de sentimiento.**

Es el enfoque "tradicional" — y todavía es válido en 2024 para muchos casos de uso reales.
