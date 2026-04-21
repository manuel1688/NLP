# Lab 04 — TF-IDF

**Curso:** Diplomado en Inteligencia Artificial  
**Nivel:** Introductorio–Intermedio  
**Duración estimada:** 45–60 minutos  
**Archivo de trabajo:** `lab_tfidf_solucion.py`  
**Prerequisito:** Lab 03 — Bag of Words

---

## Contexto — El problema de BoW

En el Lab anterior construiste una matriz BoW sobre el corpus de Wikipedia. Ejecuta mentalmente este experimento:

> En un corpus de 50 artículos de Wikipedia en español, ¿cuáles son las palabras con mayor frecuencia total?

La respuesta casi siempre es: *de, la, el, en, que, se, un, los, con, por…*

Estas palabras aparecen en **todos** los documentos. BoW les asigna conteos altísimos — y eso hace que dominen cualquier comparación entre documentos. El resultado: dos artículos sobre temas completamente distintos parecen muy similares porque comparten "de", "la" y "un".

**La pregunta clave:**
> ¿Cómo diferenciamos una palabra informativa (como "Pirineos" o "Constitución") de una que no aporta nada ("de", "la")?

La respuesta es **TF-IDF**.

> [!NOTE]
> TF-IDF no es una alternativa a BoW — es una **mejora**. Sigue siendo una bolsa de palabras,
> pero en lugar de contar ocurrencias usa pesos que reflejan la importancia real de cada término.

---

## Parte 1 — ¿Por qué BoW falla?

### Ejercicio 1 — Visualiza el problema

Abre `lab_tfidf_solucion.py` y ejecuta el bloque **[PARTE 1]**.

Verás el top-10 de palabras por frecuencia BoW sobre las primeras 50 oraciones del corpus.

> ¿Alguna de esas palabras dice algo útil sobre el **tema** de los documentos?

> ¿Qué pasaría si usaras esas palabras para comparar un artículo sobre Andorra con uno sobre la Revolución Francesa?

---

## Parte 2 — Cálculo Manual de TF-IDF

Usaremos el mismo mini-corpus del Lab anterior para calcular TF-IDF paso a paso.

### Mini-corpus (3 documentos)

| Doc | Texto |
|-----|-------|
| D1  | `andorra es un estado pequeño` |
| D2  | `andorra tiene capital en andorra` |
| D3  | `españa es un estado europeo` |

---

### Paso 2a — TF (Term Frequency)

**Definición:**

$$TF(t, d) = \frac{\text{veces que aparece } t \text{ en } d}{\text{total de palabras en } d}$$

> El TF no compara entre documentos — mide la importancia relativa de una palabra *dentro de un documento*.

#### Ejercicio 2a — Calcula el TF para D1 (5 palabras en total)

| Palabra  | Conteo en D1 | TF = conteo / 5 |
|----------|-------------|-----------------|
| andorra  | 1           | 1/5 = __        |
| es       | 1           | 1/5 = __        |
| estado   | 1           | 1/5 = __        |
| pequeño  | 1           | 1/5 = __        |
| un       | 1           | 1/5 = __        |

<details>
<summary>▶ Ver solución</summary>

Todas las palabras de D1 tienen TF = **0.20** — aparecen exactamente una vez en un documento de 5 palabras.

| Palabra | TF |
|---|---|
| andorra | 0.20 |
| es | 0.20 |
| estado | 0.20 |
| pequeño | 0.20 |
| un | 0.20 |

</details>

---

### Paso 2b — IDF (Inverse Document Frequency)

**Definición:**

$$IDF(t) = \ln\left(\frac{N}{df_t}\right)$$

Donde:
- **N** = número total de documentos en el corpus
- **df_t** = número de documentos que contienen el término *t*

> Si una palabra aparece en **todos** los documentos, su IDF es 0 — no distingue nada.
> Si aparece en un **solo** documento, su IDF es máximo — es muy específica.

#### Ejercicio 2b — Completa la tabla IDF (N = 3 documentos)

Para cada palabra, cuenta en cuántos documentos aparece (df) y calcula IDF = ln(3 / df):

| Palabra  | ¿En D1? | ¿En D2? | ¿En D3? | df | IDF = ln(3/df) |
|----------|---------|---------|---------|----|----------------|
| andorra  | Sí      | Sí      | No      | 2  | ln(3/2) = __   |
| capital  | No      | Sí      | No      | 1  | ln(3/1) = __   |
| en       | No      | Sí      | No      | 1  | ln(3/1) = __   |
| es       | Sí      | No      | Sí      | 2  | ln(3/2) = __   |
| estado   | Sí      | No      | Sí      | 2  | ln(3/2) = __   |
| europeo  | No      | No      | Sí      | 1  | ln(3/1) = __   |
| españa   | No      | No      | Sí      | 1  | ln(3/1) = __   |
| pequeño  | Sí      | No      | No      | 1  | ln(3/1) = __   |
| tiene    | No      | Sí      | No      | 1  | ln(3/1) = __   |
| un       | Sí      | No      | Sí      | 2  | ln(3/2) = __   |

<details>
<summary>▶ Ver solución</summary>

Usando ln(1.5) ≈ 0.41 y ln(3) ≈ 1.10:

| Palabra | df | IDF |
|---|---|---|
| andorra | 2 | **0.41** — aparece en 2 de 3 docs |
| capital | 1 | **1.10** — solo en D2 |
| en      | 1 | **1.10** — solo en D2 |
| es      | 2 | **0.41** — en D1 y D3 |
| estado  | 2 | **0.41** — en D1 y D3 |
| europeo | 1 | **1.10** — solo en D3 |
| españa  | 1 | **1.10** — solo en D3 |
| pequeño | 1 | **1.10** — solo en D1 |
| tiene   | 1 | **1.10** — solo en D2 |
| un      | 2 | **0.41** — en D1 y D3 |

**Patrón clave:** palabras compartidas tienen IDF bajo (0.41); palabras únicas tienen IDF alto (1.10).

</details>

---

### Paso 2c — TF-IDF

**Definición:** simplemente multiplica TF por IDF.

$$\text{TF-IDF}(t, d) = TF(t, d) \times IDF(t)$$

#### Ejercicio 2c — TF-IDF para D1

Con TF = 0.20 para todas las palabras de D1 y los IDF calculados:

| Palabra | TF   | IDF  | TF-IDF = TF × IDF |
|---------|------|------|-------------------|
| andorra | 0.20 | 0.41 | __                |
| es      | 0.20 | 0.41 | __                |
| estado  | 0.20 | 0.41 | __                |
| pequeño | 0.20 | 1.10 | __                |
| un      | 0.20 | 0.41 | __                |

> ¿Cuál es la palabra más característica de D1 según TF-IDF? ¿Tiene sentido?

<details>
<summary>▶ Ver solución</summary>

| Palabra | TF-IDF |
|---|---|
| andorra | 0.20 × 0.41 = **0.082** |
| es      | 0.20 × 0.41 = **0.082** |
| estado  | 0.20 × 0.41 = **0.082** |
| pequeño | 0.20 × 1.10 = **0.220** ← más alta |
| un      | 0.20 × 0.41 = **0.082** |

La palabra más característica de D1 es **"pequeño"** — es la única que solo aparece en ese documento.
"es" y "estado" tienen TF-IDF bajo porque también aparecen en D3.

</details>

#### Ejercicio 2d — TF-IDF para D2 y D3

D2 tiene 5 palabras. "andorra" aparece **2 veces**. Rellena la tabla:

| Palabra | TF      | IDF  | TF-IDF |
|---------|---------|------|--------|
| andorra | 2/5=0.40| 0.41 | __     |
| capital | 1/5=0.20| 1.10 | __     |
| en      | 1/5=0.20| 1.10 | __     |
| tiene   | 1/5=0.20| 1.10 | __     |

D3 tiene 5 palabras, todas con TF = 0.20:

| Palabra | TF   | IDF  | TF-IDF |
|---------|------|------|--------|
| españa  | 0.20 | 1.10 | __     |
| es      | 0.20 | 0.41 | __     |
| estado  | 0.20 | 0.41 | __     |
| europeo | 0.20 | 1.10 | __     |
| un      | 0.20 | 0.41 | __     |

<details>
<summary>▶ Ver solución</summary>

**D2:**
| Palabra | TF-IDF |
|---|---|
| andorra | 0.40 × 0.41 = **0.164** |
| capital | 0.20 × 1.10 = **0.220** ← |
| en      | 0.20 × 1.10 = **0.220** ← |
| tiene   | 0.20 × 1.10 = **0.220** ← |

D2 se caracteriza por "capital", "en" y "tiene". TF-IDF correcto: a pesar de que "andorra" aparece 2 veces, las palabras exclusivas de D2 son más características.

**D3:**
| Palabra | TF-IDF |
|---|---|
| españa  | 0.20 × 1.10 = **0.220** ← |
| es      | 0.20 × 0.41 = **0.082** |
| estado  | 0.20 × 0.41 = **0.082** |
| europeo | 0.20 × 1.10 = **0.220** ← |
| un      | 0.20 × 0.41 = **0.082** |

D3 se caracteriza por "españa" y "europeo". Tiene sentido.

</details>

> [!NOTE]
> **sklearn usa una fórmula ligeramente diferente** con suavizado para evitar división por cero:
> `IDF(t) = ln((1 + N) / (1 + df_t)) + 1`
> Los valores numéricos serán distintos a los del cálculo manual, pero el **ranking** de palabras es el mismo.

---

## Parte 3 — `TfidfVectorizer` sobre el corpus real

### Ejercicio 3a — Aplicar TF-IDF al corpus

Descomenta el bloque **[PARTE 3a]** en `lab_tfidf_solucion.py`.

```python
# tfidf = TfidfVectorizer(max_features=500, stop_words=list(STOPWORDS_ES))
# X_tfidf = tfidf.fit_transform(oraciones_limpias)
# df_tfidf = pd.DataFrame(X_tfidf.toarray(),
#                         columns=tfidf.get_feature_names_out())
# print(df_tfidf.shape)
```

> ¿La forma de la matriz es la misma que en BoW? ¿Por qué?

### Ejercicio 3b — Palabras más características por documento

Descomenta el bloque **[PARTE 3b]** que muestra las top-5 palabras con mayor TF-IDF para los primeros 3 documentos.

> ¿Las palabras que aparecen son más descriptivas del contenido que las del top-10 BoW?

---

## Parte 4 — BoW vs TF-IDF: Comparación Directa

### Ejercicio 4 — Mismo documento, dos representaciones

Descomenta el bloque **[PARTE 4]** en `lab_tfidf_solucion.py`. Verás una tabla que muestra las top-10 palabras para el primer documento, según BoW y según TF-IDF.

**Ejemplo de salida esperada:**

```
Documento 0 — Top 10 palabras

BoW (conteos)          TF-IDF (pesos)
------------------------+------------------------
de               45     andorra          0.312
la               38     principado       0.298
el               31     constitución     0.287
en               28     parlamentario    0.275
es               22     copríncipes      0.261
...
```

> ¿Cuál de las dos representaciones describe mejor el contenido del documento?

> ¿Qué información perdería un algoritmo de clasificación si usara BoW en lugar de TF-IDF?

---

## Parte 5 — Similitud Coseno

TF-IDF convierte cada documento en un vector. Podemos medir qué tan similares son dos documentos calculando el **ángulo** entre sus vectores — si apuntan en la misma dirección, los documentos son similares.

$$\text{similitud}(d_1, d_2) = \cos(\theta) = \frac{d_1 \cdot d_2}{||d_1|| \cdot ||d_2||}$$

El resultado está entre 0 (completamente distintos) y 1 (idénticos).

### Ejercicio 5a — Similitud en el mini-corpus

Con los vectores TF-IDF calculados en la Parte 2:

- D1 y D3 comparten "es", "estado" y "un" con valores similares
- D1 y D2 solo comparten "andorra"
- D2 y D3 **no comparten ninguna palabra**

> Antes de ejecutar el código: ¿qué par de documentos esperas que tenga mayor similitud coseno?

### Ejercicio 5b — Similitud sobre el corpus real

Descomenta el bloque **[PARTE 5]** en `lab_tfidf_solucion.py`.

```python
# from sklearn.metrics.pairwise import cosine_similarity
# sim = cosine_similarity(X_tfidf[:10])
# df_sim = pd.DataFrame(sim, columns=[f"D{i}" for i in range(10)],
#                            index=[f"D{i}" for i in range(10)])
# print(df_sim.round(2))
```

> ¿Qué par de documentos tiene mayor similitud? ¿Tiene sentido temáticamente?

> ¿Qué par tiene similitud 0.0? ¿Por qué (pista: mira las palabras que tienen en común)?

---

## Parte 6 (Bonus) — TF-IDF con N-gramas

### Ejercicio 6 — Bigramas + TF-IDF

Descomenta el bloque **[PARTE 6]** con `ngram_range=(1, 2)`.

> ¿Aparecen bigramas en las palabras más características?  
> ¿Mejora la representación al combinar unigramas y bigramas?

---

## Reflexión Final

1. **IDF = 0:** Si una palabra aparece en **todos** los documentos del corpus, su IDF es 0 y su TF-IDF también. ¿Qué tipo de palabras caen en esta categoría? ¿Cuándo sería esto un problema?

2. **Tamaño del corpus importa:** El IDF de "Andorra" en un corpus de 3 documentos es muy distinto al de un corpus de 50.000. ¿Por qué? ¿Mejora o empeora al aumentar el corpus?

3. **TF-IDF vs Word2Vec:** TF-IDF sigue siendo una bolsa de palabras — "rey" y "monarca" tienen vectores completamente independientes aunque signifiquen lo mismo. ¿Qué representación necesitarías para que palabras con significados similares produzcan vectores similares? → **Word2Vec** (Lab 05).
