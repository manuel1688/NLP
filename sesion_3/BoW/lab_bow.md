# Lab 03 — Bag of Words (BoW)

**Curso:** Diplomado en Inteligencia Artificial  
**Nivel:** Introductorio–Intermedio  
**Duración estimada:** 45–60 minutos  
**Archivo de trabajo:** `lab_bow_solucion.py`

---

## Contexto

**Bag of Words (BoW)** es la representación más simple de texto para modelos de machine learning. La idea: ignorar el orden de las palabras y contar cuántas veces aparece cada una. El resultado es un **vector de conteos** — un documento se convierte en un punto en un espacio de |V| dimensiones, donde V es el vocabulario completo.

**Cuándo se usa:** clasificación de texto, detección de spam, análisis de sentimiento básico, búsqueda de documentos.

> [!IMPORTANT]
> **Limitación central:** "no es bueno" y "bueno es no" producen el **mismo vector BoW**.
> El orden de las palabras desaparece por completo.

---

## Parte 1 — Cálculo Manual

Antes de usar ninguna librería, vamos a construir una matriz BoW a mano con un mini-corpus inspirado en el corpus de Wikipedia que usaremos después.

### Mini-corpus (3 documentos)

| Doc | Texto |
|-----|-------|
| D1  | `andorra es un estado pequeño` |
| D2  | `andorra tiene capital en andorra` |
| D3  | `españa es un estado europeo` |

### Vocabulario

Todas las palabras únicas del corpus, ordenadas alfabéticamente:

`andorra · capital · en · es · estado · europeo · pequeño · tiene · un · españa`

### Ejercicio 1a — Completa la matriz BoW

Cuenta cuántas veces aparece cada palabra en cada documento y rellena las celdas `__`:

| Doc | andorra | capital | en | es | estado | europeo | pequeño | tiene | un | españa |
|-----|---------|---------|----|----|--------|---------|---------|-------|----|--------|
| D1  | __      | __      | __ | __ | __     | __      | __      | __    | __ | __     |
| D2  | __      | __      | __ | __ | __     | __      | __      | __    | __ | __     |
| D3  | __      | __      | __ | __ | __     | __      | __      | __    | __ | __     |

<details>
<summary>▶ Ver solución</summary>

| Doc | andorra | capital | en | es | estado | europeo | pequeño | tiene | un | españa |
|-----|---------|---------|----|----|--------|---------|---------|-------|----|--------|
| D1  | 1       | 0       | 0  | 1  | 1      | 0       | 1       | 0     | 1  | 0      |
| D2  | 2       | 1       | 1  | 0  | 0      | 0       | 0       | 1     | 0  | 0      |
| D3  | 0       | 0       | 0  | 1  | 1      | 1       | 0       | 0     | 1  | 1      |

Nota: "andorra" aparece **2 veces** en D2 porque el texto dice "andorra tiene capital en **andorra**".

</details>

### Ejercicio 1b — Preguntas de análisis

> 1. Mirando solo los vectores, ¿qué par de documentos es más similar: (D1, D2) o (D1, D3)? ¿Por qué?

> 2. D1 y D3 no mencionan los mismos países, pero comparten "es", "estado" y "un". ¿Qué nos dice eso sobre el tema de ambos documentos?

> 3. Considera D4: `"pequeño estado es andorra un"`. ¿Cómo se vería su vector BoW? ¿Qué problema ves?

---

## Parte 2 — Preprocesamiento del Corpus Real

El corpus `eswiki_corpus.txt` es un dump de Wikipedia en español. Antes de aplicar BoW necesita limpieza porque contiene:

```
<ref>Polibio Historias III, 35, 1</ref>          ← referencias wiki
[[Iglesia de Sant Joan de Caselles]]              ← enlaces internos
''texto en cursiva'' y '''texto en negrita'''     ← formato wiki
&amp;nbsp; &lt;ref&gt; &quot;                     ← entidades HTML
{{plantilla|parámetro=valor}}                     ← plantillas wiki
```

### Ejercicio 2 — Observa la función de limpieza

Abre `lab_bow_solucion.py` y localiza la función `limpiar(texto)`.  
Ejecuta el bloque **[PARTE 2]** y observa la salida antes/después.

```python
# Ejemplo de entrada ruidosa del corpus real
antes  = "Andorra &lt;ref&gt;ver nota&lt;/ref&gt; es un [[Estado]] ''soberano''."
despues = limpiar(antes)
print(despues)
# → "andorra es un estado soberano"
```

> ¿Qué transformaciones aplica la función? Lista al menos 4 tipos de ruido que elimina.

> ¿Qué pasa con números y signos de puntuación? ¿Por qué se eliminan para BoW?

---

## Parte 3 — BoW con `CountVectorizer`

### Ejercicio 3a — Unigramas sobre el corpus

Descomenta y ejecuta el bloque **[PARTE 3a]** en `lab_bow_solucion.py`.

```python
# vectorizador = CountVectorizer(max_features=500)
# X = vectorizador.fit_transform(oraciones_limpias[:50])
# df = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())
# print(f"Forma de la matriz: {X.shape}")
# print(df.head())
```

**Salida esperada:**
```
Forma de la matriz: (50, 500)
   andorra  capital  cataluña  ...
0        3        0         0  ...
1        0        1         0  ...
...
```

> ¿Qué significa que la matriz tenga forma `(50, 500)`?

> ¿Qué porcentaje aproximado de celdas tienen valor 0 (son "vacías")? Esto se llama **dispersión** (sparsity).

### Ejercicio 3b — Top-10 palabras más frecuentes

Descomenta el bloque **[PARTE 3b]**.

> ¿Qué palabras aparecen en el top-10?  
> ¿Son útiles para entender el contenido de los documentos?

---

## Parte 4 — Stop Words

Palabras como *el, la, de, en, que, se, un…* son extremadamente frecuentes en cualquier texto pero no aportan información sobre el tema. Se llaman **stop words**.

### Ejercicio 4 — Eliminar stop words

Descomenta el bloque **[PARTE 4]** que usa la lista `STOPWORDS_ES`.

```python
# vectorizador_sw = CountVectorizer(max_features=500,
#                                   stop_words=list(STOPWORDS_ES))
# X_sw = vectorizador_sw.fit_transform(oraciones_limpias[:50])
# ...
```

> Compara el top-10 con y sin stop words.  
> ¿Cuántas palabras del top-10 original desaparecen?

> ¿Qué palabras nuevas aparecen en el top-10 al eliminar stop words? ¿Son más informativas?

---

## Parte 5 — N-gramas

Un **bigrama** es una secuencia de 2 palabras consecutivas. Captura cierta información de orden que el unigrama pierde:

- Unigrama: `"andorra"`, `"la"`, `"vieja"` — tres palabras separadas
- Bigrama: `"andorra la"`, `"la vieja"` — preserva la co-ocurrencia

### Ejercicio 5 — Bigramas sobre el corpus

Descomenta el bloque **[PARTE 5]** con `ngram_range=(2, 2)`.

```python
# vectorizador_bg = CountVectorizer(ngram_range=(2, 2), max_features=300)
# X_bg = vectorizador_bg.fit_transform(oraciones_limpias[:50])
# ...
```

> ¿Aparecen bigramas que corresponden a nombres propios o lugares?  
> Ejemplo: ¿ves "andorra la" o "la vieja"?

> ¿Qué ventaja tienen los bigramas para este corpus de Wikipedia?  
> ¿Qué desventaja tiene usar bigramas en lugar de unigramas?

---

## Reflexión Final

1. **Orden ignorado:** Hemos visto que D1 = `"andorra es un estado pequeño"` y D4 = `"pequeño estado es andorra un"` producen el mismo vector BoW. ¿Es esto siempre un problema, o hay casos donde no importa?

2. **Dispersión:** Con 50 documentos y vocabulario de 500 palabras, la mayoría de celdas son 0. En un corpus real de 50.000 documentos con vocabulario de 100.000 palabras, ¿cuántos valores necesitarías almacenar si no usaras matrices dispersas? ¿Qué implica esto para la memoria?

3. **Relevancia ignorada:** BoW trata "de" (aparece 5.000 veces) y "Pirineos" (aparece 3 veces) de forma proporcional a sus conteos. ¿Cuál de las dos palabras es más informativa para identificar un documento sobre Andorra? ¿Cómo podríamos penalizar las palabras muy frecuentes y premiar las específicas?

> La respuesta a la pregunta 3 es **TF-IDF** → continúa en el **Lab 04**.
