# Lab 04 — TF-IDF · Respuestas de referencia

---

## Parte 2a — Aplicar TF-IDF al corpus

> ¿Los valores de la matriz son enteros o decimales? ¿Por qué?

**Decimales.** TF-IDF multiplica TF (frecuencia relativa, siempre fraccionaria) × IDF (logaritmo, también fraccionario). El resultado casi nunca es entero.

> ¿Cuántas palabras distintas captura con `max_features=500`?

**500 columnas** — ese es exactamente el límite que fija el parámetro `max_features`. sklearn selecciona las 500 palabras con mayor frecuencia de documento (df) en todo el corpus.

---

## Parte 2b — Palabras más características por documento

> ¿Las palabras resultantes describen bien el tema del documento?

**Sí.** A diferencia de BoW puro, las palabras con TF-IDF alto son sustantivos o términos específicos del tema (p.ej. nombres propios, tecnicismos). Las palabras vacías como *"de"*, *"la"*, *"en"* quedan con peso 0 o muy bajo.

> ¿Cuál es más informativa: TF-IDF o BoW puro?

**TF-IDF.** Con BoW, las palabras más frecuentes del corpus (*"de"*, *"el"*, *"que"*) dominan todos los documentos. Con TF-IDF esas palabras se penalizan porque aparecen en demasiados documentos, y suben términos realmente característicos de cada texto.

---

## Parte 3a — Predicción con el mini-corpus

Mini-corpus:
- D1: *el gato duerme en el sofá*
- D2: *el perro duerme en el jardín*
- D3: *el gato persigue el perro*

| Par | Palabras compartidas (IDF > 0) | Similitud esperada |
|-----|-------------------------------|-------------------|
| D1 – D2 | `"duerme"` + `"en"` (2 palabras) | **Mayor** |
| D1 – D3 | `"gato"` (1 palabra) | Media |
| D2 – D3 | `"perro"` (1 palabra) | Similar a D1–D3 |

**El par D1–D2 debería tener la mayor similitud** porque comparten dos palabras con IDF > 0 (`"duerme"` y `"en"`), mientras que los otros pares solo comparten una.

> Nota: `"el"` no cuenta — su IDF = 0, así que su TF-IDF = 0 y no contribuye al producto punto.

---

## Parte 3b — Similitud sobre el corpus real

> ¿Qué par tiene mayor similitud? ¿Tiene sentido temáticamente?

Depende del corpus y del valor de `N_LINEAS`. En general, **el par con mayor similitud suele ser dos oraciones del mismo artículo de Wikipedia** (hablan del mismo tema, usan el mismo vocabulario). Tiene sentido: si dos textos hablan de lo mismo, sus vectores TF-IDF apuntan en la misma dirección.

> ¿Qué par tiene similitud 0.0? ¿Por qué?

Dos documentos tienen similitud 0 cuando **no comparten ninguna palabra con IDF > 0** — es decir, su vocabulario específico es completamente distinto. El producto punto de sus vectores es 0 porque no hay ninguna dimensión donde ambos tengan peso positivo simultáneamente.

---

## Parte 4 (Bonus) — TF-IDF con bigramas

> ¿Aparecen bigramas entre las palabras más características?

**Sí, normalmente sí.** Los bigramas de nombres propios compuestos (*"nueva york"*, *"edad media"*) o colocaciones frecuentes suelen tener TF-IDF alto porque son muy específicos de ciertos documentos.

> ¿Mejora la representación al combinar unigramas y bigramas?

**Depende del corpus y la tarea.** Los bigramas capturan contexto local y reducen ambigüedad (p.ej. *"banco"* vs *"banco central"*). La desventaja es que el vocabulario crece mucho (más columnas, más dispersión). Para documentos cortos o con nombres propios, suelen ayudar. Para corpus muy grandes y variados, el beneficio es marginal.

---

## Reflexión Final

### 1. IDF = 0

> ¿Qué tipo de palabras caen en esta categoría?

**Palabras funcionales o stopwords** (*"de"*, *"la"*, *"en"*, *"que"*…): aparecen en prácticamente todos los documentos de cualquier corpus en español. También palabras muy genéricas del dominio (p.ej. *"wikipedia"* si el corpus es 100% Wikipedia).

> ¿Cuándo podría esto ser un problema?

Cuando una palabra clave del dominio es tan común en el corpus que su IDF cae a 0 aunque sea informativa. Por ejemplo, en un corpus médico, *"paciente"* puede aparecer en todos los documentos y perder su peso — aunque sí distinguiría ese corpus de uno de física.

---

### 2. Tamaño del corpus importa

> ¿Por qué el IDF de `"gato"` cambia tanto entre 3 y 50 000 documentos?

$$IDF(t) = \ln\left(\frac{N}{df_t}\right)$$

Con N=3 y df=2: IDF = ln(1.5) ≈ **0.41**  
Con N=50 000 y df=2: IDF = ln(25 000) ≈ **10.1**

Con más documentos, una palabra que sigue apareciendo en solo 2 documentos se vuelve *mucho* más específica → su IDF sube considerablemente.

> ¿Mejora o empeora la representación al aumentar el corpus?

**Mejora.** Con más documentos, los IDF son más estables y discriminativos. Las palabras verdaderamente raras obtienen pesos altos; las comunes quedan penalizadas con más precisión. El vector de cada documento describe mejor su contenido específico.

---

### 3. TF-IDF vs Word2Vec

> ¿Qué representación necesitarías para que `"rey"` y `"monarca"` produzcan vectores similares?

**Word2Vec** (o cualquier embedding denso: GloVe, FastText, BERT…).

TF-IDF asigna un eje independiente a cada palabra del vocabulario → `"rey"` y `"monarca"` son perpendiculares por definición, no importa cuánto aparezcan juntas.

Word2Vec aprende vectores de baja dimensión a partir del contexto de cada palabra: si `"rey"` y `"monarca"` aparecen rodeadas de las mismas palabras (*"corona"*, *"trono"*, *"reina"*…), sus vectores aprenden a ser similares.

> Continúa en el **Lab 05 — Word2Vec**.
