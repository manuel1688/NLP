# 📋 Plan de Implementación: Word2Vec Educativo (Skip-gram)

## Fase 1: Corpus y Vocabulario

- **Corpus de trabajo** — 5-10 oraciones propias definidas inline para implementar y debuggear (ver [`corpus_recomendados.md`](corpus_recomendados.md) para opciones de escala)
- **Tokenización** — Reutilizar pipeline de `sesion_2/tokenization.py`
- **Construcción de vocabulario** — Diccionarios `word2idx` / `idx2word`
- **Ventana deslizante** — Definir `window_size` y generar pares `(target, context)`
- **Explicación conceptual de one-hot** — Solo como intuición; en código se usan índices enteros directamente

## Fase 2: Skip-gram con Negative Sampling

- **Matrices de embeddings** — Inicializar `W_embed` (input) y `W_context` (output)
- **Forward pass** — Lookup del embedding + dot product con contexto + sigmoid
- **Loss** — Binary cross-entropy (no softmax completo sobre todo el vocabulario)
- **Backward pass** — Cálculo manual de gradientes + actualización SGD

## Fase 3: Entrenamiento y Monitoreo

- **Loop de entrenamiento** — SGD puro (un par a la vez), learning rate fijo
- **Validación de escala** — Opcional: reentrenar con **Text8** (`gensim.downloader`) o **NLTK cess_esp** para confirmar que los embeddings aprenden patrones reales
- **Logging de pérdida** — Registrar loss por época
- **Gráfica loss vs. época** — Visualizar convergencia con matplotlib
- **Comparación visual antes/después** — PCA de embeddings aleatorios vs. entrenados (conecta con `sesion_2/embedding_viz.py`)

## Fase 4: Exploración de Embeddings

- **Cosine similarity** — Búsqueda de palabras más cercanas
- **Operaciones analógicas** — Implementar la mecánica `a - b + c ≈ d` con analogías alcanzables según el corpus
- **Visualización PCA 2D** — Gráfico con categorías coloreadas
- **(Bonus)** Cargar **Google News Word2Vec** o **FastText ES** pre-entrenado para demostrar analogías reales tipo `king - man + woman = queen` (instrucciones en [`corpus_recomendados.md`](corpus_recomendados.md))

---

## Notas Pedagógicas

- **One-hot** se explica pero no se implementa — en la práctica se trabaja con índices + lookup
- **Negative sampling** es parte del modelo (Fase 2), no de la optimización
- **Mini-batch** se omite intencionalmente — SGD puro es más claro para entender el flujo de gradientes
- **Analogías**: con corpus pequeño no se esperan resultados tipo Wikipedia; usar ejemplos realistas del propio corpus
- La comparación antes/después es el momento más impactante del lab
