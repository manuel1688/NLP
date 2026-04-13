# Lab 02 — Bigram Model desde Cero

Curso: Diplomado en Inteligencia Artificial  
Nivel: Introductorio  
Duración estimada: 40–60 minutos

---

## Objetivo

Construir un modelo de lenguaje de bigramas desde cero para:

1. Contar pares de palabras consecutivas.
2. Calcular probabilidades condicionales simples.
3. Generar una frase corta usando esas probabilidades.

No usarás librerías de NLP. Solo Python puro.

---

## Recordatorio rápido

Un bigrama es un par de palabras consecutivas:

- "yo estudio" es un bigrama.
- "estudio nlp" es otro bigrama.

Si una palabra actual es w_i, el modelo estima la probabilidad de la siguiente w_{i+1}:

P(w_{i+1} | w_i)

---

## Corpus simulado

Usaremos este corpus pequeño (oraciones cortas):

1. yo estudio nlp
2. yo estudio python
3. yo aprendo nlp
4. tu estudias nlp
5. yo estudio modelos

Para simplificar el inicio y fin de frase, agrega tokens especiales:

- <s> al inicio
- </s> al final

Ejemplo:

<s> yo estudio nlp </s>

---

## Tu trabajo

Crea un archivo llamado bigram_model.py y completa estos pasos.

### Paso 1: Definir el corpus

Guarda el corpus como una lista de strings.

### Paso 2: Tokenizar y agregar fronteras

Para cada oración:

1. Convierte a minúsculas.
2. Separa por espacios.
3. Agrega <s> al inicio y </s> al final.

### Paso 3: Contar unigramas y bigramas

Necesitas dos estructuras:

1. conteo_unigramas: cuántas veces aparece cada palabra como palabra actual.
2. conteo_bigramas: cuántas veces aparece cada par (palabra_actual, palabra_siguiente).

Pista:

- Recorre cada oración tokenizada con índices.
- Si estás en posición i, el bigrama es (tokens[i], tokens[i+1]).

### Paso 4: Calcular probabilidades

Usa la fórmula:

P(siguiente | actual) = conteo_bigramas[(actual, siguiente)] / conteo_unigramas[actual]

Guarda los resultados en un diccionario probabilidades_bigramas.

### Paso 5: Consultas del modelo

Imprime:

1. P(estudio | yo)
2. P(nlp | estudio)
3. P(</s> | nlp)

Si un bigrama no existe, muestra 0.0.

### Paso 6: Generación de una frase simple

Genera una frase comenzando en <s>:

1. Busca todas las palabras posibles siguientes.
2. Elige la de mayor probabilidad (estrategia greedy).
3. Repite hasta llegar a </s> o máximo 10 pasos.

Luego imprime la frase sin <s> ni </s>.

---

## Validación esperada

Con el corpus dado, deberían observar algo cercano a:

- P(estudio | yo) = 0.75
- P(nlp | estudio) = 0.3333...
- P(</s> | nlp) = 1.0

Y una frase generada probable:

- yo estudio nlp

Nota: Si empatas probabilidades, tu frase puede variar según cómo resuelvas el empate.

---

## Preguntas de reflexión

1. ¿Qué limita a un modelo de bigramas frente a modelos neuronales modernos?
2. ¿Qué pasa con frases no vistas en el corpus?
3. ¿Cómo cambia el resultado si el corpus crece?
4. ¿Por qué necesitaríamos suavizado (smoothing) en un corpus real?

---

## Extensión opcional

1. Agrega una nueva oración al corpus: yo estudio redes
2. Recalcula probabilidades.
3. Observa cómo cambian P(nlp | estudio) y la frase generada.

---

## Entrega

1. Archivo bigram_model.py con código funcional.
2. Captura o salida de consola mostrando:
   - Los tres cálculos de probabilidad.
   - Una frase generada.
