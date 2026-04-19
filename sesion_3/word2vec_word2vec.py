# ================================================================
# Word2Vec Educativo — Skip-gram con Negative Sampling
# Ver plan completo: plan_word2vec.md
# ================================================================
# Fase 1: Corpus, vocabulario y pares skip-gram       <- estamos aqui
# Fase 2: Modelo Skip-gram + Negative Sampling
# Fase 3: Entrenamiento SGD y monitoreo de perdida
# Fase 4: Evaluacion (cosine similarity, analogias, PCA)
# ================================================================

import os
import nltk
from collections import Counter

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------
WINDOW_SIZE = 2    # palabras a cada lado de la palabra objetivo
MIN_COUNT   = 3    # frecuencia minima para entrar al vocabulario

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "eswiki_corpus.txt")

# ----------------------------------------------------------------
# PASO 1 — Cargar oraciones del corpus
# ----------------------------------------------------------------
# Herramienta : nltk.word_tokenize (tokenizador para español)
# Que hace    : divide cada linea en tokens y filtra no-alfabeticos
# Por que     : Word2Vec trabaja con listas de palabras, no texto crudo

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

print("[ PASO 1 ] Tokenizacion con nltk.word_tokenize")

oraciones = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for linea in f:
        linea = linea.strip()
        if not linea:
            continue
        tokens = nltk.word_tokenize(linea.lower(), language="spanish")
        tokens = [t for t in tokens if t.isalpha()]
        if len(tokens) > 2:
            oraciones.append(tokens)

print(f"  Oraciones cargadas : {len(oraciones)}")
print(f"  Ejemplo de tokens  : {oraciones[0]}")

# ----------------------------------------------------------------
# PASO 2 — Construir vocabulario
# ----------------------------------------------------------------
# Herramienta : collections.Counter
# Que hace    : cuenta frecuencias y filtra palabras raras (min_count)
# Por que     : solo palabras frecuentes tienen suficiente contexto
#               para aprender un buen embedding
#
# Resultado   : dos diccionarios de mapeo bidireccional
#   word2idx : "perro" -> 342
#   idx2word : 342     -> "perro"

print("\n[ PASO 2 ] Construccion de vocabulario con collections.Counter")

contador = Counter(token for oracion in oraciones for token in oracion)
vocab     = sorted(w for w, c in contador.items() if c >= MIN_COUNT)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

print(f"  Palabras unicas en corpus      : {len(contador)}")
print(f"  Vocabulario filtrado (min={MIN_COUNT})  : {len(vocab)} palabras")
print(f"  Ejemplo word2idx               : 'de' -> {word2idx.get('de', 'N/A')}")
print(f"  Ejemplo idx2word               : 0    -> '{idx2word[0]}'")

# ----------------------------------------------------------------
# PASO 3 — Generar pares Skip-gram
# ----------------------------------------------------------------
# Herramienta : ventana deslizante (sliding window) manual
# Que hace    : para cada palabra objetivo extrae sus vecinos
#               dentro de un rango de WINDOW_SIZE posiciones
# Por que     : estos pares son los ejemplos de entrenamiento;
#               le dicen al modelo "estas palabras aparecen juntas"
#
# Ejemplo con window=2:
#   oracion : [el, gato, come, pescado, fresco]
#   target  : "come"  (posicion 2)
#   contexto: "el", "gato", "pescado", "fresco"
#   pares   : (come,el) (come,gato) (come,pescado) (come,fresco)

print("\n[ PASO 3 ] Generacion de pares skip-gram (ventana deslizante)")

pares = []
for oracion in oraciones:
    indices = [word2idx[t] for t in oracion if t in word2idx]

    for i, idx_target in enumerate(indices):
        inicio = max(0, i - WINDOW_SIZE)
        fin    = min(len(indices), i + WINDOW_SIZE + 1)
        for j in range(inicio, fin):
            if j != i:
                pares.append((idx_target, indices[j]))

print(f"  Pares generados : {len(pares)}")
print(f"  Primeros 5 pares (target -> contexto):")
for idx_t, idx_c in pares[:5]:
    print(f"    '{idx2word[idx_t]}' -> '{idx2word[idx_c]}'")
