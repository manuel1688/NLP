# ================================================================
# Word2Vec — Fase 1: Corpus, Vocabulario y Pares Skip-gram
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
# Cada linea del archivo es una oracion.
# La tokenizamos y conservamos solo palabras alfabeticas en minuscula.

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

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

print(f"Oraciones cargadas : {len(oraciones)}")
print(f"Ejemplo            : {oraciones[0]}")

# ----------------------------------------------------------------
# PASO 2 — Construir vocabulario
# ----------------------------------------------------------------
# Contamos cuantas veces aparece cada palabra en TODO el corpus.
# Descartamos las que aparecen menos de MIN_COUNT veces (ruido).
# Creamos dos diccionarios:
#   word2idx : palabra -> numero entero (indice)
#   idx2word : numero  -> palabra

contador = Counter(token for oracion in oraciones for token in oracion)
vocab     = sorted(w for w, c in contador.items() if c >= MIN_COUNT)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

print(f"\nVocabulario total             : {len(contador)} palabras unicas")
print(f"Vocabulario filtrado (min={MIN_COUNT}) : {len(vocab)} palabras")
print(f"Primeras 10 palabras          : {vocab[:10]}")

# ----------------------------------------------------------------
# PASO 3 — Generar pares Skip-gram
# ----------------------------------------------------------------
# Para cada palabra objetivo miramos las WINDOW_SIZE palabras
# anteriores y posteriores como contexto.
#
# Ejemplo con window=2:
#   oracion : [el, gato, come, pescado, fresco]
#   target  : "come"  (posicion 2)
#   contexto: "el", "gato", "pescado", "fresco"
#   pares   : (come,el) (come,gato) (come,pescado) (come,fresco)

pares = []
for oracion in oraciones:
    indices = [word2idx[t] for t in oracion if t in word2idx]

    for i, idx_target in enumerate(indices):
        inicio = max(0, i - WINDOW_SIZE)
        fin    = min(len(indices), i + WINDOW_SIZE + 1)
        for j in range(inicio, fin):
            if j != i:
                pares.append((idx_target, indices[j]))

print(f"\nPares skip-gram generados : {len(pares)}")
print("Primeros 5 pares (target, contexto):")
for idx_t, idx_c in pares[:5]:
    print(f"  ({idx2word[idx_t]:15s}, {idx2word[idx_c]})")
