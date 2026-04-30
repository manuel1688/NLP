# ================================================================
# Word2Vec Educativo — Skip-gram con Negative Sampling
# Ver plan completo: plan_word2vec.md
# ================================================================
# Fase 1: Corpus, vocabulario y pares skip-gram       <- implementada
# Fase 2: Modelo Skip-gram + Negative Sampling        <- implementada
# Fase 3: Entrenamiento SGD y monitoreo de perdida    <- implementada
# Fase 4: Evaluacion (cosine similarity, analogias, PCA) <- implementada
# ================================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import Counter

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------
WINDOW_SIZE  = 2    # palabras a cada lado de la palabra objetivo
MIN_COUNT    = 3    # frecuencia minima para entrar al vocabulario
EMBED_DIM    = 100  # dimensión de los vectores (D)
NEG_SAMPLES  = 5    # pares negativos por par positivo
EPOCHS       = 5    # veces que recorremos todos los pares
LR           = 0.025  # learning rate para SGD

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

# ================================================================
# FASE 2 — Modelo Skip-gram con Negative Sampling
# ================================================================
# Referencia: ejemplo.md Pasos 1-6
# ================================================================

print("\n[ FASE 2 ] Inicializacion del modelo Skip-gram")

V = len(vocab)

# ------------------------------------------------------------------
# Tabla de muestreo negativo con suavizado freq^0.75 (Mikolov 2013)
# ------------------------------------------------------------------
# Por que freq^0.75: suaviza la distribucion para que palabras
# frecuentes no dominen el muestreo y palabras raras tengan mas
# oportunidades de aparecer como negativas.
freq_array = np.array([contador[idx2word[i]] for i in range(V)], dtype=float)
freq_array = freq_array ** 0.75
freq_array /= freq_array.sum()

# ------------------------------------------------------------------
# Matrices de embeddings
# Referencia: ejemplo.md — "1. Inicializacion de Matrices"
# ------------------------------------------------------------------
# W_embed   : fila i = vector de la palabra i como TARGET
# W_context : fila i = vector de la palabra i como CONTEXTO
W_embed   = np.random.normal(0, 0.01, (V, EMBED_DIM))
W_context = np.zeros((V, EMBED_DIM))

# Snapshot ANTES de entrenar (para comparacion visual en Fase 4)
W_embed_inicial = W_embed.copy()

print(f"  Vocabulario (V)    : {V}")
print(f"  Dimension (D)      : {EMBED_DIM}")
print(f"  W_embed shape      : {W_embed.shape}")
print(f"  W_context shape    : {W_context.shape}")


def sigmoid(x):
    """σ(x) = 1 / (1 + e^{-x}). Clipea x para evitar overflow."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def train_pair(idx_target, idx_context_pos):
    """
    Un paso de entrenamiento para el par (target, context_pos).
    Muestrea NEG_SAMPLES negativos, calcula loss, gradientes y aplica SGD.
    Referencia: ejemplo.md Pasos 2-6.

    Devuelve: loss escalar del paso.
    """
    # -- Muestreo negativo ------------------------------------------
    # Evita elegir el propio contexto positivo como negativo
    negativos = []
    while len(negativos) < NEG_SAMPLES:
        candidatos = np.random.choice(V, size=NEG_SAMPLES * 2, p=freq_array)
        for c in candidatos:
            if c != idx_context_pos and len(negativos) < NEG_SAMPLES:
                negativos.append(c)

    # -- Forward pass: Paso 2 de ejemplo.md -------------------------
    v_t = W_embed[idx_target]                      # lookup target

    score_pos  = np.dot(v_t, W_context[idx_context_pos])
    scores_neg = [np.dot(v_t, W_context[n]) for n in negativos]

    sig_pos  = sigmoid(score_pos)
    sigs_neg = [sigmoid(s) for s in scores_neg]

    # -- Loss: Paso 4 de ejemplo.md ---------------------------------
    eps  = 1e-10   # evita log(0)
    loss = -np.log(sig_pos + eps)
    for s in sigs_neg:
        loss -= np.log(1.0 - s + eps)

    # -- Gradientes: Paso 5 de ejemplo.md ---------------------------
    # grad respecto a v_t (fila del target en W_embed)
    grad_v_t = (sig_pos - 1.0) * W_context[idx_context_pos]
    for n, s in zip(negativos, sigs_neg):
        grad_v_t += s * W_context[n]

    # grad respecto al contexto positivo
    grad_ctx_pos = (sig_pos - 1.0) * v_t

    # grad respecto a cada contexto negativo
    grads_ctx_neg = [s * v_t for s in sigs_neg]

    # -- SGD: Paso 6 de ejemplo.md ----------------------------------
    W_embed[idx_target]        -= LR * grad_v_t
    W_context[idx_context_pos] -= LR * grad_ctx_pos
    for n, g in zip(negativos, grads_ctx_neg):
        W_context[n] -= LR * g

    return loss


# ================================================================
# FASE 3 — Entrenamiento SGD y monitoreo de perdida
# ================================================================

print("\n[ FASE 3 ] Entrenamiento Skip-gram (SGD puro)")
print(f"  Epocas      : {EPOCHS}")
print(f"  Pares/epoca : {len(pares)}")
print(f"  LR          : {LR}")
print(f"  Neg samples : {NEG_SAMPLES}")

# Intentar usar tqdm para barra de progreso; funcionar sin el si no esta instalado
try:
    from tqdm import tqdm
    usar_tqdm = True
except ImportError:
    usar_tqdm = False
    print("  (instala tqdm para barra de progreso: pip install tqdm)")

historial_loss = []

for epoca in range(EPOCHS):
    random.shuffle(pares)
    loss_total = 0.0

    iterador = tqdm(pares, desc=f"Epoca {epoca + 1}/{EPOCHS}", leave=True) \
               if usar_tqdm else pares

    for idx_t, idx_c in iterador:
        loss_total += train_pair(idx_t, idx_c)

    loss_promedio = loss_total / len(pares)
    historial_loss.append(loss_promedio)
    print(f"  Epoca {epoca + 1}/{EPOCHS} — loss promedio: {loss_promedio:.4f}")

print("\n  Entrenamiento completado.")

# -- Grafica loss vs. epoca -----------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCHS + 1), historial_loss, marker="o", color="steelblue")
plt.title("Loss promedio por época — Word2Vec Skip-gram")
plt.xlabel("Época")
plt.ylabel("Loss promedio")
plt.xticks(range(1, EPOCHS + 1))
plt.grid(True, alpha=0.4)
plt.tight_layout()
grafica_path = os.path.join(os.path.dirname(__file__), "loss_por_epoca.png")
plt.savefig(grafica_path)
plt.close()
print(f"  Grafica guardada en: {grafica_path}")


# ================================================================
# FASE 4 — Exploración de embeddings
# ================================================================

print("\n[ FASE 4 ] Exploracion de embeddings")

# Normalizar W_embed para cosine similarity eficiente
normas     = np.linalg.norm(W_embed, axis=1, keepdims=True)
normas     = np.where(normas == 0, 1e-10, normas)   # evita division por cero
W_norm     = W_embed / normas

normas_ini = np.linalg.norm(W_embed_inicial, axis=1, keepdims=True)
normas_ini = np.where(normas_ini == 0, 1e-10, normas_ini)
W_norm_ini = W_embed_inicial / normas_ini


def most_similar(palabra, topn=10, trained=True):
    """
    Devuelve las topn palabras mas cercanas por cosine similarity.

    Args:
        palabra : str — palabra a consultar
        topn    : int — cuantos vecinos devolver
        trained : bool — True usa W_embed entrenado, False el inicial

    Returns:
        lista de (palabra, similarity) ordenada de mayor a menor
    """
    if palabra not in word2idx:
        return []
    matriz = W_norm if trained else W_norm_ini
    idx    = word2idx[palabra]
    scores = matriz @ matriz[idx]          # cosine similarity contra todo V
    scores[idx] = -1                       # excluir la propia palabra
    top_idx = np.argsort(scores)[::-1][:topn]
    return [(idx2word[i], float(scores[i])) for i in top_idx]


def analogy(a, b, c, topn=5):
    """
    Analogia: a es a b como c es a ?
    Calcula W[b] - W[a] + W[c] y busca el vecino mas cercano.

    Ejemplo: analogy("rey", "reina", "hombre") → "mujer"
    (requiere corpus grande para resultados fiables)
    """
    for palabra in (a, b, c):
        if palabra not in word2idx:
            print(f"  '{palabra}' no esta en el vocabulario")
            return []

    vec = W_norm[word2idx[b]] - W_norm[word2idx[a]] + W_norm[word2idx[c]]
    vec /= (np.linalg.norm(vec) + 1e-10)

    scores  = W_norm @ vec
    excluir = {word2idx[a], word2idx[b], word2idx[c]}
    resultados = []
    for idx in np.argsort(scores)[::-1]:
        if idx not in excluir:
            resultados.append((idx2word[idx], float(scores[idx])))
        if len(resultados) == topn:
            break
    return resultados


# -- Ejemplo de most_similar ----------------------------------------
palabras_demo = [vocab[i] for i in range(0, min(5, V))]
print("\n  most_similar (primeras 5 palabras del vocabulario):")
for p in palabras_demo:
    vecinos = most_similar(p, topn=5)
    vecinos_str = ", ".join(f"{w}({s:.2f})" for w, s in vecinos)
    print(f"    '{p}' → {vecinos_str}")

# -- Ejemplo de analogia --------------------------------------------
print("\n  Ejemplo de analogia (necesita corpus grande para ser precisa):")
primeras = vocab[:3]
if len(primeras) == 3:
    a, b, c = primeras
    resultado = analogy(a, b, c, topn=3)
    resultado_str = ", ".join(f"{w}({s:.2f})" for w, s in resultado)
    print(f"    {a} : {b} :: {c} : {resultado_str}")

# -- Visualizacion PCA 2D -------------------------------------------
# Selecciona hasta 40 palabras para graficar
# Referencia de estructura visual: sesion_2/embedding_viz.py
try:
    from sklearn.decomposition import PCA

    n_palabras = min(40, V)
    indices_demo = list(range(n_palabras))
    palabras_pca = [idx2word[i] for i in indices_demo]

    # PCA sobre embeddings entrenados
    vecs_entrenados = W_embed[indices_demo]
    vecs_iniciales  = W_embed_inicial[indices_demo]

    pca = PCA(n_components=2, random_state=42)
    coords_entrenados = pca.fit_transform(vecs_entrenados)
    coords_iniciales  = pca.transform(vecs_iniciales)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coords, titulo in [
        (axes[0], coords_iniciales,  "Antes de entrenar (aleatorio)"),
        (axes[1], coords_entrenados, f"Después de {EPOCHS} épocas"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30, color="steelblue")
        for i, palabra in enumerate(palabras_pca):
            ax.annotate(palabra, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.85)
        ax.set_title(titulo)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.3)

    plt.suptitle("PCA de embeddings — Word2Vec Skip-gram", fontsize=13)
    plt.tight_layout()
    pca_path = os.path.join(os.path.dirname(__file__), "embeddings_pca.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"\n  Grafica PCA guardada en: {pca_path}")

except ImportError:
    print("\n  (instala scikit-learn para visualizacion PCA: pip install scikit-learn)")

print("\n[ LISTO ] Pipeline completo ejecutado.")
print("  Archivos generados:")
print(f"    {grafica_path}")
try:
    print(f"    {pca_path}")
except NameError:
    pass
