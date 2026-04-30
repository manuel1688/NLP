# Word2Vec Skip-gram Pedagógico — Visualización y Explicación Paso a Paso
# -----------------------------------------------------------------------
# Objetivo: Predecir palabras de contexto dado un target (Skip-gram)
# El código muestra y explica cada paso del aprendizaje, visualiza embeddings y similitudes.

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# 1. Corpus pequeño y simple
corpus = [
    "el gato come pescado fresco",
    "el perro come carne",
    "el gato duerme mucho",
    "el perro duerme poco",
    "el pescado es fresco",
    "la carne es sabrosa"
]

print("\n[OBJETIVO] Entrenar embeddings Skip-gram para predecir palabras de contexto dado un target.\n")

# 2. Tokenización y vocabulario
oraciones = [frase.split() for frase in corpus]
palabras = [palabra for oracion in oraciones for palabra in oracion]
vocabulario = sorted(set(palabras))
palabra_a_indice = {palabra: i for i, palabra in enumerate(vocabulario)}
indice_a_palabra = {i: palabra for palabra, i in palabra_a_indice.items()}
V = len(vocabulario)

print(f"Vocabulario: {vocabulario}\n")

# 3. Generar pares skip-gram
VENTANA = 2
pares = []
for oracion in oraciones:
    indices = [palabra_a_indice[w] for w in oracion]
    for i, objetivo in enumerate(indices):
        for j in range(max(0, i-VENTANA), min(len(indices), i+VENTANA+1)):
            if i != j:
                pares.append((objetivo, indices[j]))

print(f"Ejemplo de pares (objetivo -> contexto):")
for t, c in pares[:5]:
    print(f"  '{indice_a_palabra[t]}' -> '{indice_a_palabra[c]}'")
print()

# 4. Inicializar embeddings
DIM_EMB = 2  # Para visualizar directo en 2D
W_objetivo = np.random.normal(0, 0.1, (V, DIM_EMB))
W_contexto = np.random.normal(0, 0.1, (V, DIM_EMB))
W_objetivo_ini = W_objetivo.copy()

# 5. Ejemplo de forward pass antes de entrenar
print("[EJEMPLO] Forward pass antes de entrenar:")
ejemplo_t, ejemplo_c = pares[0]
v_t = W_objetivo[ejemplo_t]
scores = W_contexto @ v_t
probs = np.exp(scores) / np.exp(scores).sum()
print(f"Objetivo: '{indice_a_palabra[ejemplo_t]}' | Contexto real: '{indice_a_palabra[ejemplo_c]}'")
print("Probabilidades de contexto predichas:")
for idx, p in enumerate(probs):
    print(f"  {indice_a_palabra[idx]:10s}: {p:.3f}")
loss = -np.log(probs[ejemplo_c])
print(f"Cross-entropy loss para el par: {loss:.4f}\n")

# 6. Entrenamiento
TASA_APRENDIZAJE = 0.1
EPOCAS = 100
print("[ENTRENAMIENTO]\n")
for epoca in range(EPOCAS):
    np.random.shuffle(pares)
    perdida_total = 0
    for t, c in pares:
        # ---- FORWARD ----
        v_t = W_objetivo[t]
        scores = W_contexto @ v_t
        probs = np.exp(scores) / np.exp(scores).sum()
        # ---- LOSS ----
        perdida = -np.log(probs[c])
        perdida_total += perdida
        # ---- BACKWARD ----
        grad_salida = probs.copy()  # (predicción)
        grad_salida[c] -= 1         # (predicción - realidad)
        # ---- UPDATE ----
        W_contexto -= TASA_APRENDIZAJE * np.outer(grad_salida, v_t)
        W_objetivo[t] -= TASA_APRENDIZAJE * (W_contexto.T @ grad_salida)
    if (epoca+1) % 20 == 0:
        print(f"Época {epoca+1:3d} | Pérdida promedio: {perdida_total/len(pares):.4f}")

# 7. Visualización de embeddings antes y después
plt.figure(figsize=(10,5))
for i, (mat, titulo) in enumerate(zip([W_objetivo_ini, W_objetivo], ["Antes de entrenar", "Después de entrenar"])):
    plt.subplot(1,2,i+1)
    plt.scatter(mat[:,0], mat[:,1], color='steelblue')
    for idx, palabra in indice_a_palabra.items():
        plt.text(mat[idx,0], mat[idx,1], palabra, fontsize=12)
    plt.title(titulo)
    plt.axis('equal')
plt.suptitle("Embeddings Skip-gram — Convergencia Visual")
plt.tight_layout()
plt.show()

# 8. Similitud coseno entre palabras
print("\n[SIMILITUD COSENO ENTRE PALABRAS]")
emb_norm = W_objetivo / (np.linalg.norm(W_objetivo, axis=1, keepdims=True) + 1e-8)
matriz_sim = cosine_similarity(emb_norm)
for idx, palabra in indice_a_palabra.items():
    sims = matriz_sim[idx]
    top_idx = np.argsort(sims)[::-1][1:4]  # top 3 (excluye sí mismo)
    vecinos = [(indice_a_palabra[i], sims[i]) for i in top_idx]
    vecinos_str = ", ".join(f"{v} ({s:.2f})" for v, s in vecinos)
    print(f"  {palabra:10s} → {vecinos_str}")

# 9. Mini test final: palabras más similares a "perro"
print("\n[TEST FINAL] Palabras más similares a 'perro':")
if "perro" in palabra_a_indice:
    idx = palabra_a_indice["perro"]
    sims = matriz_sim[idx]
    top_idx = np.argsort(sims)[::-1][1:4]
    for i in top_idx:
        print(f"  {indice_a_palabra[i]} ({sims[i]:.2f})")
else:
    print("  'perro' no está en el vocabulario.")
