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
sentences = [s.split() for s in corpus]
tokens = [w for s in sentences for w in s]
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)

print(f"Vocabulario: {vocab}\n")

# 3. Generar pares skip-gram
WINDOW = 2
pairs = []
for sent in sentences:
    idxs = [word2idx[w] for w in sent]
    for i, target in enumerate(idxs):
        for j in range(max(0, i-WINDOW), min(len(idxs), i+WINDOW+1)):
            if i != j:
                pairs.append((target, idxs[j]))

print(f"Ejemplo de pares (target -> contexto):")
for t, c in pairs[:5]:
    print(f"  '{idx2word[t]}' -> '{idx2word[c]}'")
print()

# 4. Inicializar embeddings
EMBED_DIM = 2  # Para visualizar directo en 2D
W_embed = np.random.normal(0, 0.1, (V, EMBED_DIM))
W_context = np.random.normal(0, 0.1, (V, EMBED_DIM))
W_embed_ini = W_embed.copy()

# 5. Ejemplo de forward pass antes de entrenar
print("[EJEMPLO] Forward pass antes de entrenar:")
example_t, example_c = pairs[0]
v_t = W_embed[example_t]
scores = W_context @ v_t
probs = np.exp(scores) / np.exp(scores).sum()
print(f"Target: '{idx2word[example_t]}' | Contexto real: '{idx2word[example_c]}'")
print("Probabilidades de contexto predichas:")
for idx, p in enumerate(probs):
    print(f"  {idx2word[idx]:10s}: {p:.3f}")
loss = -np.log(probs[example_c])
print(f"Cross-entropy loss para el par: {loss:.4f}\n")

# 6. Entrenamiento
LR = 0.1
EPOCHS = 100
print("[ENTRENAMIENTO]\n")
for epoch in range(EPOCHS):
    np.random.shuffle(pairs)
    total_loss = 0
    for t, c in pairs:
        # ---- FORWARD ----
        v_t = W_embed[t]
        scores = W_context @ v_t
        probs = np.exp(scores) / np.exp(scores).sum()
        # ---- LOSS ----
        loss = -np.log(probs[c])
        total_loss += loss
        # ---- BACKWARD ----
        grad_out = probs.copy()  # (predicción)
        grad_out[c] -= 1         # (predicción - realidad)
        # ---- UPDATE ----
        W_context -= LR * np.outer(grad_out, v_t)
        W_embed[t] -= LR * (W_context.T @ grad_out)
    if (epoch+1) % 20 == 0:
        print(f"Época {epoch+1:3d} | Loss promedio: {total_loss/len(pairs):.4f}")

# 7. Visualización de embeddings antes y después
plt.figure(figsize=(10,5))
for i, (mat, title) in enumerate(zip([W_embed_ini, W_embed], ["Antes de entrenar", "Después de entrenar"])):
    plt.subplot(1,2,i+1)
    plt.scatter(mat[:,0], mat[:,1], color='steelblue')
    for idx, w in idx2word.items():
        plt.text(mat[idx,0], mat[idx,1], w, fontsize=12)
    plt.title(title)
    plt.axis('equal')
plt.suptitle("Embeddings Skip-gram — Convergencia Visual")
plt.tight_layout()
plt.show()

# 8. Similitud coseno entre palabras
print("\n[SIMILITUD COSENO ENTRE PALABRAS]")
emb_norm = W_embed / (np.linalg.norm(W_embed, axis=1, keepdims=True) + 1e-8)
sim_matrix = cosine_similarity(emb_norm)
for idx, w in idx2word.items():
    sims = sim_matrix[idx]
    top_idx = np.argsort(sims)[::-1][1:4]  # top 3 (excluye sí mismo)
    vecinos = [(idx2word[i], sims[i]) for i in top_idx]
    vecinos_str = ", ".join(f"{v} ({s:.2f})" for v, s in vecinos)
    print(f"  {w:10s} → {vecinos_str}")

# 9. Mini test final: palabras más similares a "perro"
print("\n[TEST FINAL] Palabras más similares a 'perro':")
if "perro" in word2idx:
    idx = word2idx["perro"]
    sims = sim_matrix[idx]
    top_idx = np.argsort(sims)[::-1][1:4]
    for i in top_idx:
        print(f"  {idx2word[i]} ({sims[i]:.2f})")
else:
    print("  'perro' no está en el vocabulario.")
