# ================================================================
# Embedding Visualization — Tokens sin entrenamiento
# ================================================================
# Toma los tokens del corpus, les asigna vectores ALEATORIOS
# (embedding no entrenado) y los grafica en 2D con PCA.
#
# Objetivo: ver que sin entrenamiento los tokens se ubican
# sin ningun patron semantico.
#
# Dependencias: pip3 install matplotlib numpy scikit-learn nltk
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registra la proyeccion 3D
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize

from tokenization import CORPUS

# ---------------------------------------------------------------
# 1. Tokenizar y construir vocabulario unico
# ---------------------------------------------------------------
all_tokens = word_tokenize(CORPUS)
vocab = sorted(set(t.lower() for t in all_tokens))

print("=== Tokens unicos ===")
print(vocab)
print(f"Total tokens: {len(all_tokens)}  |  Vocabulario: {len(vocab)}")

# ---------------------------------------------------------------
# 2. Asignar un vector aleatorio a cada token (dim=50)
# ---------------------------------------------------------------
EMBEDDING_DIM = 50
np.random.seed(42)

# Diccionario: token -> vector aleatorio
embeddings = {token: np.random.randn(EMBEDDING_DIM) for token in vocab}

print(f"\n=== Embedding aleatorio (dim={EMBEDDING_DIM}, primeros 5 valores) ===")
for token, vec in embeddings.items():
    print(f"  {token:>15s} → [{', '.join(f'{v:.2f}' for v in vec[:5])}, ...]")

# ---------------------------------------------------------------
# 3. Reducir de 50D a 3D con PCA
# ---------------------------------------------------------------
matrix = np.array([embeddings[t] for t in vocab])
pca = PCA(n_components=3)
coords_3d = pca.fit_transform(matrix)

# ---------------------------------------------------------------
# 4. Graficar en 3D
# ---------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    coords_3d[:, 0],
    coords_3d[:, 1],
    coords_3d[:, 2],
    c="steelblue",
    s=80,
    edgecolors="black",
    linewidths=0.5,
)

for i, token in enumerate(vocab):
    ax.text(
        coords_3d[i, 0],
        coords_3d[i, 1],
        coords_3d[i, 2],
        f"  {token}",
        fontsize=9,
    )

ax.set_title("Embedding aleatorio (sin entrenamiento) — PCA 3D", fontsize=14)
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.set_zlabel("Componente 3")

output_path = "embedding_random.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n=== Grafica guardada en: {output_path} ===")
print("(Puedes rotar la grafica con el mouse al ejecutar el script)")

# ---------------------------------------------------------------
# 5. Explicacion
# ---------------------------------------------------------------
print("\n=== Explicacion ===")
print("Los vectores fueron asignados al azar (np.random.randn).")
print("Por eso los tokens aparecen dispersos sin agrupacion semantica:")
print("  - 'name' y 'student' no estan cerca aunque sean sustantivos.")
print("  - ',' y '.' no se agrupan aunque ambos sean puntuacion.")
print("Un modelo ENTRENADO (Word2Vec, GloVe, etc.) ubicaria tokens")
print("similares en regiones cercanas del espacio.")

plt.show()
