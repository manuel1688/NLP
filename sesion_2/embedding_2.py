import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize

# --- 1. Corpus y Tokenización ---
CORPUS = "Juan es un estudiante. Maria es una estudiante. Juan estudia programacion, Maria estudia diseño."
all_tokens = word_tokenize(CORPUS)
vocab = sorted(set(t.lower() for t in all_tokens))

# --- 2. Definir "Zonas Semánticas" (Simulación de entrenamiento) ---
# Creamos puntos base para diferentes tipos de palabras
CENTROS = {
    "nombres": np.array([5, 5, 5]),      # Zona alta
    "puntuacion": np.array([-5, -5, -5]), # Zona baja
    "verbos": np.array([0, 5, 0]),       # Zona media lateral
    "otros": np.array([0, 0, 0])         # Centro
}

def asignar_categoria(token):
    if token in ["juan", "maria"]: return "nombres"
    if token in [".", ","]: return "puntuacion"
    if token in ["estudia", "es"]: return "verbos"
    return "otros"

# --- 3. Generar Embeddings "Agrupados" ---
EMBEDDING_DIM = 50
embeddings = {}
np.random.seed(42)

for token in vocab:
    cat = asignar_categoria(token)
    centro_base = np.zeros(EMBEDDING_DIM)
    # Ponemos el centro definido en las primeras 3 dimensiones
    centro_base[:3] = CENTROS[cat]
    
    # El "entrenamiento" es el centro + un poquito de ruido aleatorio
    ruido = np.random.randn(EMBEDDING_DIM) * 0.5 
    embeddings[token] = centro_base + ruido

# --- 4. PCA y Gráfica ---
matrix = np.array([embeddings[t] for t in vocab])
pca = PCA(n_components=3)
coords_3d = pca.fit_transform(matrix)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Colorear por categoría para facilitar la vista
colores = {"nombres": "red", "puntuacion": "blue", "verbos": "green", "otros": "gray"}

for i, token in enumerate(vocab):
    cat = asignar_categoria(token)
    ax.scatter(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], 
               c=colores[cat], s=100, edgecolors="black")
    ax.text(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], f" {token}")

ax.set_title("Simulación de Embeddings Entrenados (Agrupación Semántica)")
plt.show()