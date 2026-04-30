# Word2Vec Skip-gram Simplificado — Visualización de Convergencia
# ---------------------------------------------------------------
# Script didáctico: muestra cómo los embeddings "aprenden" relaciones semánticas.
# Entrena sobre un corpus pequeño y visualiza los vectores antes y después.

# === DEPENDENCIAS ===
import numpy as np  # Operaciones numéricas y matrices
import matplotlib.pyplot as plt  # Gráficas

# === 1. Corpus pequeño y simple ===
corpus = [
    "el gato come pescado fresco",
    "el perro come carne",
    "el gato duerme mucho",
    "el perro duerme poco",
    "el pescado es fresco",
    "la carne es sabrosa"
]

# === 2. Tokenización y vocabulario ===
# Divide frases en palabras y crea diccionario palabra <-> índice
oraciones = [frase.split() for frase in corpus]
palabras = [palabra for oracion in oraciones for palabra in oracion]
vocabulario = sorted(set(palabras))
palabra_a_indice = {palabra: i for i, palabra in enumerate(vocabulario)}
indice_a_palabra = {i: palabra for palabra, i in palabra_a_indice.items()}
V = len(vocabulario)

# === 3. Generar pares skip-gram ===
# Para cada palabra, crea pares (objetivo, contexto) usando una ventana deslizante
VENTANA = 2
pares = []
for oracion in oraciones:
    indices = [palabra_a_indice[w] for w in oracion]
    for i, objetivo in enumerate(indices):
        for j in range(max(0, i-VENTANA), min(len(indices), i+VENTANA+1)):
            if i != j:
                pares.append((objetivo, indices[j]))

# === 4. Inicializar embeddings ===
# Matrices aleatorias: cada palabra tiene un vector objetivo y uno de contexto
DIM_EMB = 2  # 2D para graficar
W_objetivo = np.random.normal(0, 0.1, (V, DIM_EMB))
W_contexto = np.random.normal(0, 0.1, (V, DIM_EMB))
W_objetivo_ini = W_objetivo.copy()

# === 5. Entrenamiento simple (sin negative sampling) ===
# Aprende a predecir palabras de contexto a partir de la palabra objetivo
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

TASA_APRENDIZAJE = 0.1
EPOCAS = 100
for epoca in range(EPOCAS):
    np.random.shuffle(pares)
    for t, c in pares:
        v_t = W_objetivo[t]  # Vector de la palabra objetivo
        scores = W_contexto @ v_t  # Producto con todos los contextos
        probs = softmax(scores)  # Probabilidades para cada palabra
        grad_salida = probs.copy()
        grad_salida[c] -= 1  # (predicción - realidad)
        # Actualiza vectores (descenso de gradiente)
        W_contexto -= TASA_APRENDIZAJE * np.outer(grad_salida, v_t)
        W_objetivo[t] -= TASA_APRENDIZAJE * (W_contexto.T @ grad_salida)
    if (epoca+1) % 20 == 0:
        print(f"Época {epoca+1}/{EPOCAS}")

# === 6. Visualización de embeddings antes y después ===
# Muestra cómo los vectores "se agrupan" según el contexto aprendido
plt.figure(figsize=(10,5))
for i, (mat, titulo) in enumerate(zip([W_objetivo_ini, W_objetivo], ["Antes de entrenar", "Después de entrenar"])):
    plt.subplot(1,2,i+1)
    plt.scatter(mat[:,0], mat[:,1], color='steelblue')
    for idx, palabra in indice_a_palabra.items():
        plt.text(mat[idx,0], mat[idx,1], palabra, fontsize=12)
    plt.title(titulo)
    # --- ZOOM AUTOMÁTICO ---
    x_min, x_max = mat[:,0].min(), mat[:,0].max()
    y_min, y_max = mat[:,1].min(), mat[:,1].max()
    x_margin = (x_max - x_min) * 0.2 if x_max > x_min else 1
    y_margin = (y_max - y_min) * 0.2 if y_max > y_min else 1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.axis('equal')
plt.suptitle("Embeddings Skip-gram — Convergencia Visual")
plt.tight_layout()
plt.show()
