
# Ejemplo: Clasificación de textos con embeddings semánticos y 20 Newsgroups usando PyTorch
# -------------------------------------------------------------------------
# Script didáctico: muestra cómo usar embeddings y un clasificador simple.
# Usa un modelo preentrenado para vectorizar textos y entrena una red neuronal.
# -------------------------------------------------------------------------

# === DEPENDENCIAS ===
import torch  # Operaciones con tensores y redes neuronales
from transformers import AutoTokenizer, AutoModel  # Modelos y tokenización de HuggingFace
from sklearn.datasets import fetch_20newsgroups  # Dataset de textos
from sklearn.metrics import classification_report  # Métricas de evaluación
import numpy as np  # Operaciones numéricas
import torch.nn as nn
import torch.optim as optim

# === 1. Cargar el dataset 20 Newsgroups ===
# Descarga textos de 20 categorías para entrenar y probar
entrenamiento = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
prueba = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# === 2. Tokenizar y obtener embeddings con un modelo de HuggingFace ===
# Convierte textos en vectores numéricos usando un modelo preentrenado
NOMBRE_MODELO = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizador = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
modelo = AutoModel.from_pretrained(NOMBRE_MODELO)
modelo.eval()

y_train = entrenamiento.target
y_test = prueba.target

# Función: convierte una lista de textos en una matriz de embeddings
def obtener_embeddings(textos, tam_lote=32):
    todos_embeddings = []
    with torch.no_grad():
        for i in range(0, len(textos), tam_lote):
            lote = textos[i:i+tam_lote]
            codificado = tokenizador(lote, padding=True, truncation=True, return_tensors='pt', max_length=128)
            salidas = modelo(**codificado)
            # Promedia los vectores de cada palabra del texto
            embeddings = salidas.last_hidden_state.mean(dim=1)
            todos_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(todos_embeddings)

y_entrenamiento = entrenamiento.target
y_prueba = prueba.target
# Obtiene los embeddings para entrenamiento y prueba
X_entrenamiento = obtener_embeddings(entrenamiento.data)
X_prueba = obtener_embeddings(prueba.data)
y_entrenamiento = entrenamiento.target
y_prueba = prueba.target

dimension_entrada = X_entrenamiento.shape[1]
# === 3. Clasificador simple en PyTorch (una capa lineal) ===
# Red neuronal mínima: solo una capa lineal
class ClasificadorSimple(nn.Module):
    def __init__(self, dimension_entrada, num_clases):
        super().__init__()
        self.lineal = nn.Linear(dimension_entrada, num_clases)
    def forward(self, x):
        return self.lineal(x)

dimension_entrada = X_entrenamiento.shape[1]
num_clases = len(entrenamiento.target_names)
clasificador = ClasificadorSimple(dimension_entrada, num_clases)

# === 4. Entrenamiento del clasificador ===
# Ajusta los pesos para minimizar el error de clasificación
EPOCAS = 10
TAM_LOTE = 64
optimizador = optim.Adam(clasificador.parameters(), lr=1e-3)
funcion_perdida = nn.CrossEntropyLoss()

X_entrenamiento_tensor = torch.tensor(X_entrenamiento, dtype=torch.float32)
y_entrenamiento_tensor = torch.tensor(y_entrenamiento, dtype=torch.long)

for epoca in range(EPOCAS):
    clasificador.train()
    permutacion = torch.randperm(X_entrenamiento_tensor.size(0))
    for i in range(0, X_entrenamiento_tensor.size(0), TAM_LOTE):
        indices = permutacion[i:i+TAM_LOTE]
        lote_x, lote_y = X_entrenamiento_tensor[indices], y_entrenamiento_tensor[indices]
        optimizador.zero_grad()
        salidas = clasificador(lote_x)
        perdida = funcion_perdida(salidas, lote_y)
        perdida.backward()
        optimizador.step()
    print(f"Época {epoca+1}/{EPOCAS}, Pérdida: {perdida.item():.4f}")

# === 5. Evaluación ===
# Calcula métricas de desempeño sobre el conjunto de prueba
clasificador.eval()
X_prueba_tensor = torch.tensor(X_prueba, dtype=torch.float32)
with torch.no_grad():
    logits = clasificador(X_prueba_tensor)
    y_predicho = logits.argmax(dim=1).cpu().numpy()
print(classification_report(y_prueba, y_predicho, target_names=prueba.target_names))
