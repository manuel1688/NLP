

# ===============================
# Ejemplo: Clasificación de textos con embeddings semánticos y 20 Newsgroups
# ===============================
# Objetivo: Usar SentenceTransformer para obtener embeddings semánticos y entrenar un clasificador simple.
# Comentarios breves y claros para estudiantes.

# --- DEPENDENCIAS ---
from sklearn.datasets import fetch_20newsgroups  # Dataset de textos
from sentence_transformers import SentenceTransformer  # Modelo de embeddings
from sklearn.linear_model import LogisticRegression  # Clasificador lineal
from sklearn.metrics import classification_report  # Métricas de evaluación



# --- 1. Cargar el dataset 20 Newsgroups ---
# Descarga textos de 20 categorías para entrenar y probar
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
print(f"Categorías: {train.target_names}")
print(f"Ejemplos de entrenamiento: {len(train.data)} | Ejemplos de prueba: {len(test.data)}")



# --- 2. Obtener embeddings semánticos para cada texto ---
# Usa modelo preentrenado para transformar textos en vectores
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_train = encoder.encode(train.data, show_progress_bar=True)
X_test = encoder.encode(test.data, show_progress_bar=True)
print(f"Shape embeddings entrenamiento: {X_train.shape}")
print(f"Shape embeddings prueba: {X_test.shape}")



# --- 3. Entrenar un clasificador ---
# Entrena un clasificador lineal sobre los embeddings
clf = LogisticRegression(max_iter=100)
clf.fit(X_train, train.target)



# --- 4. Evaluar el modelo ---
# Imprime métricas de desempeño sobre el conjunto de prueba
y_pred = clf.predict(X_test)
print(classification_report(test.target, y_pred, target_names=test.target_names))



# Notas:
# - Puedes comparar este enfoque con el uso de TfidfVectorizer para ver la diferencia en desempeño.
# - Los embeddings semánticos permiten capturar similitud de significado, no solo coincidencia de palabras.
# - Puedes probar otros modelos de SentenceTransformer según tus recursos y necesidades.
