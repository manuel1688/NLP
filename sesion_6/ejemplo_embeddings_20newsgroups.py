
# Ejemplo: Clasificación de textos con embeddings semánticos y 20 Newsgroups
# -------------------------------------------------------------------------
# Este script muestra cómo usar SentenceTransformer para obtener
# embeddings semánticos a partir de los textos del dataset 20newsgroups,
# y luego entrenar un clasificador LogisticRegression sobre esos vectores.
# ------------------------------------------------------------------------- 



# Importa el dataset 20 Newsgroups, que contiene textos de 20 categorías temáticas en inglés.
from sklearn.datasets import fetch_20newsgroups
# Importa SentenceTransformer para convertir textos en vectores densos (embeddings) que capturan el significado semántico.
from sentence_transformers import SentenceTransformer
# Importa el clasificador lineal y la función para imprimir métricas de evaluación.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# 1. Cargar el dataset 20 Newsgroups
#    Descarga los datos de entrenamiento y prueba. Se eliminan encabezados, pies y citas para que el modelo se enfoque en el contenido del mensaje.
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


# 2. Obtener embeddings semánticos para cada texto
#    Se utiliza un modelo preentrenado ('all-MiniLM-L6-v2') para transformar textos en vectores de 384 dimensiones.
encoder = SentenceTransformer('all-MiniLM-L6-v2')
# Convierte los textos de entrenamiento en embeddings. El resultado es una matriz donde cada fila representa un texto y cada columna una dimensión del embedding.
X_train = encoder.encode(train.data, show_progress_bar=True)
# Convierte los textos de prueba en embeddings usando el mismo modelo.
X_test = encoder.encode(test.data, show_progress_bar=True)


# 3. Entrenar un clasificador
#    Se crea un clasificador de regresión logística y se entrena usando los embeddings de entrenamiento y sus etiquetas.
#    El parámetro max_iter=1000 asegura que el optimizador tenga suficientes iteraciones para converger.
clf = LogisticRegression(max_iter=100)
clf.fit(X_train, train.target)


# 4. Evaluar el modelo
#    Se predicen las categorías de los textos de prueba y se imprime un reporte con métricas de precisión, recall y F1-score para cada clase.
y_pred = clf.predict(X_test)
print(classification_report(test.target, y_pred, target_names=test.target_names))


# Notas:
# - Puedes comparar este enfoque con el uso de TfidfVectorizer para ver la diferencia en desempeño.
# - Los embeddings semánticos permiten capturar similitud de significado, no solo coincidencia de palabras.
# - Puedes probar otros modelos de SentenceTransformer según tus recursos y necesidades.
