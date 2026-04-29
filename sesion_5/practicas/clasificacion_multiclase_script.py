
# Este script implementa una clasificación multiclase de reseñas en español usando embeddings semánticos.
# Los embeddings semánticos permiten representar textos como vectores densos que capturan el significado y la similitud semántica.
# Se utiliza SentenceTransformer para obtener los embeddings y LogisticRegression (estrategia One-vs-Rest) como clasificador.

import json
from sentence_transformers import SentenceTransformer  # Para embeddings semánticos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np


# Definición de las clases posibles y semilla para reproducibilidad
CLASES = ['peliculas', 'restaurantes', 'productos', 'servicios', 'hoteles']
RANDOM_STATE = 42

# 1. Cargar el corpus desde un archivo JSON
def cargar_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extraemos los textos y las categorías de cada review
    reviews = [item['texto'] for item in data['reviews']]
    categorias = [item['categoria'] for item in data['reviews']]
    return reviews, categorias

# 2. Separar en train/test de forma estratificada
def split_datos(reviews, categorias):
    # train_test_split con stratify asegura que la proporción de clases se mantenga en ambos conjuntos
    return train_test_split(
        reviews, categorias, test_size=0.3, random_state=RANDOM_STATE, stratify=categorias
    )

# 3. Embeddings semánticos y entrenamiento del modelo
def entrenar_modelo(reviews_train, y_train):
    # Cargamos un modelo preentrenado de SentenceTransformer (multilingüe)
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Obtenemos los embeddings para cada texto de entrenamiento
    X_train = encoder.encode(reviews_train, show_progress_bar=True)
    # Normalizamos los embeddings para evitar problemas numéricos
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    # Creamos el clasificador LogisticRegression (sin multi_class, ya que OvR es el valor por defecto)
    modelo = LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=1000)
    modelo.fit(X_train_norm, y_train)
    # Retornamos el encoder, el scaler y el modelo entrenado
    return encoder, scaler, modelo

# 4. Evaluación del modelo usando sklearn
def evaluar_modelo(modelo, scaler, encoder, reviews_test, y_test):
    # Convertimos los textos de test a embeddings usando el mismo encoder
    X_test = encoder.encode(reviews_test, show_progress_bar=True)
    # Normalizamos los embeddings de test con el mismo scaler
    X_test_norm = scaler.transform(X_test)
    # Predecimos las categorías usando el modelo entrenado
    y_pred = modelo.predict(X_test_norm)
    # Mostramos métricas estándar de clasificación
    print(classification_report(y_test, y_pred))
    return y_pred

# 5. Funciones para métricas manuales (útil para entender el desempeño por clase)
def confusion_matrix_manual(y_true, y_pred, pos_label='positivo'):
    TP = sum(t == pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    TN = sum(t != pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    FP = sum(t != pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    FN = sum(t == pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_manual(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred, pos_label='positivo'):
    TP, _, FP, _ = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred, pos_label='positivo'):
    TP, _, _, FN = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1(y_true, y_pred, pos_label='positivo'):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# Matriz de confusión personalizada para problemas multiclase
def confusion_matrix_multiclase(y_true, y_pred, clases):
    ancho = 14
    cabecera = ' ' * ancho + ''.join(c[:ancho].ljust(ancho) for c in clases)
    print(cabecera)
    print('-' * (ancho * (len(clases) + 1)))
    for real in clases:
        fila = real[:ancho].ljust(ancho)
        for pred in clases:
            count = sum(t == real and p == pred for t, p in zip(y_true, y_pred))
            fila += str(count).ljust(ancho)
        print(fila)

# 6. Métricas manuales por clase (útil para análisis detallado)
def metricas_por_clase(y_test, y_pred):
    print(f"{'Categoría':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print('-' * 48)
    f1_por_clase = []
    for c in CLASES:
        p = precision(y_test, y_pred, pos_label=c)
        r = recall(y_test, y_pred, pos_label=c)
        f = f1(y_test, y_pred, pos_label=c)
        f1_por_clase.append(f)
        print(f'{c:<15} {p:>10.2f} {r:>10.2f} {f:>10.2f}')
    print('-' * 48)
    macro_f1 = sum(f1_por_clase) / len(f1_por_clase)
    print(f"{'Macro F1':<15} {'':>10} {'':>10} {macro_f1:>10.2f}")
    print()
    print(f'Accuracy global: {accuracy(y_test, y_pred):.2f}')

# 7. Mostrar matriz de confusión multiclase
def mostrar_matriz_confusion(y_test, y_pred):
    confusion_matrix_multiclase(y_test, y_pred, CLASES)

# 8. Predicción sobre textos nuevos usando embeddings semánticos
def predecir_nuevos(modelo, encoder):
    nuevas_reviews = [
        'La actuación fue brillante y la historia muy emotiva',
        'El sushi estaba fresco y el servicio impecable',
        'El producto llegó en perfectas condiciones y funciona genial',
        'Tardaron semanas en responder y no solucionaron el problema',
        'Habitación limpia, cama cómoda y muy buena ubicación',
    ]
    # Convertimos los nuevos textos a embeddings y predecimos la categoría
    preds = modelo.predict(encoder.encode(nuevas_reviews))
    for texto, pred in zip(nuevas_reviews, preds):
        print(f'  [{pred.upper():<13}]  {texto}')

if __name__ == "__main__":
    # Cargar y explorar el corpus
    reviews, categorias = cargar_corpus("corpus_sentimiento_reviews.json")
    print(f'Total de reseñas: {len(reviews)}')
    print('Ejemplos por categoría:')
    for c in CLASES:
        print(f'  {c:<15}: {categorias.count(c):>3}')
    # Separar en train/test
    reviews_train, reviews_test, y_train, y_test = split_datos(reviews, categorias)
    print(f'Train : {len(reviews_train)} reseñas')
    print(f'Test  : {len(reviews_test)} reseñas')
    # Entrenar modelo con embeddings semánticos y normalización
    encoder, scaler, modelo = entrenar_modelo(reviews_train, y_train)
    print('Modelo entrenado.')
    print(f'Clases aprendidas : {modelo.classes_.tolist()}')
    print(f'Dimensión embeddings: {encoder.encode([reviews_train[0]]).shape[1]}')
    # Evaluar modelo
    y_pred = evaluar_modelo(modelo, scaler, encoder, reviews_test, y_test)
    # Métricas manuales y matriz de confusión
    metricas_por_clase(y_test, y_pred)
    mostrar_matriz_confusion(y_test, y_pred)
    # Predicción sobre textos nuevos
    predecir_nuevos(modelo, encoder)
