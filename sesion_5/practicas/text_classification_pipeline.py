from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Carga y exploración de datos
def explore_data(data):
    print(f"Total documentos: {len(data.data)}")
    print(f"Clases: {data.target_names}")
    print(f"Ejemplo:\n{data.data[0][:500]}")

# 2. Preprocesamiento y vectorización
def preprocess_texts(train_texts, test_texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 3. Selección de modelo
def get_model():
    # LogisticRegression en scikit-learn implementa por defecto la estrategia One-vs-Rest (OvR)
    # para clasificación multiclase. Esto significa que entrena un clasificador binario por cada clase:
    # para N clases, se entrenan N clasificadores, cada uno distingue una clase vs el resto.
    # Puedes forzar explícitamente multi_class='ovr' si lo deseas.
    return LogisticRegression(max_iter=1000, multi_class='ovr')

# 4. Entrenamiento y evaluación
def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(xticks_rotation=90)
    plt.show()

def main():
    # Cargar datos
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    explore_data(train)

    # Preprocesamiento
    X_train, X_test, vectorizer = preprocess_texts(train.data, test.data)

    # Modelo
    model = get_model()
    # Entrenamiento del modelo:
    # Aquí se entrena un clasificador OvR por cada clase (20 en 20newsgroups),
    # aunque el usuario solo llama fit() una vez. El modelo se encarga internamente
    # de crear los clasificadores binarios y combinarlos para predicción multiclase.
    model.fit(X_train, train.target)

    # Evaluación
    evaluate_model(model, X_test, test.target, train.target_names)

if __name__ == "__main__":
    main()

# ---------------------------------------------
# ¿Qué significan precision, recall, f1-score y support?
#
# - precision: De todas las predicciones positivas para una clase, ¿cuántas fueron correctas?
#   (TP / (TP + FP))
# - recall: De todos los ejemplos reales de una clase, ¿cuántos fueron correctamente identificados?
#   (TP / (TP + FN))
# - f1-score: Media armónica entre precision y recall. Resume ambos en un solo valor.
#   (2 * precision * recall) / (precision + recall)
# - support: Número de muestras reales de esa clase en el conjunto de test.
#
# accuracy: Proporción de aciertos globales sobre todas las clases.
# macro avg: Promedio simple de las métricas para todas las clases (no pondera por tamaño de clase).
# weighted avg: Promedio ponderado por el número de muestras de cada clase (support).
#
# Ejemplo de interpretación:
#   - Una precision de 0.80 en 'comp.windows.x' significa que el 80% de las predicciones para esa clase fueron correctas.
#   - Un recall de 0.71 en esa clase significa que el 71% de los textos realmente pertenecientes a esa clase fueron detectados.
#   - El f1-score resume ambos valores.
#   - support indica cuántos ejemplos reales había de esa clase.
#
# Para más detalles: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
