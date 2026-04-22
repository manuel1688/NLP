import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

with open("corpus_sentimiento_reviews.json", encoding="utf-8") as f:
    data = json.load(f)

reviews      = [item["texto"]       for item in data["reviews"]]
sentimientos = [item["sentimiento"] for item in data["reviews"]]

vectorizador = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
X = vectorizador.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(
    X, sentimientos, test_size=0.3, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# predicción sobre ejemplos nuevos
nuevas_reviews = [
    "Me gustó mucho, muy entretenida",
    "Qué porquería, no la terminen de ver"
]
print(modelo.predict(vectorizador.transform(nuevas_reviews)))


# --- métricas desde cero ---

def confusion_matrix_manual(y_true, y_pred, pos_label="positivo"):
    TP = sum(t == pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    TN = sum(t != pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    FP = sum(t != pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    FN = sum(t == pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_manual(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred, pos_label="positivo"):
    TP, _, FP, _ = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred, pos_label="positivo"):
    TP, _, _, FN = confusion_matrix_manual(y_true, y_pred, pos_label)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1(y_true, y_pred, pos_label="positivo"):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


TP, TN, FP, FN = confusion_matrix_manual(y_test, y_pred)
print(f"\n--- métricas manuales ---")
print(f"Matriz de confusión: TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"Accuracy:  {accuracy(y_test, y_pred):.2f}")
print(f"Precision: {precision(y_test, y_pred):.2f}")
print(f"Recall:    {recall(y_test, y_pred):.2f}")
print(f"F1-score:  {f1(y_test, y_pred):.2f}")
