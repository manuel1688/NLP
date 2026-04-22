import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

with open("corpus_sentimiento_reviews.json", encoding="utf-8") as f:
    data = json.load(f)

reviews      = [item["texto"]       for item in data["reviews"]]
sentimientos = [item["sentimiento"] for item in data["reviews"]]

vectorizador = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
X = vectorizador.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(
    X, sentimientos, test_size=0.3, random_state=42
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, list(modelo.classes_).index("positivo")]
y_bin   = [1 if y == "positivo" else 0 for y in y_test]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Matriz de confusión ---
cm = confusion_matrix(y_test, y_pred, labels=["positivo", "negativo"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["positivo", "negativo"],
            yticklabels=["positivo", "negativo"])
axes[0].set_title("Matriz de Confusión")
axes[0].set_xlabel("Predicho")
axes[0].set_ylabel("Real")

# --- Curva ROC ---
fpr, tpr, _ = roc_curve(y_bin, y_proba)
auc = roc_auc_score(y_bin, y_proba)
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.2f}")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("Curva ROC")
axes[1].set_xlabel("FPR (falsos positivos)")
axes[1].set_ylabel("TPR / Recall")
axes[1].legend()

# --- Curva Precision-Recall ---
precision, recall, _ = precision_recall_curve(y_bin, y_proba)
ap = average_precision_score(y_bin, y_proba)
axes[2].plot(recall, precision, label=f"AP = {ap:.2f}")
axes[2].set_title("Curva Precision-Recall")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend()

plt.tight_layout()
plt.show()
