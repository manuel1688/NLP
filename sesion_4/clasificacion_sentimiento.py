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
