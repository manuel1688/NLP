from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# 1. Definimos las frases
# Nota cómo la Frase 1 y 2 tienen las mismas palabras pero en distinto orden
corpus = [
    "el cine es bueno",
    "bueno es el cine",
    "el cine no es bueno"
]

# 2. Inicializamos el CountVectorizer (por defecto usa unigramas)
# Al no poner ngram_range, es equivalente a ngram_range=(1, 1)
vectorizador = CountVectorizer()

# 3. Creamos la matriz
X = vectorizador.fit_transform(corpus)

# 4. Lo pasamos a un DataFrame para comparar
df = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())

print("Representación BoW con Unigramas:")
print(df)