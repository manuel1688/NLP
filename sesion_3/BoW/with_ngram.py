from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Corpus de ejemplo
textos = [
    "el aprendizaje automatico es genial",
    "el aprendizaje profundo es genial"
]

# Inicializamos CountVectorizer para Trigramas (3-grams)
# ngram_range=(3, 3) captura grupos de exactamente 3 palabras
vectorizador_bow = CountVectorizer(ngram_range=(3, 3))

# Creamos la matriz de conteos
matriz_bow = vectorizador_bow.fit_transform(textos)

# Lo convertimos a DataFrame para verlo claro
df_bow = pd.DataFrame(
    matriz_bow.toarray(), 
    columns=vectorizador_bow.get_feature_names_out(),
    index=["Frase 1", "Frase 2"]
)

print(df_bow)