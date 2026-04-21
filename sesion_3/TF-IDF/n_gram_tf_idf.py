from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# 1. Definimos nuestros datos
corpus = [
    "el cine es bueno",
    "el cine no es bueno"
]

# --- PARTE 1: Bag of Words (BoW) con Bigramas ---
# ngram_range=(2, 2) significa que solo queremos pares de palabras
bow_vectorizer = CountVectorizer(ngram_range=(2, 2))
bow_matrix = bow_vectorizer.fit_transform(corpus)

# --- PARTE 2: TF-IDF con Bigramas ---
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 2. Visualizamos los resultados en DataFrames para que sea legible
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("--- Representación BoW (Conteos) ---")
print(df_bow)
print("\n--- Representación TF-IDF (Pesos) ---")
print(df_tfidf)