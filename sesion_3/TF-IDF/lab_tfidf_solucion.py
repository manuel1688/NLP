# ================================================================
# Lab 04 — TF-IDF — Script de referencia
# Corpus: eswiki_corpus.txt (Wikipedia ES)
# ================================================================
# Prerequisito: Lab 03 (BoW) — reutilizamos la función limpiar()
# y la lista STOPWORDS_ES del lab anterior.
# ================================================================
# Descomenta cada bloque cuando llegues a esa parte del lab_tfidf.md
# Ejecuta con: python3 lab_tfidf_solucion.py
# ================================================================

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "..", "eswiki_corpus.txt")

# ----------------------------------------------------------------
# Stop words en español (misma lista que lab_bow_solucion.py)
# ----------------------------------------------------------------
STOPWORDS_ES = {
    'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como',
    'con', 'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde',
    'durante', 'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre',
    'era', 'es', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estas',
    'este', 'estos', 'fue', 'ha', 'hay', 'he', 'la', 'las', 'le', 'les',
    'lo', 'los', 'mas', 'me', 'mi', 'muy', 'ni', 'no', 'nos', 'o', 'os',
    'para', 'pero', 'por', 'que', 'se', 'si', 'sin', 'sobre', 'son',
    'su', 'sus', 'tambien', 'tanto', 'te', 'tiene', 'tu', 'tus',
    'un', 'una', 'uno', 'unos', 'unas', 'y', 'ya', 'yo', 'ser', 'han',
    'sido', 'esta', 'este', 'parte', 'form', 'como', 'dicho', 'tras',
}

# ----------------------------------------------------------------
# Función de limpieza (misma que lab_bow_solucion.py)
# ----------------------------------------------------------------
def limpiar(texto):
    texto = re.sub(r'&lt;.*?&gt;', ' ', texto)
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = re.sub(r'\[\[.*?\]\]', ' ', texto)
    texto = re.sub(r'\{\{.*?\}\}', ' ', texto)
    texto = re.sub(r"'{2,}", '', texto)
    texto = re.sub(r'&\w+;', ' ', texto)
    texto = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', ' ', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', ' ', texto).strip().lower()
    return texto


# ================================================================
# CARGA Y LIMPIEZA DEL CORPUS
# ================================================================
print("=" * 60)
print("Cargando corpus eswiki_corpus.txt ...")
print("=" * 60)

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    lineas_raw = [l.strip() for l in f if l.strip()]

oraciones_limpias = [limpiar(l) for l in lineas_raw[:50]]
oraciones_limpias = [o for o in oraciones_limpias if len(o.split()) > 3]

print(f"  Oraciones disponibles: {len(oraciones_limpias)}")
print()


# ================================================================
# [PARTE 1] — El problema de BoW: stop words dominan el top-10
# ================================================================
print("=" * 60)
print("[PARTE 1] Top-10 BoW sin stop words (el problema)")
print("=" * 60)

bow_raw = CountVectorizer(max_features=500)
X_raw   = bow_raw.fit_transform(oraciones_limpias)

conteo  = X_raw.toarray().sum(axis=0)
palabras = bow_raw.get_feature_names_out()
top10   = sorted(zip(palabras, conteo), key=lambda x: -x[1])[:10]

print("  Palabras más frecuentes (sin filtrar stop words):")
for p, f in top10:
    print(f"    {p:<20} {int(f):>5} apariciones")
print()
print("  → ¿Alguna de estas palabras describe el TEMA de los documentos?")
print()


# ================================================================
# VERIFICACIÓN DE LA PARTE 2 — Mini corpus manual
# ================================================================
print("=" * 60)
print("Verificación Parte 2 — TF-IDF mini corpus (3 docs)")
print("=" * 60)

mini_corpus = [
    "andorra es un estado pequeño",
    "andorra tiene capital en andorra",
    "españa es un estado europeo",
]

tfidf_mini = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None)
X_mini     = tfidf_mini.fit_transform(mini_corpus)
df_mini    = pd.DataFrame(
    X_mini.toarray().round(3),
    columns=tfidf_mini.get_feature_names_out(),
    index=["D1", "D2", "D3"],
)

print("  Matriz TF-IDF (sin suavizado, sin normalización):")
print(df_mini.to_string())
print()
print("  Nota: sklearn calcula IDF = ln((N+1)/(df+1)) + 1 (con suavizado)")
print("  Los valores son distintos al cálculo manual pero el ranking es el mismo.")
print()

# Palabra más característica de cada documento
print("  Palabra más característica por documento:")
for doc_idx, doc_nombre in enumerate(["D1", "D2", "D3"]):
    row = df_mini.iloc[doc_idx]
    palabra_max = row.idxmax()
    valor_max   = row.max()
    print(f"    {doc_nombre}: '{palabra_max}' (TF-IDF = {valor_max:.3f})")
print()


# ================================================================
# [PARTE 3a] — TfidfVectorizer sobre el corpus real
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 3a

# tfidf = TfidfVectorizer(max_features=500, stop_words=list(STOPWORDS_ES))
# X_tfidf = tfidf.fit_transform(oraciones_limpias)
#
# print("=" * 60)
# print(f"[PARTE 3a] Matriz TF-IDF — forma: {X_tfidf.shape}")
# print("=" * 60)
# print(f"  Filas    = documentos : {X_tfidf.shape[0]}")
# print(f"  Columnas = palabras   : {X_tfidf.shape[1]}")
# print()


# ================================================================
# [PARTE 3b] — Top-5 palabras más características por documento
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 3b
# (requiere haber ejecutado el bloque PARTE 3a)

# df_tfidf = pd.DataFrame(X_tfidf.toarray(),
#                         columns=tfidf.get_feature_names_out())
#
# print("[PARTE 3b] Top-5 palabras más características por documento")
# print("-" * 60)
# for i in range(min(3, len(df_tfidf))):
#     row      = df_tfidf.iloc[i]
#     top5     = row.nlargest(5)
#     palabras = "  ,  ".join([f"{p} ({v:.3f})" for p, v in top5.items()])
#     print(f"  Doc {i}: {palabras}")
# print()


# ================================================================
# [PARTE 4] — BoW vs TF-IDF: comparación directa
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 4
# (requiere haber ejecutado el bloque PARTE 3a)

# bow_sw    = CountVectorizer(max_features=500, stop_words=list(STOPWORDS_ES))
# X_bow_sw  = bow_sw.fit_transform(oraciones_limpias)
# df_bow_sw = pd.DataFrame(X_bow_sw.toarray(),
#                          columns=bow_sw.get_feature_names_out())
#
# print("=" * 60)
# print("[PARTE 4] BoW vs TF-IDF — mismo documento, top-10")
# print("=" * 60)
#
# DOC_IDX = 0   # cambia este número para ver otro documento
#
# top_bow   = df_bow_sw.iloc[DOC_IDX].nlargest(10)
# top_tfidf = df_tfidf.iloc[DOC_IDX].nlargest(10)
#
# print(f"  Documento {DOC_IDX} — Top 10 palabras\n")
# print(f"  {'BoW (conteos)':<25} {'TF-IDF (pesos)'}")
# print(f"  {'-'*24} {'-'*24}")
# for (p_bow, v_bow), (p_tfidf, v_tfidf) in zip(top_bow.items(), top_tfidf.items()):
#     print(f"  {p_bow:<20} {int(v_bow):>3}     {p_tfidf:<20} {v_tfidf:.3f}")
# print()


# ================================================================
# [PARTE 5] — Similitud coseno entre documentos
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 5
# (requiere haber ejecutado el bloque PARTE 3a)

# N_DOCS = min(10, len(oraciones_limpias))
# sim    = cosine_similarity(X_tfidf[:N_DOCS])
# df_sim = pd.DataFrame(
#     sim.round(2),
#     columns=[f"D{i}" for i in range(N_DOCS)],
#     index=[f"D{i}" for i in range(N_DOCS)],
# )
#
# print("=" * 60)
# print(f"[PARTE 5] Similitud coseno — primeros {N_DOCS} documentos")
# print("=" * 60)
# print(df_sim.to_string())
# print()
#
# # Par más similar (ignorando la diagonal — docs consigo mismos)
# import numpy as np
# sim_copy = sim.copy()
# np.fill_diagonal(sim_copy, 0)
# i, j = divmod(sim_copy.argmax(), N_DOCS)
# print(f"  Par más similar: D{i} y D{j}  (similitud = {sim_copy[i,j]:.3f})")
# print(f"  D{i}: {oraciones_limpias[i][:80]}")
# print(f"  D{j}: {oraciones_limpias[j][:80]}")
# print()


# ================================================================
# [PARTE 6 - BONUS] — TF-IDF con bigramas
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 6

# tfidf_bg = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
#                            stop_words=list(STOPWORDS_ES))
# X_tfidf_bg = tfidf_bg.fit_transform(oraciones_limpias)
# df_tfidf_bg = pd.DataFrame(X_tfidf_bg.toarray(),
#                             columns=tfidf_bg.get_feature_names_out())
#
# print("=" * 60)
# print("[PARTE 6] TF-IDF con bigramas — top-5 por documento")
# print("=" * 60)
# for i in range(min(3, len(df_tfidf_bg))):
#     row  = df_tfidf_bg.iloc[i]
#     top5 = row.nlargest(5)
#     reporte = "  ,  ".join([f"{p} ({v:.3f})" for p, v in top5.items()])
#     print(f"  Doc {i}: {reporte}")
# print()
# print("  → ¿Aparecen bigramas entre las palabras más características?")
