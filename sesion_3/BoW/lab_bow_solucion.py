# ================================================================
# Lab 03 — Bag of Words (BoW) — Script de referencia
# Corpus: eswiki_corpus.txt (Wikipedia ES)
# ================================================================
# Descomenta cada bloque de ejercicio cuando llegues a esa parte
# del lab_bow.md. Ejecuta con: python3 lab_bow_solucion.py
# ================================================================

import os
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "..", "eswiki_corpus.txt")

# ----------------------------------------------------------------
# Stop words en español (lista manual — sin dependencias externas)
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

# ================================================================
# FUNCIÓN DE LIMPIEZA — usada por todos los ejercicios
# ================================================================
def limpiar(texto):
    """
    Elimina ruido típico de un dump de Wikipedia en español:
    - &lt;ref&gt;...&lt;/ref&gt;  → referencias HTML escapadas
    - <ref>...</ref>          → etiquetas HTML reales
    - [[texto]]               → enlaces internos Wiki
    - {{plantilla|param=val}} → plantillas Wiki
    - ''cursiva'' '''negrita'' → formato Wiki
    - &amp; &nbsp; &quot; etc → entidades HTML
    - Números y puntuación    → solo conserva letras y espacios
    """
    texto = re.sub(r'&lt;.*?&gt;', ' ', texto)           # tags HTML escapados
    texto = re.sub(r'<[^>]+>', ' ', texto)               # tags HTML reales
    texto = re.sub(r'\[\[.*?\]\]', ' ', texto)           # [[wiki links]]
    texto = re.sub(r'\{\{.*?\}\}', ' ', texto)           # {{plantillas}}
    texto = re.sub(r"'{2,}", '', texto)                  # ''cursiva'' '''negrita'''
    texto = re.sub(r'&\w+;', ' ', texto)                 # &amp; &nbsp; &quot; etc
    texto = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', ' ', texto)  # solo letras
    texto = re.sub(r'\s+', ' ', texto).strip().lower()  # espacios múltiples y lowercase
    return texto


# ================================================================
# [PARTE 2] — Observar la función de limpieza
# ================================================================
print("=" * 60)
print("[PARTE 2] Función de limpieza — antes vs después")
print("=" * 60)

ejemplos_ruidosos = [
    "Andorra &lt;ref&gt;ver nota&lt;/ref&gt; es un [[Estado]] ''soberano''.",
    "Su población es de 85.101&amp;nbsp;habitantes (2024).",
    "{{plantilla|tipo=país|nombre=Andorra}} fundado en 1278.",
    "Limita con [[España]] y con [[Francia]] al norte.",
]

for ejemplo in ejemplos_ruidosos:
    limpio = limpiar(ejemplo)
    print(f"  ANTES:  {ejemplo}")
    print(f"  DESPUÉS: {limpio}")
    print()


# ================================================================
# CARGA Y LIMPIEZA DEL CORPUS
# ================================================================
print("=" * 60)
print("Cargando corpus eswiki_corpus.txt ...")
print("=" * 60)

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    lineas_raw = [l.strip() for l in f if l.strip()]

# Limpiar solo las primeras 50 líneas para los ejercicios
oraciones_raw    = lineas_raw[:50]
oraciones_limpias = [limpiar(o) for o in oraciones_raw]
oraciones_limpias = [o for o in oraciones_limpias if len(o.split()) > 3]

print(f"  Oraciones cargadas (primeras 50): {len(oraciones_limpias)}")
print(f"  Ejemplo limpio:\n  → {oraciones_limpias[2][:120]}")
print()


# ================================================================
# [PARTE 3a] — CountVectorizer unigramas
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 3a

# vectorizador = CountVectorizer(max_features=500)
# X = vectorizador.fit_transform(oraciones_limpias)
# df = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())
#
# print("=" * 60)
# print(f"[PARTE 3a] Matriz BoW — forma: {X.shape}")
# print("=" * 60)
# print(f"  Filas    = documentos : {X.shape[0]}")
# print(f"  Columnas = palabras   : {X.shape[1]}")
# celdas_totales = X.shape[0] * X.shape[1]
# celdas_vacias  = (X.toarray() == 0).sum()
# print(f"  Sparsity (% celdas = 0): {100 * celdas_vacias / celdas_totales:.1f}%")
# print()
# print(df.iloc[:5, :10])   # primeras 5 filas, primeras 10 columnas
# print()


# ================================================================
# [PARTE 3b] — Top-10 palabras más frecuentes (sin stop words)
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 3b

# conteo_total = X.toarray().sum(axis=0)
# palabras     = vectorizador.get_feature_names_out()
# frecuencias  = sorted(zip(palabras, conteo_total), key=lambda x: -x[1])
#
# print("=" * 60)
# print("[PARTE 3b] Top-10 palabras más frecuentes (unigramas)")
# print("=" * 60)
# for palabra, freq in frecuencias[:10]:
#     print(f"  {palabra:<20} {int(freq):>5} apariciones")
# print()


# ================================================================
# [PARTE 4] — BoW con stop words eliminadas
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 4

# vectorizador_sw = CountVectorizer(max_features=500,
#                                   stop_words=list(STOPWORDS_ES))
# X_sw = vectorizador_sw.fit_transform(oraciones_limpias)
#
# conteo_sw   = X_sw.toarray().sum(axis=0)
# palabras_sw = vectorizador_sw.get_feature_names_out()
# frecuencias_sw = sorted(zip(palabras_sw, conteo_sw), key=lambda x: -x[1])
#
# print("=" * 60)
# print("[PARTE 4] Top-10 con stop words eliminadas")
# print("=" * 60)
# print(f"  Vocabulario sin stop words: {X_sw.shape[1]} palabras")
# print()
# for palabra, freq in frecuencias_sw[:10]:
#     print(f"  {palabra:<20} {int(freq):>5} apariciones")
# print()


# ================================================================
# [PARTE 5] — CountVectorizer con bigramas
# ================================================================
# Descomenta el bloque siguiente cuando llegues a la Parte 5

# vectorizador_bg = CountVectorizer(ngram_range=(2, 2), max_features=300,
#                                   stop_words=list(STOPWORDS_ES))
# X_bg = vectorizador_bg.fit_transform(oraciones_limpias)
#
# conteo_bg   = X_bg.toarray().sum(axis=0)
# palabras_bg = vectorizador_bg.get_feature_names_out()
# frecuencias_bg = sorted(zip(palabras_bg, conteo_bg), key=lambda x: -x[1])
#
# print("=" * 60)
# print("[PARTE 5] Top-15 bigramas más frecuentes")
# print("=" * 60)
# for bigrama, freq in frecuencias_bg[:15]:
#     print(f"  {bigrama:<30} {int(freq):>5} apariciones")
# print()


# ================================================================
# VERIFICACIÓN DEL EJERCICIO 1a — Mini corpus manual
# ================================================================
print("=" * 60)
print("Verificación Parte 1 — Mini corpus (3 documentos)")
print("=" * 60)

mini_corpus = [
    "andorra es un estado pequeño",
    "andorra tiene capital en andorra",
    "españa es un estado europeo",
]

v_mini = CountVectorizer()
X_mini = v_mini.fit_transform(mini_corpus)
df_mini = pd.DataFrame(
    X_mini.toarray(),
    columns=v_mini.get_feature_names_out(),
    index=["D1", "D2", "D3"],
)

print(df_mini.to_string())
print()
print("  → Compara con tu respuesta del Ejercicio 1a")
