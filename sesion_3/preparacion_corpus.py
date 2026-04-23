# ================================================================
# Preparacion del Corpus — Reconocer Sentimiento
# Ver plan completo: reconocer_sentimiento.md
# ================================================================
# Fase 1: Carga del dataset amazon_reviews_multi (es)    <- implementada
# Fase 2: Binarizacion de etiquetas (0/1) y filtrado     <- implementada
# Fase 3: Balanceo de clases                             <- implementada
# Fase 4: Tokenizacion (NLTK, igual que word2vec.py)     <- implementada
# Fase 5: Split train / test (splits oficiales)          <- implementada
# Fase 6: Guardar corpus_train.jsonl / corpus_test.jsonl <- implementada
# ================================================================
#
# DEPENDENCIAS
# ------------
# pip install datasets nltk
# pip install nltk
#   Se usa la copia MTEB (mteb/amazon_reviews_multi) en formato Parquet,
#   compatible con datasets 3.x. El original (amazon_reviews_multi) fue
#   dado de baja por sus autores.
#
# ================================================================

from collections import Counter
import json
import os
import random

import nltk
from datasets import load_dataset

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------
LANG = "es"   # idioma del dataset en HuggingFace


# ----------------------------------------------------------------
# PASO 1 — Carga del dataset desde HuggingFace
# ----------------------------------------------------------------
# Herramienta : datasets.load_dataset (HuggingFace)
# Que hace    : descarga amazon_reviews_multi en español y devuelve
#               un DatasetDict con splits train / validation / test
# Por que     : usar splits oficiales garantiza reproducibilidad y
#               es la practica estandar en NLP

print("[ PASO 1 ] Descargando amazon_reviews_multi (es) desde HuggingFace...")

dataset = load_dataset("mteb/amazon_reviews_multi", LANG)

print(f"\n  Splits disponibles : {list(dataset.keys())}")
print(f"  Columnas           : {dataset['train'].column_names}")
print("\n  Ejemplos por split (antes de filtrar):")
for split in dataset:
    print(f"    {split:12s} : {len(dataset[split]):>7,} ejemplos")

# En la version MTEB la columna se llama 'label' con valores 0-4
# (0 = 1 estrella ... 4 = 5 estrellas)
print("\n  Distribucion de label (0-4) en train:")
dist = Counter(dataset["train"]["label"])
for lbl in sorted(dist):
    print(f"    label {lbl} ({lbl+1} estrella/s) : {dist[lbl]:>7,}")


# ----------------------------------------------------------------
# PASO 2 — Binarizacion de etiquetas
# ----------------------------------------------------------------
# Herramienta : Dataset.map + Dataset.filter (HuggingFace datasets)
# Que hace    : convierte la columna 'stars' (1-5) en 'label' (0/1)
#               y elimina las resenas de 3 estrellas (ambiguas)
# Por que     : un problema binario limpio es necesario para que la
#               comparacion BoW / TF-IDF / Word2Vec sea justa;
#               las 3 estrellas introducen ruido sin etiqueta clara
#
# Mapeo (columna 'label' MTEB, valores 0-4):
#   0, 1  →  negativo (0)   [1-2 estrellas]
#   2     →  descartar (-1) [3 estrellas]
#   3, 4  →  positivo  (1)  [4-5 estrellas]

print("\n[ PASO 2 ] Binarizacion de etiquetas (label MTEB 0-4 -> binario 0/1)")


def binarizar(ejemplo):
    """Convierte label MTEB (0-4) en binario; -1 indica que se debe descartar."""
    s = ejemplo["label"]   # 0=1★ 1=2★ 2=3★ 3=4★ 4=5★
    if s <= 1:
        return {"label_bin": 0}
    elif s >= 3:
        return {"label_bin": 1}
    else:
        return {"label_bin": -1}   # 3 estrellas — se filtrara


dataset = dataset.map(binarizar, desc="Binarizando")

print("  Filtrando resenas de 3 estrellas (label_bin == -1)...")
dataset_filtrado = dataset.filter(
    lambda ej: ej["label_bin"] != -1,
    desc="Filtrando neutras",
)

# ----------------------------------------------------------------
# Reporte final por split
# ----------------------------------------------------------------
print("\n  Resumen por split (despues de filtrar):")
print(f"  {'Split':<12} {'Total':>8} {'Positivos':>12} {'Negativos':>12} {'Descartados':>13}")
print("  " + "-" * 61)

for split in dataset_filtrado:
    total_original = len(dataset[split])
    total_filtrado = len(dataset_filtrado[split])
    labels         = dataset_filtrado[split]["label_bin"]
    positivos      = labels.count(1)
    negativos      = labels.count(0)
    descartados    = total_original - total_filtrado
    pct_desc       = 100 * descartados / total_original if total_original > 0 else 0

    print(
        f"  {split:<12} {total_filtrado:>8,} {positivos:>12,} {negativos:>12,}"
        f"  {descartados:>6,} ({pct_desc:.1f}%)"
    )

# Verificacion de integridad: positivos + negativos == total filtrado
print("\n  Verificacion de integridad:")
for split in dataset_filtrado:
    labels    = dataset_filtrado[split]["label_bin"]
    pos       = labels.count(1)
    neg       = labels.count(0)
    total     = len(dataset_filtrado[split])
    invalidos = [l for l in labels if l not in (0, 1)]
    ok = (pos + neg == total) and (len(invalidos) == 0)
    print(f"    {split:<12} pos+neg==total: {ok}  |  etiquetas invalidas: {len(invalidos)}")

print("\n[ FASE 1 y 2 COMPLETAS ]")

# ================================================================
# FASE 3 — Balanceo de clases
# ================================================================
# Herramienta : random.sample (Python stdlib)
# Que hace    : recorta la clase mayoritaria al tamaño de la minoritaria
#               en cada split de forma independiente
# Por que     : sin balanceo un clasificador que siempre diga "positivo"
#               tendria accuracy alta y la comparacion seria engañosa

print("\n[ FASE 3 ] Balanceo de clases")


def balancear(dataset_split):
    indices_pos = [i for i, l in enumerate(dataset_split["label_bin"]) if l == 1]
    indices_neg = [i for i, l in enumerate(dataset_split["label_bin"]) if l == 0]
    n = min(len(indices_pos), len(indices_neg))
    random.seed(42)
    seleccionados = sorted(
        random.sample(indices_pos, n) + random.sample(indices_neg, n)
    )
    return dataset_split.select(seleccionados)


dataset_balanceado = {}
for split in ["train", "test"]:
    dataset_balanceado[split] = balancear(dataset_filtrado[split])
    labels = dataset_balanceado[split]["label_bin"]
    print(
        f"  {split:<12} → {len(dataset_balanceado[split]):>7,} ejemplos  "
        f"(pos: {labels.count(1):,}  neg: {labels.count(0):,})"
    )

# ================================================================
# FASE 4 — Tokenizacion
# ================================================================
# Herramienta : nltk.word_tokenize (mismo que word2vec_word2vec.py)
# Que hace    : divide el texto en tokens, filtra no-alfabeticos
#               y pasa a minusculas
# Por que     : los tres metodos (BoW, TF-IDF, Word2Vec) deben partir
#               de los mismos tokens para que la comparacion sea justa

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

print("\n[ FASE 4 ] Tokenizacion con nltk.word_tokenize")


def tokenizar(ejemplo):
    tokens = nltk.word_tokenize(ejemplo["text"].lower(), language="spanish")
    tokens = [t for t in tokens if t.isalpha()]
    return {"tokens": tokens}


dataset_tokenizado = {}
for split in ["train", "test"]:
    dataset_tokenizado[split] = dataset_balanceado[split].map(
        tokenizar, desc=f"Tokenizando {split}"
    )

print(f"  Ejemplo de tokens (train[0]) : {dataset_tokenizado['train'][0]['tokens'][:10]}")

# ================================================================
# FASE 5 — Split train / test
# ================================================================
# Herramienta : splits oficiales del dataset
# Que hace    : usa 'train' para entrenar y 'test' para evaluar
# Por que     : reproducible y es la practica estandar en NLP

print("\n[ FASE 5 ] Split train / test (splits oficiales)")
print(f"  train : {len(dataset_tokenizado['train']):>7,} ejemplos")
print(f"  test  : {len(dataset_tokenizado['test']):>7,} ejemplos")

# ================================================================
# FASE 6 — Guardar en disco
# ================================================================
# Herramienta : json (Python stdlib)
# Que hace    : escribe un archivo JSONL por split con tokens y etiqueta
# Por que     : los scripts de BoW, TF-IDF y Word2Vec cargan estos
#               archivos sin repetir el preprocesamiento; JSONL es
#               texto plano facil de compartir y leer
#
# Formato por linea:
#   {"tokens": ["buen", "producto"], "label": 1}

print("\n[ FASE 6 ] Guardando corpus en disco (JSONL)")

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))

for split in ["train", "test"]:
    ruta = os.path.join(DIRECTORIO, f"corpus_{split}.jsonl")
    with open(ruta, "w", encoding="utf-8") as f:
        for ej in dataset_tokenizado[split]:
            linea = {"tokens": ej["tokens"], "label": ej["label_bin"]}
            f.write(json.dumps(linea, ensure_ascii=False) + "\n")
    print(f"  {split:<6} → {ruta}  ({len(dataset_tokenizado[split]):,} lineas)")

print("\n[ PIPELINE COMPLETO ]")
print(f"  {os.path.join(DIRECTORIO, 'corpus_train.jsonl')}")
print(f"  {os.path.join(DIRECTORIO, 'corpus_test.jsonl')}")
print('\n  Formato: {"tokens": ["buen", "producto", ...], "label": 0}')

# ----------------------------------------------------------------
# RESULTADO FINAL
# ----------------------------------------------------------------
# Al terminar este script quedan en disco dos archivos JSONL en
# el mismo directorio que este script:
#
#   corpus_train.jsonl  —  ejemplos de entrenamiento
#   corpus_test.jsonl   —  ejemplos de evaluacion
#
# Cada linea: {"tokens": ["buen", "producto", ...], "label": 0}
# Positivos y negativos estan balanceados en cada archivo.
# Los tres metodos (BoW, TF-IDF, Word2Vec) cargan estos archivos
# directamente sin repetir ningun preprocesamiento.
# ----------------------------------------------------------------
