# ================================================================
# Preparacion del Corpus — Reconocer Sentimiento
# Ver plan completo: reconocer_sentimiento.md
# ================================================================
# Fase 1: Carga del dataset amazon_reviews_multi (es)  <- implementada
# Fase 2: Binarizacion de etiquetas (0/1) y filtrado   <- implementada
# ================================================================
#
# DEPENDENCIAS
# ------------
# pip install datasets
#
# ================================================================

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

dataset = load_dataset("amazon_reviews_multi", LANG)

print(f"\n  Splits disponibles : {list(dataset.keys())}")
print(f"  Columnas           : {dataset['train'].column_names}")
print("\n  Ejemplos por split (antes de filtrar):")
for split in dataset:
    print(f"    {split:12s} : {len(dataset[split]):>7,} ejemplos")

print("\n  Distribucion de estrellas en train:")
from collections import Counter
dist = Counter(dataset["train"]["stars"])
for estrella in sorted(dist):
    print(f"    {estrella} estrellas : {dist[estrella]:>7,}")


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
# Mapeo:
#   1 – 2  →  negativo (0)
#   3      →  descartar (label = -1, luego filtrado)
#   4 – 5  →  positivo  (1)

print("\n[ PASO 2 ] Binarizacion de etiquetas (stars -> label 0 / 1)")


def binarizar(ejemplo):
    """Convierte stars en label binario; -1 indica que se debe descartar."""
    s = ejemplo["stars"]
    if s <= 2:
        label = 0
    elif s >= 4:
        label = 1
    else:
        label = -1   # estrella 3 — se filtrara en el proximo paso
    return {"label": label}


dataset = dataset.map(binarizar, desc="Binarizando")

print("  Filtrando resenas de 3 estrellas (label == -1)...")
dataset_filtrado = dataset.filter(
    lambda ej: ej["label"] != -1,
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
    labels         = dataset_filtrado[split]["label"]
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
    labels    = dataset_filtrado[split]["label"]
    pos       = labels.count(1)
    neg       = labels.count(0)
    total     = len(dataset_filtrado[split])
    invalidos = [l for l in labels if l not in (0, 1)]
    ok = (pos + neg == total) and (len(invalidos) == 0)
    print(f"    {split:<12} pos+neg==total: {ok}  |  etiquetas invalidas: {len(invalidos)}")

print("\n[ FASE 1 y 2 COMPLETAS ]")
print("  Siguiente paso: Fase 3 — Balanceo de clases (ver reconocer_sentimiento.md)")

# ----------------------------------------------------------------
# RESULTADO FINAL
# ----------------------------------------------------------------
# Al terminar este script se tiene en memoria:
#
#   dataset_filtrado  —  DatasetDict con tres splits (train / validation / test)
#                        cada uno con las columnas originales del dataset
#                        mas la columna 'label' (0 = negativo, 1 = positivo)
#                        SIN reseñas de 3 estrellas
#
# No se escribe ningun archivo en disco todavia.
# El guardado ocurre en la Fase 6 (corpus_train.jsonl / corpus_test.jsonl).
# ----------------------------------------------------------------
