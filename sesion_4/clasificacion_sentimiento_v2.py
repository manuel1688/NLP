# ================================================================
# Clasificacion de Sentimiento v2 — Train / Validation / Test
# ================================================================
# Fase 1: Carga del corpus (corpus_train.jsonl / corpus_test.jsonl)
# Fase 2: Split Train / Validation / Test
# Fase 3: Busqueda de hiperparametros en Validation
# Fase 4: Entrenamiento final y evaluacion en Test
# Fase 5: Prediccion sobre ejemplos nuevos
# ================================================================
#
# DEPENDENCIAS
# ------------
# pip install scikit-learn nltk
#
# CORPUS
# ------
# Requiere los archivos generados por sesion_3/preparacion_corpus.py:
#   ../sesion_3/corpus_train.jsonl
#   ../sesion_3/corpus_test.jsonl
#
# ================================================================

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------
SESION_3     = os.path.join(os.path.dirname(__file__), "..", "sesion_3")
TRAIN_FILE   = os.path.join(SESION_3, "corpus_train.jsonl")
TEST_FILE    = os.path.join(SESION_3, "corpus_test.jsonl")

VAL_SIZE     = 0.2    # 20% del train oficial se reserva para validacion
RANDOM_STATE = 42


# ----------------------------------------------------------------
# Metricas manuales (reutilizadas de clasificacion_sentimiento.py)
# ----------------------------------------------------------------
def confusion_matrix_manual(y_true, y_pred, pos_label=1):
    TP = sum(t == pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    TN = sum(t != pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    FP = sum(t != pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    FN = sum(t == pos_label and p != pos_label for t, p in zip(y_true, y_pred))
    return TP, TN, FP, FN


def metricas_manuales(y_true, y_pred, pos_label=1):
    TP, TN, FP, FN = confusion_matrix_manual(y_true, y_pred, pos_label)
    acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ================================================================
# FASE 1 — Carga del corpus
# ================================================================
# Herramienta : json (Python stdlib)
# Que hace    : lee los JSONL generados por preparacion_corpus.py
#               y reconstruye el texto uniendo los tokens con espacios
# Por que     : TfidfVectorizer espera cadenas de texto, no listas

print("[ FASE 1 ] Cargando corpus desde sesion_3/")


def cargar_jsonl(ruta):
    textos, etiquetas = [], []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            ej = json.loads(linea)
            textos.append(" ".join(ej["tokens"]))
            etiquetas.append(ej["label"])
    return textos, etiquetas


textos_train_full, y_train_full = cargar_jsonl(TRAIN_FILE)
textos_test,       y_test       = cargar_jsonl(TEST_FILE)

print(f"  corpus_train.jsonl : {len(textos_train_full):,} ejemplos")
print(f"  corpus_test.jsonl  : {len(textos_test):,} ejemplos")

# ================================================================
# FASE 2 — Split Train / Validation / Test
# ================================================================
# Herramienta : train_test_split (scikit-learn)
# Que hace    : divide el train oficial en Train (80%) y Validation (20%)
# Por que     : el Test Set debe permanecer intocado hasta la evaluacion
#               final; el Validation Set es el "borrador" para ajustar
#               hiperparametros sin contaminar el Test

print(f"\n[ FASE 2 ] Split Train / Validation  ({int((1-VAL_SIZE)*100)}% / {int(VAL_SIZE*100)}%)")

textos_train, textos_val, y_train, y_val = train_test_split(
    textos_train_full, y_train_full,
    test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
)

print(f"  Train      : {len(textos_train):,} ejemplos")
print(f"  Validation : {len(textos_val):,} ejemplos")
print(f"  Test       : {len(textos_test):,} ejemplos  ← no se toca hasta Fase 4")

# ================================================================
# FASE 3 — Busqueda de hiperparametros en Validation
# ================================================================
# Herramienta : bucle manual sobre rejilla de configuraciones
# Que hace    : entrena un modelo por configuracion, evalua F1 en
#               Validation y guarda los resultados
# Por que     : el F1 en Validation guia la seleccion; el Test Set
#               nunca interviene en esta fase

print("\n[ FASE 3 ] Busqueda de hiperparametros en Validation")
print(f"  {'ngram_range':<15} {'max_features':<15} {'F1 (val)':>10}")
print("  " + "-" * 42)

REJILLA = [
    {"ngram_range": (1, 1), "max_features": 500},
    {"ngram_range": (1, 1), "max_features": 1000},
    {"ngram_range": (1, 1), "max_features": 5000},
    {"ngram_range": (1, 2), "max_features": 500},
    {"ngram_range": (1, 2), "max_features": 1000},
    {"ngram_range": (1, 2), "max_features": 5000},
]

mejor_f1     = -1.0
mejor_config = None
resultados   = []

for config in REJILLA:
    vec = TfidfVectorizer(
        ngram_range=config["ngram_range"],
        max_features=config["max_features"],
        min_df=2,
    )
    X_tr  = vec.fit_transform(textos_train)
    X_val = vec.transform(textos_val)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_train)

    y_val_pred = clf.predict(X_val)
    m = metricas_manuales(y_val, y_val_pred)
    f1_val = m["f1"]

    resultados.append({**config, "f1_val": f1_val})
    print(f"  {str(config['ngram_range']):<15} {config['max_features']:<15} {f1_val:>10.4f}")

    if f1_val > mejor_f1:
        mejor_f1     = f1_val
        mejor_config = config

print(f"\n  Mejor configuracion : ngram_range={mejor_config['ngram_range']}  "
      f"max_features={mejor_config['max_features']}  F1={mejor_f1:.4f}")

# ================================================================
# FASE 4 — Entrenamiento final y evaluacion en Test
# ================================================================
# Herramienta : TfidfVectorizer + LogisticRegression (scikit-learn)
# Que hace    : reentrena con Train+Validation combinados usando la
#               mejor configuracion y evalua en Test por primera vez
# Por que     : combinar Train+Val maximiza los datos de entrenamiento;
#               el Test Set solo se mira una vez, aqui.

print("\n[ FASE 4 ] Entrenamiento final y evaluacion en Test")
print("  Reentrenando con Train + Validation combinados...")

textos_train_val = textos_train + textos_val
y_train_val      = y_train + y_val

vec_final = TfidfVectorizer(
    ngram_range=mejor_config["ngram_range"],
    max_features=mejor_config["max_features"],
    min_df=2,
)
X_train_val = vec_final.fit_transform(textos_train_val)
X_test      = vec_final.transform(textos_test)

clf_final = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf_final.fit(X_train_val, y_train_val)

y_pred = clf_final.predict(X_test)

print("\n  Classification report (Test Set):")
print(classification_report(y_test, y_pred, target_names=["negativo", "positivo"]))

m = metricas_manuales(y_test, y_pred)
print("  Metricas manuales (Test Set):")
print(f"    Matriz de confusion: TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
print(f"    Accuracy  : {m['accuracy']:.4f}")
print(f"    Precision : {m['precision']:.4f}")
print(f"    Recall    : {m['recall']:.4f}")
print(f"    F1-score  : {m['f1']:.4f}")

# ================================================================
# FASE 5 — Prediccion sobre ejemplos nuevos
# ================================================================
# Herramienta : clf_final.predict + vec_final.transform
# Que hace    : aplica el modelo entrenado a frases nuevas
# Por que     : demuestra que el pipeline funciona en produccion

print("\n[ FASE 5 ] Prediccion sobre ejemplos nuevos")

EJEMPLOS = [
    "Me encanto, muy buen producto, lo recomiendo",
    "Pesimo, llego roto y el servicio fue horrible",
    "Calidad aceptable, funciona bien por el precio",
    "No lo compren, una completa decepcion",
]

ETIQUETAS = {0: "negativo", 1: "positivo"}

preds = clf_final.predict(vec_final.transform(EJEMPLOS))
for texto, pred in zip(EJEMPLOS, preds):
    print(f"  [{ETIQUETAS[pred].upper():<9}]  {texto}")

print("\n[ PIPELINE COMPLETO ]")
print(f"  Mejor config : ngram_range={mejor_config['ngram_range']}  "
      f"max_features={mejor_config['max_features']}")
print(f"  F1 Validation : {mejor_f1:.4f}")
print(f"  F1 Test       : {m['f1']:.4f}")

# ----------------------------------------------------------------
# RESUMEN DEL FLUJO
# ----------------------------------------------------------------
# corpus_train.jsonl  →  80% Train  + 20% Validation
#                              ↓               ↓
#                        fit modelo      evaluar F1
#                              ↓               ↓
#                         rejilla de hiperparametros
#                              ↓
#                        mejor config
#                              ↓
#                   Train + Validation  →  fit modelo final
#                                                ↓
# corpus_test.jsonl  →  Test Set (una sola vez)  →  metricas finales
# ----------------------------------------------------------------
