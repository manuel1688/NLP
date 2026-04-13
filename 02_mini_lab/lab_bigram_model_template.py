from collections import defaultdict

# ================================================================
# Lab 02 — Bigram Model (completar)
# Tu tarea: implementar las dos funciones marcadas con # TODO
# ================================================================

# Corpus dado — no modificar
corpus = [
    "agrega sal al agua",
    "agrega sal y pimienta",
    "agrega las cebollas al caldo",
    "agrega las zanahorias al caldo",
    "calienta el agua en la olla",
    "calienta el aceite en la sartén",
    "mezcla la harina con el agua",
    "mezcla la harina con la leche",
    "corta las cebollas en trozos",
    "corta las zanahorias en trozos",
    "hierve el agua por cinco minutos",
    "hierve el caldo por diez minutos",
    "sirve el plato caliente",
]


# --- Dada: no necesitas modificar esta función ---
def tokenizar_con_fronteras(oracion):
    tokens = oracion.lower().split()
    return ["<s>"] + tokens + ["</s>"]


# ----------------------------------------------------------------
# TODO 1: Implementa construir_conteos
#
# Para cada oración del corpus:
#   1. Tokenízala con tokenizar_con_fronteras()
#   2. Recorre los tokens con un índice i desde 0 hasta len-1
#   3. Para cada posición i:
#      - suma 1 a conteo_unigramas[tokens[i]]
#      - suma 1 a conteo_bigramas[(tokens[i], tokens[i+1])]
#
# Consulta la Tabla de Bigramas en el .md para verificar.
# ----------------------------------------------------------------
def construir_conteos(corpus_oraciones):
    conteo_unigramas = defaultdict(int)
    conteo_bigramas = defaultdict(int)

    for oracion in corpus_oraciones:
        tokens = tokenizar_con_fronteras(oracion)
        for i in range(len(tokens) - 1):
            # TODO: registra el unigrama tokens[i]
            pass
            # TODO: registra el bigrama (tokens[i], tokens[i+1])
            pass

    return conteo_unigramas, conteo_bigramas


# ----------------------------------------------------------------
# TODO 2: Implementa calcular_probabilidades
#
# Para cada entrada en conteo_bigramas:
#   clave = (actual, siguiente)
#   valor = conteo del bigrama
#
# Aplica la fórmula:
#   P(siguiente | actual) = conteo_bigramas[(actual, siguiente)]
#                           / conteo_unigramas[actual]
#
# Guarda el resultado en el diccionario probabilidades.
# ----------------------------------------------------------------
def calcular_probabilidades(conteo_unigramas, conteo_bigramas):
    probabilidades = {}

    for (actual, siguiente), c_bigram in conteo_bigramas.items():
        # TODO: calcula la probabilidad y guárdala en probabilidades[(actual, siguiente)]
        pass

    return probabilidades


# --- Dada: no necesitas modificar ---
def probabilidad(probabilidades, actual, siguiente):
    return probabilidades.get((actual, siguiente), 0.0)


# --- Dada: no necesitas modificar ---
def generar_frase_greedy(probabilidades, inicio="<s>", fin="</s>", max_pasos=10):
    frase = []
    actual = inicio

    for _ in range(max_pasos):
        candidatos = [
            (w_sig, p)
            for (w_actual, w_sig), p in probabilidades.items()
            if w_actual == actual
        ]
        if not candidatos:
            break
        siguiente, _ = max(candidatos, key=lambda x: x[1])
        if siguiente == fin:
            break
        frase.append(siguiente)
        actual = siguiente

    return " ".join(frase)


def main():
    conteo_unigramas, conteo_bigramas = construir_conteos(corpus)
    probabilidades = calcular_probabilidades(conteo_unigramas, conteo_bigramas)

    print("=== Probabilidades solicitadas ===")
    print(f"P(las | agrega)  = {probabilidad(probabilidades, 'agrega', 'las')}")
    print(f"P(el | hierve)   = {probabilidad(probabilidades, 'hierve', 'el')}")
    print(f"P(con | harina)  = {probabilidad(probabilidades, 'harina', 'con')}")

    frase = generar_frase_greedy(probabilidades)
    print("\n=== Frase generada (greedy) ===")
    print(frase)


if __name__ == "__main__":
    main()
