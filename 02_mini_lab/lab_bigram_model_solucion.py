from collections import defaultdict

# ================================================================
# Lab 02 — Solucion de Bigram Model (corpus simulado)
# ================================================================

corpus = [
    "yo estudio nlp",
    "yo estudio python",
    "yo aprendo nlp",
    "tu estudias nlp",
    "yo estudio modelos",
]


def tokenizar_con_fronteras(oracion):
    tokens = oracion.lower().split()
    return ["<s>"] + tokens + ["</s>"]


def construir_conteos(corpus_oraciones):
    conteo_unigramas = defaultdict(int)
    conteo_bigramas = defaultdict(int)

    for oracion in corpus_oraciones:
        tokens = tokenizar_con_fronteras(oracion)
        for i in range(len(tokens) - 1):
            actual = tokens[i]
            siguiente = tokens[i + 1]
            conteo_unigramas[actual] += 1
            conteo_bigramas[(actual, siguiente)] += 1

    return conteo_unigramas, conteo_bigramas


def calcular_probabilidades(conteo_unigramas, conteo_bigramas):
    probabilidades = {}
    for (actual, siguiente), c_bigram in conteo_bigramas.items():
        probabilidades[(actual, siguiente)] = c_bigram / conteo_unigramas[actual]
    return probabilidades


def probabilidad(probabilidades, actual, siguiente):
    return probabilidades.get((actual, siguiente), 0.0)


def generar_frase_greedy(probabilidades, inicio="<s>", fin="</s>", max_pasos=10):
    frase = []
    actual = inicio

    for _ in range(max_pasos):
        candidatos = []
        for (w_actual, w_sig), p in probabilidades.items():
            if w_actual == actual:
                candidatos.append((w_sig, p))

        if not candidatos:
            break

        # Empate: Python conserva orden de insercion; este criterio basta para el lab.
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
    print(f"P(estudio | yo) = {probabilidad(probabilidades, 'yo', 'estudio')}")
    print(f"P(nlp | estudio) = {probabilidad(probabilidades, 'estudio', 'nlp')}")
    print(f"P(</s> | nlp) = {probabilidad(probabilidades, 'nlp', '</s>')}")

    frase = generar_frase_greedy(probabilidades)
    print("\n=== Frase generada (greedy) ===")
    print(frase)


if __name__ == "__main__":
    main()
