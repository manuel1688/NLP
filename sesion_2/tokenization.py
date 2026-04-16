# ================================================================
# Tokenization in Natural Language Processing
# ================================================================
# Tokenization es el proceso de dividir un texto en unidades
# más pequenas llamadas tokens (oraciones, palabras o sub-palabras).
# ================================================================

import nltk
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize, word_tokenize

CORPUS = """Hello, My name is Pinki Dagar.
I am a student of b.tech computer science."""


def main():
    nltk.download("punkt", quiet=True)
    # Algunas versiones antiguas de nltk no incluyen este recurso.
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    # ---------------------------------------------------------------
    # 2. Definicion del corpus
    # ---------------------------------------------------------------
    print("=== Corpus ===")
    print(CORPUS)

    # ---------------------------------------------------------------
    # 3. Sentence Tokenization
    # Divide el corpus en oraciones individuales.
    # ---------------------------------------------------------------
    documents = sent_tokenize(CORPUS)

    print("\n=== Sentence Tokenization ===")
    for sentence in documents:
        print(sentence)

    # ---------------------------------------------------------------
    # 4. Word Tokenization
    # Divide cada oracion en palabras y signos de puntuacion.
    # Al tokenizar por oracion, el punto final suele separarse como token ".".
    # ---------------------------------------------------------------
    word_tokens_per_sentence = []
    print("\n=== Word Tokenization ===")
    for sentence in documents:
        tokens = word_tokenize(sentence)
        word_tokens_per_sentence.append(tokens)
        print(tokens)

    # ---------------------------------------------------------------
    # 5. Treebank Word Tokenizer
    # Usa convenciones del Penn Treebank: maneja la puntuacion
    # (como el punto al final de oracion) de forma diferente.
    # Aplicado al corpus completo puede mantener secuencias como "Dagar." juntas.
    # ---------------------------------------------------------------
    tokenizer = TreebankWordTokenizer()
    treebank_tokens = tokenizer.tokenize(CORPUS)

    print("\n=== Treebank Word Tokenizer ===")
    print(treebank_tokens)

    # ---------------------------------------------------------------
    # Explicacion de la diferencia observada en salida
    # ---------------------------------------------------------------
    print("\n=== Explicacion ===")
    print("1) Sentence Tokenization divide el texto en dos oraciones.")
    print("2) word_tokenize se aplico por cada oracion, por eso separa mejor la puntuacion final (ej: 'Dagar' y '.').")
    print("3) TreebankWordTokenizer se aplico al corpus completo, por eso puede conservar 'Dagar.' como un solo token.")
    print("4) Ambos son correctos: cambia el resultado segun el tokenizador y el nivel al que lo aplicas (oracion vs corpus completo).")


if __name__ == "__main__":
    main()
