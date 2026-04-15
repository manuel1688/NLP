# ================================================================
# Tokenization in Natural Language Processing
# ================================================================
# Tokenization es el proceso de dividir un texto en unidades
# más pequenas llamadas tokens (oraciones, palabras o sub-palabras).
# ================================================================

import importlib
import subprocess
import sys

CORPUS = """Hello, My name is Pinki Dagar.
I am a student of b.tech computer science."""


def ensure_nltk_installed():
    """Instala nltk solo si no esta disponible."""
    if importlib.util.find_spec("nltk") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "-q"])


def main():
    # ---------------------------------------------------------------
    # 1. Instalacion de nltk (solo si no esta disponible)
    # ---------------------------------------------------------------
    ensure_nltk_installed()

    import nltk
    from nltk.tokenize import TreebankWordTokenizer, sent_tokenize, word_tokenize

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

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
    # ---------------------------------------------------------------
    print("\n=== Word Tokenization ===")
    for sentence in documents:
        print(word_tokenize(sentence))

    # ---------------------------------------------------------------
    # 5. Treebank Word Tokenizer
    # Usa convenciones del Penn Treebank: maneja la puntuacion
    # (como el punto al final de oracion) de forma diferente.
    # ---------------------------------------------------------------
    tokenizer = TreebankWordTokenizer()

    print("\n=== Treebank Word Tokenizer ===")
    print(tokenizer.tokenize(CORPUS))


if __name__ == "__main__":
    main()
