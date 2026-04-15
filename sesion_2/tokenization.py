# ================================================================
# Tokenization in Natural Language Processing
# ================================================================
# Tokenization es el proceso de dividir un texto en unidades
# más pequeñas llamadas tokens (oraciones, palabras o sub-palabras).
# ================================================================

import subprocess
import sys

# ---------------------------------------------------------------
# 1. Instalación de nltk (solo si no está disponible)
# ---------------------------------------------------------------
subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "-q"])

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------
# 2. Definición del corpus
# ---------------------------------------------------------------
corpus = """Hello, My name is Pinki Dagar. 
I am a student of b.tech computer science."""

print("=== Corpus ===")
print(corpus)

# ---------------------------------------------------------------
# 3. Sentence Tokenization
# Divide el corpus en oraciones individuales.
# ---------------------------------------------------------------
from nltk.tokenize import sent_tokenize

documents = sent_tokenize(corpus)

print("\n=== Sentence Tokenization ===")
for sentence in documents:
    print(sentence)

# ---------------------------------------------------------------
# 4. Word Tokenization
# Divide cada oración en palabras y signos de puntuación.
# ---------------------------------------------------------------
from nltk.tokenize import word_tokenize

print("\n=== Word Tokenization ===")
for sentence in documents:
    print(word_tokenize(sentence))

# ---------------------------------------------------------------
# 5. Treebank Word Tokenizer
# Usa convenciones del Penn Treebank: maneja la puntuación
# (como el punto al final de oración) de forma diferente.
# ---------------------------------------------------------------
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

print("\n=== Treebank Word Tokenizer ===")
print(tokenizer.tokenize(corpus))
