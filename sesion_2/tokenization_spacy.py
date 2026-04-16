# ================================================================
# Tokenización con spaCy
# ================================================================
# spaCy es una libreria moderna usada en produccion.
# A diferencia de NLTK, tokeniza con un pipeline completo:
# separacion de oraciones, etiquetas gramaticales (POS), y mas.
#
# Dependencias:
#   pip3 install spacy
#   python3 -m spacy download es_core_news_sm
# ================================================================

import spacy

# TODO: busca en internet un texto corto en español (noticia, artículo, Wikipedia)
# y define aquí la variable corpus con ese texto.

# ---------------------------------------------------------------
# 1. Cargar modelo de español (pipeline completo)
# ---------------------------------------------------------------
# es_core_news_sm incluye: tokenizador, etiquetas POS, parser, NER
nlp = spacy.load("es_core_news_sm")

# Procesar el texto — genera un objeto Doc con todos los atributos
doc = nlp(corpus)

# ---------------------------------------------------------------
# 2. Tokens
# ---------------------------------------------------------------
print("=== Tokens ===")
for token in doc:
    print(token.text)

# ---------------------------------------------------------------
# 3. Tokenización de oraciones
# spaCy detecta oraciones automaticamente con el parser
# ---------------------------------------------------------------
print("\n=== Tokenización de oraciones ===")
for oracion in doc.sents:
    print(oracion.text)

# ---------------------------------------------------------------
# 4. Tokenización de palabras con atributos extra
# Cada token expone: texto, POS, si es puntuacion, si es stopword
# ---------------------------------------------------------------
print("\n=== Tokenización de palabras + atributos ===")
print(f"{'Token':<20} {'Categoría POS':<15} {'Puntuación':<12} {'Stopword'}")
print("-" * 58)
for token in doc:
    print(f"{token.text:<20} {token.pos_:<15} {str(token.is_punct):<12} {token.is_stop}")

# ---------------------------------------------------------------
# Comparacion con NLTK
# ---------------------------------------------------------------
print("\n=== Comparación spaCy vs NLTK ===")
print("NLTK word_tokenize: separa puntuacion pero no da contexto semantico.")
print("spaCy: cada token trae categoria gramatical (POS), lema, si es stopword, si es puntuacion, etc.")
print("spaCy se usa mas en produccion; NLTK es mas adecuado para aprender los fundamentos.")
