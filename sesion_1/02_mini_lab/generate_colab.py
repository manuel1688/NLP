"""
Genera el notebook de Google Colab para el Lab 02 — N-Gramas.
Uso: python generate_colab.py
Salida: Lab02_NGramas_Colab.ipynb
"""
import json, uuid

def _id():
    return str(uuid.uuid4())[:8]

def md(text):
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": source}

def code(text):
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": _id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": source}

# ─────────────────────────────────────────────────────────────────────────────
SETUP = '''\
corpus = [
    'lava las verduras antes de cortar',
    'corta el pollo en piezas',
    'mezcla la harina con el agua',
    'hierve el agua por cinco minutos',
    'calienta el aceite en la sartén',
    'fríe el pollo en aceite caliente',
    'añade sal al gusto',
    'añade pimienta al gusto',
    'mezcla el arroz con pollo',
    'sirve el arroz con pollo',
    'cocina el arroz en la olla',
    'sirve la sopa con pan',
    'deja enfriar la comida antes de servir',
]

trigram_counts = {
    ('lava','las','verduras'): 1, ('las','verduras','antes'): 1,
    ('verduras','antes','de'): 1, ('antes','de','cortar'): 1,
    ('corta','el','pollo'): 1,   ('el','pollo','en'): 1,
    ('pollo','en','piezas'): 1,  ('mezcla','la','harina'): 1,
    ('la','harina','con'): 1,    ('harina','con','el'): 1,
    ('con','el','agua'): 1,      ('hierve','el','agua'): 1,
    ('el','agua','por'): 1,      ('agua','por','cinco'): 1,
    ('por','cinco','minutos'): 1,('calienta','el','aceite'): 1,
    ('el','aceite','en'): 1,     ('aceite','en','la'): 1,
    ('en','la','sartén'): 1,     ('fríe','el','pollo'): 1,
    ('pollo','en','aceite'): 1,  ('en','aceite','caliente'): 1,
    ('añade','sal','al'): 1,     ('sal','al','gusto'): 1,
    ('añade','pimienta','al'): 1,('pimienta','al','gusto'): 1,
    ('mezcla','el','arroz'): 1,  ('el','arroz','con'): 2,
    ('arroz','con','pollo'): 2,  ('sirve','el','arroz'): 1,
    ('sirve','la','sopa'): 1,    ('la','sopa','con'): 1,
    ('sopa','con','pan'): 1,     ('cocina','el','arroz'): 1,
    ('el','arroz','en'): 1,      ('arroz','en','la'): 1,
    ('en','la','olla'): 1,       ('deja','enfriar','la'): 1,
    ('enfriar','la','comida'): 1,('la','comida','antes'): 1,
    ('comida','antes','de'): 1,  ('antes','de','servir'): 1,
}

bigram_counts = {
    ('lava','las'): 1,    ('las','verduras'): 1,  ('verduras','antes'): 1,
    ('antes','de'): 2,    ('de','cortar'): 1,     ('corta','el'): 1,
    ('el','pollo'): 2,    ('pollo','en'): 2,      ('en','piezas'): 1,
    ('mezcla','la'): 1,   ('la','harina'): 1,     ('harina','con'): 1,
    ('con','el'): 1,      ('el','agua'): 2,       ('hierve','el'): 1,
    ('agua','por'): 1,    ('por','cinco'): 1,     ('cinco','minutos'): 1,
    ('calienta','el'): 1, ('el','aceite'): 1,     ('aceite','en'): 1,
    ('en','la'): 2,       ('la','sartén'): 1,     ('fríe','el'): 1,
    ('en','aceite'): 1,   ('aceite','caliente'): 1,('añade','sal'): 1,
    ('sal','al'): 1,      ('al','gusto'): 2,      ('añade','pimienta'): 1,
    ('pimienta','al'): 1, ('mezcla','el'): 1,     ('el','arroz'): 3,
    ('arroz','con'): 2,   ('con','pollo'): 2,     ('sirve','el'): 1,
    ('cocina','el'): 1,   ('arroz','en'): 1,      ('la','olla'): 1,
    ('sirve','la'): 1,    ('la','sopa'): 1,       ('sopa','con'): 1,
    ('con','pan'): 1,     ('deja','enfriar'): 1,  ('enfriar','la'): 1,
    ('la','comida'): 1,   ('comida','antes'): 1,  ('de','servir'): 1,
}

four_gram_counts = {
    ('lava','las','verduras','antes'): 1,  ('las','verduras','antes','de'): 1,
    ('verduras','antes','de','cortar'): 1, ('corta','el','pollo','en'): 1,
    ('el','pollo','en','piezas'): 1,       ('mezcla','la','harina','con'): 1,
    ('la','harina','con','el'): 1,         ('harina','con','el','agua'): 1,
    ('hierve','el','agua','por'): 1,       ('el','agua','por','cinco'): 1,
    ('agua','por','cinco','minutos'): 1,   ('calienta','el','aceite','en'): 1,
    ('el','aceite','en','la'): 1,          ('aceite','en','la','sartén'): 1,
    ('fríe','el','pollo','en'): 1,         ('el','pollo','en','aceite'): 1,
    ('pollo','en','aceite','caliente'): 1, ('añade','sal','al','gusto'): 1,
    ('añade','pimienta','al','gusto'): 1,  ('mezcla','el','arroz','con'): 1,
    ('sirve','el','arroz','con'): 1,       ('el','arroz','con','pollo'): 2,
    ('cocina','el','arroz','en'): 1,       ('el','arroz','en','la'): 1,
    ('arroz','en','la','olla'): 1,         ('sirve','la','sopa','con'): 1,
    ('la','sopa','con','pan'): 1,          ('deja','enfriar','la','comida'): 1,
    ('enfriar','la','comida','antes'): 1,  ('la','comida','antes','de'): 1,
    ('comida','antes','de','servir'): 1,
}

print(f"Corpus cargado: {len(corpus)} oraciones")
print(f"Bigramas: {len(bigram_counts)}  |  Trigramas: {len(trigram_counts)}  |  4-gramas: {len(four_gram_counts)}")'''

FUNCIONES = '''\
# Busca bigramas que empiecen con w1 → devuelve [(palabra, conteo, %), ...]
def get_bigram(w1):
    coincidencias = {w2: c for (a, w2), c in bigram_counts.items() if a == w1}
    if not coincidencias:
        return []
    total = sum(bigram_counts.values())
    return [(w2, c, round(c/total*100, 2)) for w2, c in sorted(coincidencias.items(), key=lambda x: -x[1])]

# Busca trigramas que empiecen con (w1, w2) → devuelve [(palabra, conteo, %), ...]
def get_trigram(w1, w2):
    coincidencias = {w3: c for (a, b, w3), c in trigram_counts.items() if a == w1 and b == w2}
    if not coincidencias:
        return []
    total = sum(trigram_counts.values())
    return [(w3, c, round(c/total*100, 2)) for w3, c in sorted(coincidencias.items(), key=lambda x: -x[1])]

# Busca 4-gramas que empiecen con (w1, w2, w3) → devuelve [(palabra, conteo, %), ...]
def get_fourgram(w1, w2, w3):
    coincidencias = {w4: c for (a, b, cc, w4), c in four_gram_counts.items() if a==w1 and b==w2 and cc==w3}
    if not coincidencias:
        return []
    total = sum(four_gram_counts.values())
    return [(w4, c, round(c/total*100, 2)) for w4, c in sorted(coincidencias.items(), key=lambda x: -x[1])]

# Recibe lista de predicciones y retorna la palabra más probable
def predecir_bigram(predicciones):
    return predicciones[0][0] if predicciones else None

def predecir_trim_gram(predicciones):
    return predicciones[0][0] if predicciones else None

def predecir_fourgram(predicciones):
    return predicciones[0][0] if predicciones else None

# Genera una oración encadenando predicciones de bigramas
def generar_oracion(palabra_inicio, n_palabras):
    palabras = [palabra_inicio]
    for _ in range(n_palabras - 1):
        siguiente = predecir_bigram(get_bigram(palabras[-1]))
        if siguiente is None:
            break
        palabras.append(siguiente)
    return ' '.join(palabras)

# Compara predicciones de bigrama, trigrama y 4-grama desde una palabra inicial
def comparar_modelos(palabra):
    bi  = predecir_bigram(get_bigram(palabra))
    tri = predecir_trim_gram(get_trigram(palabra, bi)) if bi else None
    four = predecir_fourgram(get_fourgram(palabra, bi, tri)) if bi and tri else None
    return {'bigram': bi, 'trigram': tri, 'fourgram': four}

print("✓ Funciones cargadas correctamente")'''

# ─────────────────────────────────────────────────────────────────────────────
cells = [
    md(
        "# Lab 02 — Modelos de Lenguaje con N-Gramas\n"
        "\n"
        "**Curso:** Diplomado en Inteligencia Artificial  \n"
        "**Nivel:** Introductorio  \n"
        "**Duración estimada:** 15 minutos\n"
        "\n"
        "---"
    ),
    md(
        "## Contexto\n"
        "\n"
        "Un **modelo de lenguaje n-grama** predice la siguiente palabra dado un contexto de *n−1* palabras anteriores.\n"
        "\n"
        "En este lab vas a trabajar con un corpus de 13 oraciones de instrucciones de cocina en español.\n"
        "Explorarás cómo funciona este tipo de modelo paso a paso.\n"
        "\n"
        "> **Instrucción general:** Ejecuta cada celda de configuración (▶). Luego descomentar el código "
        "de cada ejercicio, ejecutarlo y responder las preguntas en las celdas de texto."
    ),
    md("## ⚙️ Configuración — Ejecuta esta celda primero"),
    code(SETUP),
    md("## ⚙️ Funciones — Ejecuta esta celda después del corpus"),
    code(FUNCIONES),

    # ── Parte 1 ──────────────────────────────────────────────────────────────
    md(
        "---\n"
        "## Parte 1 — Consultar N-Gramas\n"
        "\n"
        "Las funciones `get_bigram(w1)` y `get_trigram(w1, w2)` buscan en los diccionarios de conteos "
        "y devuelven una lista de tuplas `(palabra, conteo, porcentaje)`, ordenada de mayor a menor frecuencia.\n"
        "\n"
        "**Ejercicio 1a:** Descomentar, ejecutar y observar el resultado."
    ),
    code(
        "# resultado = get_bigram('la')\n"
        "# print(resultado)"
    ),
    md("> ¿Qué palabras pueden seguir a **\"la\"**? ¿Cuál tiene mayor porcentaje?"),
    md("**Ejercicio 1b:** Descomentar, ejecutar y observar el resultado."),
    code(
        "# resultado = get_trigram('antes', 'de')\n"
        "# print(resultado)"
    ),
    md("> ¿Cuántas opciones aparecen después de **\"antes de\"**? ¿Tiene sentido dado el corpus?"),

    # ── Parte 2 ──────────────────────────────────────────────────────────────
    md(
        "---\n"
        "## Parte 2 — Predecir la Siguiente Palabra\n"
        "\n"
        "Las funciones `predecir_bigram()` y `predecir_trim_gram()` reciben la lista de `get_*` "
        "y retornan **solo la palabra más probable**. La idea es encadenarlas:\n"
        "\n"
        "```\n"
        "predecir_bigram( get_bigram('el') )\n"
        "```\n"
        "\n"
        "**Ejercicio 2a:** Descomentar y ejecutar."
    ),
    code(
        "# siguiente = predecir_bigram(get_bigram('el'))\n"
        "# print(siguiente)"
    ),
    md("> ¿Cuál es la palabra que el modelo bigrama predice después de **\"el\"**?"),
    md("**Ejercicio 2b:** Descomentar y ejecutar."),
    code(
        "# siguiente = predecir_trim_gram(get_trigram('el', 'arroz'))\n"
        "# print(siguiente)"
    ),
    md("> ¿Cambia la predicción al usar trigrama en lugar de bigrama? ¿Por qué crees que ocurre eso?"),

    # ── Parte 3 ──────────────────────────────────────────────────────────────
    md(
        "---\n"
        "## Parte 3 — Comparar Modelos\n"
        "\n"
        "`comparar_modelos(palabra)` encadena bigrama → trigrama → 4-grama a partir de una palabra inicial "
        "y devuelve un diccionario con la predicción de cada modelo.\n"
        "\n"
        "**Ejercicio 3a:** Descomentar y ejecutar."
    ),
    code(
        "# comparacion = comparar_modelos('el')\n"
        "# print(comparacion)"
    ),
    md("**Ejercicio 3b:** Descomentar y ejecutar con otra palabra."),
    code(
        "# comparacion = comparar_modelos('la')\n"
        "# print(comparacion)"
    ),
    md(
        "Completa la tabla con tus resultados:\n"
        "\n"
        "| Palabra inicial | Bigrama | Trigrama | 4-grama |\n"
        "|-----------------|---------|----------|---------|\n"
        "| `'el'`          |         |          |         |\n"
        "| `'la'`          |         |          |         |\n"
        "\n"
        "> ¿En qué casos el modelo de mayor orden da una predicción diferente? ¿Cuándo devuelve `None`?"
    ),

    # ── Parte 4 ──────────────────────────────────────────────────────────────
    md(
        "---\n"
        "## Parte 4 — Generar Oraciones\n"
        "\n"
        "`generar_oracion(palabra_inicio, n_palabras)` encadena predicciones de bigramas para construir "
        "una oración completa.\n"
        "\n"
        "**Ejercicio 4a:** Descomentar y ejecutar."
    ),
    code(
        "# oracion = generar_oracion('mezcla', 6)\n"
        "# print(oracion)"
    ),
    md("**Ejercicio 4b:** Descomentar y ejecutar."),
    code(
        "# oracion = generar_oracion('lava', 8)\n"
        "# print(oracion)"
    ),
    md("> ¿Las oraciones generadas tienen sentido? ¿Por qué el modelo se detiene antes de alcanzar `n_palabras` en algunos casos?"),

    # ── Resumen ───────────────────────────────────────────────────────────────
    md(
        "---\n"
        "## Resumen\n"
        "\n"
        "| Función | Entrada | Salida |\n"
        "|---------|---------|--------|\n"
        "| `get_bigram(w1)` | 1 palabra | Lista `(palabra, conteo, %)` |\n"
        "| `get_trigram(w1, w2)` | 2 palabras | Lista `(palabra, conteo, %)` |\n"
        "| `predecir_bigram(predicciones)` | Lista | Palabra más probable |\n"
        "| `predecir_trim_gram(predicciones)` | Lista | Palabra más probable |\n"
        "| `comparar_modelos(palabra)` | 1 palabra | Dict bigram/trigram/fourgram |\n"
        "| `generar_oracion(inicio, n)` | Palabra + longitud | Oración generada |\n"
        "\n"
        "---\n"
        "## Para pensar\n"
        "\n"
        "1. ¿Qué limitación tiene un modelo bigrama frente a uno de 4-gramas?\n"
        "2. ¿Qué pasa cuando el modelo encuentra una combinación que no existe en el corpus?\n"
        "3. ¿Por qué los porcentajes son tan bajos (< 5%)? ¿Qué dice eso sobre el tamaño del corpus?"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": cells,
}

OUTPUT = "Lab02_NGramas_Colab.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Notebook generado: {OUTPUT}")
print(f"  {len(cells)} celdas  —  listo para abrir en Google Colab")
print()
print("Para abrirlo en Colab:")
print("  1. Ve a https://colab.research.google.com")
print("  2. Archivo → Subir notebook → selecciona Lab02_NGramas_Colab.ipynb")
