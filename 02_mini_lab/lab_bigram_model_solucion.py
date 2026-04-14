corpus = [
    "agrega sal al agua",
    "agrega sal y pimienta",
    "añade sal al gusto",
    "añade azúcar al gusto",
    "añade pimienta al gusto",
    "añade sal al gusto",
    "añade azúcar al gusto",
    "sirve la comida antes de servir",
    "sirve la sopa antes de servir",
    "calienta la sopa antes de servir",
    "deja enfriar la comida antes de servir",
    "lava las verduras antes de servir",
    "sirve el plato antes de servir",
    "sirve el arroz con pollo",
    "sirve el arroz con pollo",
    "sirve el arroz con pollo",
    "mezcla el arroz con pollo",
    "calienta el arroz con pollo",
    "calienta el aceite en la sartén",
    "fríe el huevo en la sartén",
    "agrega aceite en la sartén",
    "calienta aceite en la sartén",
    "sofríe el pollo en la sartén",
    "fríe las papas con aceite caliente",
    "fríe el pollo con aceite caliente",
    "fríe el pescado con aceite caliente",
    "fríe las verduras con aceite caliente",
    "mezcla la harina con el agua",
    "mezcla la harina con el agua",
    "mezcla la harina con el agua",
    "corta el pollo en piezas",
    "corta el pollo en piezas",
    "corta el pollo en piezas"
]

trigram_counts = {
    ('añade','sal','al'): 2,
    ('sal','al','gusto'): 4,
    ('añade','azúcar','al'): 2,
    ('azúcar','al','gusto'): 2,
    ('añade','pimienta','al'): 1,
    ('antes','de','servir'): 6,
    ('el','arroz','con'): 5,
    ('arroz','con','pollo'): 5,
    ('en','la','sartén'): 5,
    ('con','aceite','caliente'): 4,
    ('la','harina','con'): 3,
    ('harina','con','el'): 3,
    ('con','el','agua'): 3,
    ('el','pollo','en'): 3,
    ('pollo','en','piezas'): 3
}

bigram_counts = {
    ('antes','de'): 6,
    ('de','servir'): 6,
    ('al','gusto'): 5,
    ('el','arroz'): 5,
    ('el','pollo'): 5,
    ('arroz','con'): 5,
    ('con','pollo'): 5,
    ('en','la'): 5,
    ('la','sartén'): 5,
    ('aceite','caliente'): 4,
    ('con','aceite'): 4,
    ('pollo','en'): 4,
    ('sirve','el'): 4,
    ('aceite','en'): 3,
    ('con','el'): 3,
    ('corta','el'): 3,
    ('el','agua'): 3,
    ('en','piezas'): 3,
    ('fríe','el'): 3,
    ('harina','con'): 3,
    ('la','harina'): 3,
    ('mezcla','la'): 3,
    ('sal','al'): 3,
    ('agrega','sal'): 2,
    ('añade','azúcar'): 2,
    ('añade','sal'): 2,
    ('azúcar','al'): 2,
    ('calienta','el'): 2,
    ('comida','antes'): 2,
    ('fríe','las'): 2,
    ('la','comida'): 2,
    ('la','sopa'): 2,
    ('las','verduras'): 2,
    ('sirve','la'): 2,
    ('sopa','antes'): 2,
}

def predecir_siguiente_trigram(w1, w2):
    coincidencias = {w3: count for (a, b, w3), count in trigram_counts.items() if a == w1 and b == w2}
    total = sum(coincidencias.values())
    if total == 0:
        return []
    return [(w3, round(count / total * 100, 2)) for w3, count in sorted(coincidencias.items(), key=lambda x: -x[1])]


if __name__ == "__main__":
    ejemplos = [
        ("antes", "de"),
        ("añade", "sal"),
        ("el", "arroz"),
        ("con", "aceite"),
        ("sal", "al"),
        ("la", "harina"),
        ("hola", "mundo"),  # no existe en el corpus
    ]

    for w1, w2 in ejemplos:
        resultado = predecir_siguiente_trigram(w1, w2)
        print(f"predecir_siguiente_trigram('{w1}', '{w2}')")
        if resultado:
            for palabra, porcentaje in resultado:
                print(f"  → '{palabra}': {porcentaje}%")
        else:
            print("  → (sin resultados)")
        print()


