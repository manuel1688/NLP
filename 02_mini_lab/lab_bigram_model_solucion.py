
corpus = [
    "lava las verduras antes de cortar",
    "corta el pollo en piezas",
    "mezcla la harina con el agua",
    "hierve el agua por cinco minutos",
    "calienta el aceite en la sartén",
    "fríe el pollo en aceite caliente",
    "añade sal al gusto",
    "añade pimienta al gusto",
    "mezcla el arroz con pollo",
    "sirve el arroz con pollo",
    "cocina el arroz en la olla", 
    "sirve la sopa con pan",
    "deja enfriar la comida antes de servir"
]

trigram_counts = {
    ('lava','las','verduras'): 1,
    ('las','verduras','antes'): 1,
    ('verduras','antes','de'): 1,
    ('antes','de','cortar'): 1,

    ('corta','el','pollo'): 1,
    ('el','pollo','en'): 1,
    ('pollo','en','piezas'): 1,

    ('mezcla','la','harina'): 1,
    ('la','harina','con'): 1,
    ('harina','con','el'): 1,
    ('con','el','agua'): 1,

    ('hierve','el','agua'): 1,
    ('el','agua','por'): 1,
    ('agua','por','cinco'): 1,
    ('por','cinco','minutos'): 1,

    ('calienta','el','aceite'): 1,
    ('el','aceite','en'): 1,
    ('aceite','en','la'): 1,
    ('en','la','sartén'): 1,

    ('fríe','el','pollo'): 1,
    ('el','pollo','en'): 1,
    ('pollo','en','aceite'): 1,
    ('en','aceite','caliente'): 1,

    ('añade','sal','al'): 1,
    ('sal','al','gusto'): 1,

    ('añade','pimienta','al'): 1,
    ('pimienta','al','gusto'): 1,

    ('mezcla','el','arroz'): 1,
    ('el','arroz','con'): 2,
    ('arroz','con','pollo'): 2,

    ('sirve','el','arroz'): 1,
    ('sirve','la','sopa'): 1,
    ('la','sopa','con'): 1,
    ('sopa','con','pan'): 1,

    ('cocina','el','arroz'): 1,
    ('el','arroz','en'): 1,
    ('arroz','en','la'): 1,
    ('en','la','olla'): 1,

    ('deja','enfriar','la'): 1,
    ('enfriar','la','comida'): 1,
    ('la','comida','antes'): 1,
    ('comida','antes','de'): 1,
    ('antes','de','servir'): 1
}

bigram_counts = {
    # Oración 1: "lava las verduras antes de cortar"
    ('lava', 'las'): 1,
    ('las', 'verduras'): 1,
    ('verduras', 'antes'): 1,
    ('antes', 'de'): 2,
    ('de', 'cortar'): 1,

    # Oración 2: "corta el pollo en piezas"
    ('corta', 'el'): 1,
    ('el', 'pollo'): 2,
    ('pollo', 'en'): 2,
    ('en', 'piezas'): 1,

    # Oración 3: "mezcla la harina con el agua"
    ('mezcla', 'la'): 1,
    ('la', 'harina'): 1,
    ('harina', 'con'): 1,
    ('con', 'el'): 1,
    ('el', 'agua'): 2,

    # Oración 4: "hierve el agua por cinco minutos"
    ('hierve', 'el'): 1,
    # ('el', 'agua') ya contado
    ('agua', 'por'): 1,
    ('por', 'cinco'): 1,
    ('cinco', 'minutos'): 1,

    # Oración 5: "calienta el aceite en la sartén"
    ('calienta', 'el'): 1,
    ('el', 'aceite'): 1,
    ('aceite', 'en'): 1,
    ('en', 'la'): 2,
    ('la', 'sartén'): 1,

    # Oración 6: "fríe el pollo en aceite caliente"
    ('fríe', 'el'): 1,
    # ('el', 'pollo') ya contado
    # ('pollo', 'en') ya contado
    ('en', 'aceite'): 1,
    ('aceite', 'caliente'): 1,

    # Oración 7: "añade sal al gusto"
    ('añade', 'sal'): 1,
    ('sal', 'al'): 1,
    ('al', 'gusto'): 2,

    # Oración 8: "añade pimienta al gusto"
    ('añade', 'pimienta'): 1,
    ('pimienta', 'al'): 1,
    # ('al', 'gusto') ya contado

    # Oración 9: "mezcla el arroz con pollo"
    ('mezcla', 'el'): 1,
    ('el', 'arroz'): 3,
    ('arroz', 'con'): 2,
    ('con', 'pollo'): 2,

    # Oración 10: "sirve el arroz con pollo"
    ('sirve', 'el'): 1,
    # ('el', 'arroz') ya contado
    # ('arroz', 'con') ya contado
    # ('con', 'pollo') ya contado

    # Oración 11: "cocina el arroz en la olla"
    ('cocina', 'el'): 1,
    # ('el', 'arroz') ya contado
    ('arroz', 'en'): 1,
    # ('en', 'la') ya contado
    ('la', 'olla'): 1,

    # Oración 12: "sirve la sopa con pan"
    ('sirve', 'la'): 1,
    ('la', 'sopa'): 1,
    ('sopa', 'con'): 1,
    ('con', 'pan'): 1,

    # Oración 13: "deja enfriar la comida antes de servir"
    ('deja', 'enfriar'): 1,
    ('enfriar', 'la'): 1,
    ('la', 'comida'): 1,
    ('comida', 'antes'): 1,
    # ('antes', 'de') ya contado
    ('de', 'servir'): 1
}

def get_predicciones_trigram(w1, w2):
    coincidencias = {w3: count for (a, b, w3), count in trigram_counts.items() if a == w1 and b == w2}
    if not coincidencias:
        return []
    total_corpus = sum(trigram_counts.values())
    return [(w3, count, round(count / total_corpus * 100, 2)) for w3, count in sorted(coincidencias.items(), key=lambda x: -x[1])]


def predecir_siguiente(predicciones):
    if not predicciones:
        return None
    palabra, conteo, porcentaje = predicciones[0]
    return palabra


if __name__ == "__main__":
    tri_gram = get_predicciones_trigram("el", "arroz")
    resultado = predecir_siguiente(tri_gram) 
    print(resultado)

