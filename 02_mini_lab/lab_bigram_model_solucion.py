
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
    ('lava', 'las'): 1,
    ('las', 'verduras'): 1,
    ('verduras', 'antes'): 1,
    ('antes', 'de'): 2,
    ('de', 'cortar'): 1,
    ('corta', 'el'): 1,
    ('el', 'pollo'): 2,
    ('pollo', 'en'): 2,
    ('en', 'piezas'): 1,
    ('mezcla', 'la'): 1,
    ('la', 'harina'): 1,
    ('harina', 'con'): 1,
    ('con', 'el'): 1,
    ('el', 'agua'): 2,
    ('hierve', 'el'): 1,
    ('agua', 'por'): 1,
    ('por', 'cinco'): 1,
    ('cinco', 'minutos'): 1,
    ('calienta', 'el'): 1,
    ('el', 'aceite'): 1,
    ('aceite', 'en'): 1,
    ('en', 'la'): 2,
    ('la', 'sartén'): 1,
    ('fríe', 'el'): 1,
    ('en', 'aceite'): 1,
    ('aceite', 'caliente'): 1,
    ('añade', 'sal'): 1,
    ('sal', 'al'): 1,
    ('al', 'gusto'): 2,
    ('añade', 'pimienta'): 1,
    ('pimienta', 'al'): 1,
    ('mezcla', 'el'): 1,
    ('el', 'arroz'): 3,
    ('arroz', 'con'): 2,
    ('con', 'pollo'): 2,
    ('sirve', 'el'): 1,
    ('cocina', 'el'): 1,
    ('arroz', 'en'): 1,
    ('la', 'olla'): 1,
    ('sirve', 'la'): 1,
    ('la', 'sopa'): 1,
    ('sopa', 'con'): 1,
    ('con', 'pan'): 1,
    ('deja', 'enfriar'): 1,
    ('enfriar', 'la'): 1,
    ('la', 'comida'): 1,
    ('comida', 'antes'): 1,
    ('de', 'servir'): 1
}

four_gram_counts = {
    ('lava', 'las', 'verduras', 'antes'): 1,
    ('las', 'verduras', 'antes', 'de'): 1,
    ('verduras', 'antes', 'de', 'cortar'): 1,
    ('corta', 'el', 'pollo', 'en'): 1,
    ('el', 'pollo', 'en', 'piezas'): 1,
    ('mezcla', 'la', 'harina', 'con'): 1,
    ('la', 'harina', 'con', 'el'): 1,
    ('harina', 'con', 'el', 'agua'): 1,
    ('hierve', 'el', 'agua', 'por'): 1,
    ('el', 'agua', 'por', 'cinco'): 1,
    ('agua', 'por', 'cinco', 'minutos'): 1,
    ('calienta', 'el', 'aceite', 'en'): 1,
    ('el', 'aceite', 'en', 'la'): 1,
    ('aceite', 'en', 'la', 'sartén'): 1,
    ('fríe', 'el', 'pollo', 'en'): 1,
    ('el', 'pollo', 'en', 'aceite'): 1,
    ('pollo', 'en', 'aceite', 'caliente'): 1,
    ('añade', 'sal', 'al', 'gusto'): 1,
    ('añade', 'pimienta', 'al', 'gusto'): 1,
    ('mezcla', 'el', 'arroz', 'con'): 1,
    ('sirve', 'el', 'arroz', 'con'): 1,
    ('el', 'arroz', 'con', 'pollo'): 2,
    ('cocina', 'el', 'arroz', 'en'): 1,
    ('el', 'arroz', 'en', 'la'): 1,
    ('arroz', 'en', 'la', 'olla'): 1,
    ('sirve', 'la', 'sopa', 'con'): 1,
    ('la', 'sopa', 'con', 'pan'): 1,
    ('deja', 'enfriar', 'la', 'comida'): 1,
    ('enfriar', 'la', 'comida', 'antes'): 1,
    ('la', 'comida', 'antes', 'de'): 1,
    ('comida', 'antes', 'de', 'servir'): 1
}


def get_trigram(w1, w2):
    coincidencias = {w3: count for (a, b, w3), count in trigram_counts.items() if a == w1 and b == w2}
    if not coincidencias:
        return []
    total_corpus = sum(trigram_counts.values())
    return [(w3, count, round(count / total_corpus * 100, 2)) for w3, count in sorted(coincidencias.items(), key=lambda x: -x[1])]


def get_bigram(w1):
    coincidencias = {w2: count for (a, w2), count in bigram_counts.items() if a == w1}
    if not coincidencias:
        return []
    total_corpus = sum(bigram_counts.values())
    return [(w2, count, round(count / total_corpus * 100, 2)) for w2, count in sorted(coincidencias.items(), key=lambda x: -x[1])]


def get_fourgram(w1, w2, w3):
    coincidencias = {w4: count for (a, b, c, w4), count in four_gram_counts.items() if a == w1 and b == w2 and c == w3}
    if not coincidencias:
        return []
    total_corpus = sum(four_gram_counts.values())
    return [(w4, count, round(count / total_corpus * 100, 2)) for w4, count in sorted(coincidencias.items(), key=lambda x: -x[1])]


def predecir_trim_gram(predicciones):
    if not predicciones:
        return None
    palabra, conteo, porcentaje = predicciones[0]
    return palabra


def predecir_bigram(predicciones):
    if not predicciones:
        return None
    palabra, conteo, porcentaje = predicciones[0]
    return palabra


def predecir_fourgram(predicciones):
    if not predicciones:
        return None
    palabra, conteo, porcentaje = predicciones[0]
    return palabra


if __name__ == "__main__":

    print("el")
    bi_gram = get_bigram("el")
    print(bi_gram)
    resultado_bi = predecir_bigram(bi_gram)
    print(resultado_bi)

    tri_gram = get_trigram("el", "arroz")
    print(predecir_trim_gram(tri_gram))

    four_gram = get_fourgram("el", "arroz", "con")
    print(predecir_fourgram(four_gram))

    

