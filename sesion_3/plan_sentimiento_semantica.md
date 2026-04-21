# Plan: Sentimiento + Demostración de Cercanía Semántica

---

## Experimento 1 — Clasificación de sentimiento
**Herramientas: BoW y TF-IDF**

1. Cargar corpus Amazon reviews, binarizar, balancear y hacer split train/test.
2. Representar el corpus con **Bag of Words (BoW)**.
3. Representar el corpus con **TF-IDF**.
4. Entrenar una **Regresión Logística** sobre cada representación.
5. Comparar la **accuracy** de BoW vs TF-IDF en el conjunto de test.

---

## Experimento 2 — Demostración de cercanía semántica
**Herramienta: Word2Vec**

1. Entrenar Word2Vec con el mismo corpus (solo el texto, sin etiquetas).
2. Consultar `most_similar("excelente")` y `most_similar("horrible")`.
3. Visualizar en **PCA 2D** que ambas palabras quedan en la misma zona del espacio.

---

## Conclusión — ¿Por qué necesitamos BERT?

| Método | Qué hace bien | Por qué |
|--------|--------------|---------|
| BoW / TF-IDF | Clasifica sentimiento | Cuenta palabras clave de polaridad |
| Word2Vec | Agrupa palabras similares en contexto | Aprendió gramática, _no_ polaridad |
| BERT | Ambas cosas | Contexto completo + entrenamiento supervisado |

**El punto clave:**
Word2Vec coloca "excelente" y "horrible" **cerca** en el espacio vectorial porque
aparecen en contextos gramaticalmente similares ("el producto es _excelente_" /
"el producto es _horrible_"). Eso demuestra exactamente por qué Word2Vec solo
no basta para clasificar sentimiento.

```
BoW / TF-IDF  →  clasifican bien sentimiento     ← cuentan palabras clave

Word2Vec      →  "excelente" y "horrible"         ← aprendió gramática,
                  quedan CERCA en el espacio          no polaridad
                         ↓
     ¿por qué Word2Vec no sirve directo para sentimiento?
     exactamente por eso ↑

BERT          →  resuelve ambos problemas         ← contexto completo
                                                     + entrenamiento supervisado
```

---

## Scripts a crear

| Script | Experimento |
|--------|-------------|
| `bow_tfidf_sentimiento.py` | Experimento 1 — BoW + TF-IDF + Regresión Logística |
| `w2v_cercania_semantica.py` | Experimento 2 — Word2Vec + most_similar + PCA |
