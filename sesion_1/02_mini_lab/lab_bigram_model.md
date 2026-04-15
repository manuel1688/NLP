# Lab 02 — Modelos de Lenguaje con N-Gramas

**Curso:** Diplomado en Inteligencia Artificial  
**Nivel:** Introductorio  
**Duración estimada:** 15 minutos

---

## Contexto

Un **modelo de lenguaje n-grama** predice la siguiente palabra dado un contexto de *n−1* palabras anteriores.  
En este laboratorio vas a trabajar con un corpus de 13 oraciones de instrucciones de cocina en español y explorarás cómo funciona este tipo de modelo paso a paso.

El archivo de trabajo es `lab_bigram_model_solucion.py`. Ya tiene los diccionarios de conteos precalculados (`bigram_counts`, `trigram_counts`, `four_gram_counts`) y las funciones definidas. Tu tarea es **descomentar y ejecutar** cada bloque, observar los resultados y responder las preguntas.

---

## Parte 1 — Consultar N-Gramas

Las funciones `get_bigram(w1)` y `get_trigram(w1, w2)` buscan en los diccionarios de conteos y devuelven una lista de tuplas con la forma `(palabra, conteo, porcentaje)`, ordenada de mayor a menor frecuencia.

**Ejercicio 1a:** Descomentar y ejecutar.

```python
# resultado = get_bigram("la")
# print(resultado)
```

> ¿Qué palabras pueden seguir a **"la"**? ¿Cuál tiene mayor porcentaje?

**Ejercicio 1b:** Descomentar y ejecutar.

```python
# resultado = get_trigram("antes", "de")
# print(resultado)
```

> ¿Cuántas opciones aparecen después de **"antes de"**? ¿Tiene sentido dado el corpus?

---

## Parte 2 — Predecir la Siguiente Palabra

Las funciones `predecir_bigram(predicciones)` y `predecir_trim_gram(predicciones)` reciben la lista devuelta por `get_*` y retornan **solo la palabra más probable**.

La idea es encadenarlas:

```
predecir_bigram( get_bigram("el") )
```

**Ejercicio 2a:** Descomentar y ejecutar.

```python
# siguiente = predecir_bigram(get_bigram("el"))
# print(siguiente)
```

> ¿Cuál es la palabra que el modelo bigrama predice después de **"el"**?

**Ejercicio 2b:** Descomentar y ejecutar.

```python
# siguiente = predecir_trim_gram(get_trigram("el", "arroz"))
# print(siguiente)
```

> ¿Cambia la predicción al usar trigrama en lugar de bigrama? ¿Por qué crees que ocurre eso?

---

## Parte 3 — Comparar Modelos

La función `comparar_modelos(palabra)` encadena automáticamente bigrama → trigrama → 4-grama a partir de una palabra inicial y devuelve un diccionario con la predicción de cada modelo.

**Ejercicio 3a:** Descomentar y ejecutar.

```python
# comparacion = comparar_modelos("el")
# print(comparacion)
```

**Ejercicio 3b:** Descomentar y ejecutar con otra palabra.

```python
# comparacion = comparar_modelos("la")
# print(comparacion)
```

Completa la tabla con los resultados:

| Palabra inicial | Bigrama | Trigrama | 4-grama |
|-----------------|---------|----------|---------|
| `"el"`          |         |          |         |
| `"la"`          |         |          |         |

> ¿En qué casos el modelo de mayor orden da una predicción diferente? ¿Cuándo devuelve `None`?

---

## Parte 4 — Generar Oraciones

La función `generar_oracion(palabra_inicio, n_palabras)` encadena predicciones de bigramas para construir una oración completa.

**Ejercicio 4a:** Descomentar y ejecutar.

```python
# oracion = generar_oracion("mezcla", 6)
# print(oracion)
```

**Ejercicio 4b:** Descomentar y ejecutar.

```python
# oracion = generar_oracion("lava", 8)
# print(oracion)
```

> ¿Las oraciones generadas tienen sentido? ¿Por qué el modelo se detiene antes de alcanzar `n_palabras` en algunos casos?

---

## Resumen

| Función                          | Entrada                   | Salida                              |
|----------------------------------|---------------------------|-------------------------------------|
| `get_bigram(w1)`                 | 1 palabra                 | Lista `(palabra, conteo, %)`        |
| `get_trigram(w1, w2)`            | 2 palabras                | Lista `(palabra, conteo, %)`        |
| `predecir_bigram(predicciones)`  | Lista de predicciones     | Palabra más probable                |
| `predecir_trim_gram(predicciones)` | Lista de predicciones   | Palabra más probable                |
| `comparar_modelos(palabra)`      | 1 palabra                 | Dict con bigram / trigram / fourgram |
| `generar_oracion(inicio, n)`     | Palabra inicial + longitud | Oración generada                   |

---

## Para pensar

1. ¿Qué limitación tiene un modelo bigrama frente a uno de 4-gramas?
2. ¿Qué pasa cuando el modelo encuentra una combinación que no existe en el corpus?
3. ¿Por qué los porcentajes son tan bajos (menores a 5%)? ¿Qué dice eso sobre el tamaño del corpus?
