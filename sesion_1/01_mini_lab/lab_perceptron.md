# Lab 01 — El Perceptrón Original (1958)

**Curso:** Diplomado en Inteligencia Artificial  
**Nivel:** Introductorio  
**Duración estimada:** 30–45 minutos

---

## Contexto histórico

En 1958, el psicólogo Frank Rosenblatt propuso el **Perceptrón**, inspirado en cómo funciona una neurona biológica. Era la primera "neurona artificial" capaz de tomar una decisión basándose en varias entradas.

La idea era simple: cada factor que influye en una decisión tiene un *peso* que indica su importancia. El perceptrón los combina y decide: **sí (1)** o **no (0)**.

En este laboratorio vamos a simular exactamente eso, **sin librerías**, **sin entrenamiento automático**, y con los pesos elegidos a mano, tal como se hacía al principio.

---

## El escenario: ¿Voy al concierto?

Imagina que un perceptrón tiene que decidir si vas o no a un concierto esta noche, basándose en tres factores:

| Variable | Pregunta                   | Valor de ejemplo |
|----------|----------------------------|-----------------|
| `x1`     | ¿Tengo dinero?             | `1` (Sí)        |
| `x2`     | ¿Van mis amigos?           | `0` (No)        |
| `x3`     | ¿Va a llover esta noche?   | `1` (Sí)        |

Los valores son **binarios**: `1` significa que el factor se cumple, `0` que no.

---

## Los pesos y el sesgo

No todos los factores influyen igual. Aquí le asignamos manualmente la importancia de cada uno:

| Parámetro | Valor | Interpretación                                          |
|-----------|-------|---------------------------------------------------------|
| `w1`      | `6`   | El dinero es muy importante (positivo y alto)           |
| `w2`      | `2`   | Que vayan tus amigos importa, pero no decide solo       |
| `w3`      | `-4`  | La lluvia es un factor negativo que te desanima         |
| `bias`    | `-3`  | Eres un poco perezoso: el resultado debe ser muy positivo para que salgas |

El **sesgo (bias)** es una constante que representa una predisposición. Un bias negativo hace que la decisión "cueste más" ser un 1.

---

## Los 5 pasos del perceptrón

### Paso 1 — Entradas

Definimos las tres entradas con los valores del escenario:

```
x1 = 1   # Tengo dinero: Sí
x2 = 0   # Van mis amigos: No
x3 = 1   # Va a llover: Sí
```

### Paso 2 — Pesos y sesgo

```
w1 = 6
w2 = 2
w3 = -4
bias = -3
```

### Paso 3 — Suma ponderada (z)

Multiplicamos cada entrada por su peso y sumamos el sesgo:

$$z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + (x_3 \cdot w_3) + \text{bias}$$

Con nuestros valores:

$$z = (1 \cdot 6) + (0 \cdot 2) + (1 \cdot -4) + (-3) = 6 + 0 - 4 - 3 = -1$$

### Paso 4 — Función de activación (el umbral)

El perceptrón original usa una **función de paso**:

- Si `z > 0` → salida = `1` → **Vas al concierto**
- Si `z ≤ 0` → salida = `0` → **Te quedas en casa**

### Paso 5 — Resultado

Como `z = -1`, que es ≤ 0, la salida es **0**.

**Conclusión del perceptrón:** No vas al concierto. Aunque tienes dinero, la lluvia y tu propia pereza pesan más.

---

## Tu turno: escribe el código

Crea un archivo llamado `mi_perceptron.py` y sigue estos pasos:

1. **Declara las entradas.** Crea tres variables `x1`, `x2` y `x3` con los valores del escenario.

2. **Declara los pesos y el sesgo.** Crea variables `w1`, `w2`, `w3` y `bias` con los valores de la tabla anterior.

3. **Calcula la suma ponderada.** Crea una variable `z` que sea el resultado de:
   `(x1 * w1) + (x2 * w2) + (x3 * w3) + bias`

4. **Aplica la función de activación.** Usa un `if`/`else` para asignar `1` o `0` a una variable `salida`, según si `z > 0` o no.

5. **Imprime el resultado.** Muestra un mensaje legible, por ejemplo:
   - Si `salida == 1`: imprime `"Vas al concierto"`
   - Si `salida == 0`: imprime `"Te quedas en casa"`

6. **Ejecuta tu script** con:
   ```
   python mi_perceptron.py
   ```
   Deberías ver: `Te quedas en casa`

---

## Preguntas de reflexión

Después de que tu script funcione, modifica los valores y observa qué cambia:

1. **¿Qué pasa si `x2 = 1`?** (Tus amigos sí van al concierto.)  
   Calcula z a mano antes de correr el código. ¿Coincide con tu predicción?

2. **¿Qué pasa si cambias `w3` de `-4` a `-1`?**  
   ¿Qué significa esto en términos del "modelo mental" de la persona?

3. **¿Qué valor mínimo necesita tener `x2` para que la decisión cambie a 1, manteniendo todo lo demás igual?**  
   (Pista: `x2` solo puede ser 0 o 1 en un perceptrón binario. ¿Es posible lograrlo solo con `x2`?)

4. **¿En qué se diferencia este script del archivo `perceptron.py` que tiene un `for epoch in range(10)`?**  
   En este lab los pesos los elegiste tú. En el otro, ¿quién los elige?

---

## Notas

- No necesitas importar ninguna librería. Todo es Python puro.
- Los valores de los pesos son arbitrarios en este lab. En un sistema real, el algoritmo de entrenamiento los aprende automáticamente a partir de datos.
- Este es el perceptrón de **Rosenblatt (1958)**. No tiene capas ocultas, no tiene entrenamiento en este ejercicio, y solo puede representar decisiones linealmente separables.
