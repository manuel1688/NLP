# 📖 Word2Vec — Skip-gram: Ejemplo Numérico Completo

> Un solo paso de entrenamiento, de principio a fin.
> Los mismos números se usan en **todos** los pasos. Nada aparece de la nada.

---

## 📚 Glosario

> [!TIP]
> Lee esto primero — cada término aparece en los pasos siguientes. Puedes volver aquí si alguno no queda claro.

| Término | Qué es |
|---|---|
| `W_embed` | Matriz **(V × D)**. Cada fila es el vector de una palabra cuando actúa como **objetivo** (target). La fila `i` corresponde a la palabra con índice `i`. |
| `W_context` | Matriz **(V × D)**. Igual que `W_embed` pero para cuando la palabra actúa como **contexto**. Son dos matrices separadas — cada palabra tiene dos vectores. |
| `v_t` | El vector del target en el paso actual. Se obtiene con un lookup: `v_t = W_embed[idx_target]`. |
| `dot product` | `dot(a, b) = a[0]×b[0] + a[1]×b[1] + ...` — suma de productos elemento a elemento. Mide qué tan "alineados" están dos vectores. Score alto = palabras relacionadas. |
| `score` | Resultado del producto punto entre `v_t` y una fila de `W_context`. Un número — puede ser positivo o negativo. |
| `σ` (sigmoid) | `σ(x) = 1 / (1 + e^{-x})` — convierte cualquier score en un valor entre 0 y 1. Aquí = probabilidad de que el par sea real. |
| `Loss` | Número que mide cuánto se equivocó el modelo en este paso. El objetivo del entrenamiento es reducirlo hacia 0. |
| `Gradiente` | Vector que dice **en qué dirección y cuánto** cambiar los pesos para reducir el Loss. Se calcula con la regla de la cadena. |
| `SGD` | Regla de actualización: `W -= lr × gradiente`. Da un pequeño paso en la dirección que reduce el Loss. |
| `lr` (learning rate) | Controla el tamaño del paso SGD. Muy alto → inestable. Muy bajo → aprende lento. Típico: `0.025`. |
| `Negative Sampling` | Técnica que en vez de calcular softmax sobre todo el vocabulario, entrena comparando **un par real** contra **K pares falsos** aleatorios. Mucho más rápido. |
| `par positivo` | Par *(target, contexto)* que sí aparece en el corpus. Label = `1`. |
| `par negativo` | Par *(target, palabra_aleatoria)* inventado. Label = `0`. El modelo debe aprender que es falso. |

---

## 0. 🎯 Punto de Partida

### Vocabulario

| Palabra  | Índice |
|----------|--------|
| `come`   | `0`    |
| `gato`   | `1`    |
| `camina` | `2`    |
| `perro`  | `3`    |

### Par de entrenamiento de este paso

| Rol        | Palabra   | Índice | Descripción                          |
|------------|-----------|--------|--------------------------------------|
| **target** | `"perro"` | `3`    | Palabra que damos al modelo          |
| **contexto** | `"come"` | `0`   | Palabra que debería predecir         |

### Muestras negativas *(label = 0, pares falsos)*

| #           | Palabra    | Índice |
|-------------|------------|--------|
| negativa 1  | `"gato"`   | `1`    |
| negativa 2  | `"camina"` | `2`    |

### Hiperparámetros

| Parámetro | Valor | Significado              |
|-----------|-------|--------------------------|
| `D`       | `2`   | Dimensión del embedding  |
| `lr`      | `0.1` | Learning rate (SGD)      |

---

## 1. 🔧 Inicialización de Matrices

Dos matrices de dimensión **(V × D) = (4 × 2)** — una fila por palabra.

### ¿De dónde salen los valores de estas matrices?

| Situación | ¿De dónde vienen los valores? |
|---|---|
| **En este ejemplo** | Elegidos a mano para producir gradientes no triviales y números legibles |
| **Día 0, entrenamiento real** | `np.random.normal(0, 0.01, (V, D))` — pequeños valores aleatorios, sin ningún significado aún |
| **Después de entrenar** | Modificados paso a paso por el gradiente (Pasos 5-6 de este documento) — las filas de palabras que aparecen en contextos similares convergen hacia vectores parecidos |

```python
# Inicialización real en código (V palabras, D dimensiones)
V, D = 28130, 100
W_embed   = np.random.normal(0, 0.01, (V, D))   # pequeños aleatorios
W_context = np.zeros((V, D))                     # ceros — convención común
```

> [!IMPORTANT]
> **`W_embed` es el modelo aprendido.** La fila `i` no representa a la palabra `i`
> desde el principio — empieza como números aleatorios sin sentido.
> Es el entrenamiento (miles de pasos como el de este ejemplo) lo que convierte
> esa fila en un vector con significado.
> En este ejemplo usamos valores no-cero en `W_context` también,
> para que los gradientes sean ilustrativos al llegar al Paso 5.

### `W_embed` — Matriz de palabras objetivo

| Fila | Palabra    | Vector            | Uso              |
|------|------------|-------------------|------------------|
| `0`  | `"come"`   | `[ 0.05,  0.22]`  |                  |
| `1`  | `"gato"`   | `[ 0.42,  0.11]`  |                  |
| `2`  | `"camina"` | `[-0.09,  0.33]`  |                  |
| `3`  | `"perro"`  | `[ 0.13, -0.27]`  | ← **usaremos esta** |

#### ¿Qué representa cada fila?

Cada fila es la **posición** de una palabra en un espacio de `D` dimensiones.
Con `D = 2`, cada palabra es un punto en un plano. Con `D = 100`, es un punto
en un espacio de 100 dimensiones (no visualizable, pero matemáticamente idéntico).

```
  Antes de entrenar           Después de entrenar
  (aleatorio)                 (aprendido)

  perro  •  • gato            perro • gato
                               ↑    ↑
  camina •   • come           (cerca — mamíferos)

                              camina •    come •
                              (lejos de los anteriores)
```

> [!TIP]
> En el **Paso 6** de este documento verás exactamente cómo el gradiente
> mueve la fila `[0.13, -0.27]` de "perro" una fracción hacia una posición
> que reduce el error. Ese movimiento repetido millones de veces es lo que
> produce embeddings útiles.

### `W_context` — Matriz de palabras contexto

| Fila | Palabra    | Vector            | Uso              |
|------|------------|-------------------|------------------|
| `0`  | `"come"`   | `[ 0.05,  0.22]`  | ← par positivo   |
| `1`  | `"gato"`   | `[ 0.30, -0.10]`  | ← negativa 1     |
| `2`  | `"camina"` | `[-0.15,  0.08]`  | ← negativa 2     |
| `3`  | `"perro"`  | `[ 0.01, -0.05]`  |                  |

#### ¿Por qué dos matrices y no una?

En Skip-gram cada palabra juega dos roles distintos:

| Rol | Cuándo | Acción | Matriz |
|---|---|---|---|
| **Target** | Es la palabra que damos al modelo | Su vector actúa como "pregunta" | `W_embed` |
| **Contexto** | Es la palabra que el modelo intenta predecir | Su vector actúa como "respuesta" | `W_context` |

Si usáramos una sola matriz, el vector de "perro" haría los dos roles al mismo tiempo.
Esto hace que los gradientes se interfieran entre sí y el modelo aprende mal.

> [!NOTE]
> **En producción**, al terminar el entrenamiento se descarta `W_context`
> y solo se usa `W_embed`. Las dos matrices son necesarias **durante** el
> entrenamiento, pero el producto final es únicamente `W_embed`.

---

## 2. ⚡ Forward Pass — Calcular Scores

### ¿Qué es un score y para qué sirve?

Un **score** es un único número que mide qué tan compatibles son dos vectores —
en este caso, el vector del target y el vector de una palabra de contexto.

La función que lo calcula es el **producto punto** (dot product): multiplica
los elementos en la misma posición y los suma.

```
score = dot(v_target, v_contexto)
      = v[0]×c[0]  +  v[1]×c[1]  +  ...  +  v[D-1]×c[D-1]
```

Es una **operación lineal** — no hay exponenciales, raíces, ni nada no-lineal.
Solo multiplicaciones y sumas.

#### ¿Qué significa el valor del score?

| Score | Interpretación |
|---|---|
| **Positivo alto** (`+2.5`) | Los vectores apuntan en la misma dirección → las palabras son compatibles |
| **Cercano a 0** (`0.01`) | Los vectores son casi perpendiculares → sin relación clara |
| **Negativo** (`-1.3`) | Los vectores apuntan en direcciones opuestas → las palabras son incompatibles |

Al inicio del entrenamiento los scores son casi cero (porque los pesos son pequeños aleatorios).
A medida que el modelo aprende, el score del **par positivo** sube y los de los **pares negativos** bajan.

> [!NOTE]
> El score por sí solo no es una probabilidad — puede ser cualquier número real.
> Es el **sigmoid** (Paso 3) el que lo convierte en un valor entre 0 y 1.

---

### Paso 2a — Lookup del vector target

> [!TIP]
> **One-hot vs lookup:** matemáticamente `v_t = W_embed · one_hot([0,0,0,1])`.
> En código siempre se usa el lookup directo: es equivalente y mucho más rápido.

```python
v_t = W_embed[3]      # extraer fila 3 directamente
v_t = [0.13, -0.27]
```

### Paso 2b — Score par **positivo** (`perro` → `come`)

```python
score_pos = dot(v_t, W_context[0])
          = dot([0.13, -0.27], [0.05, 0.22])
          = (0.13 × 0.05) + (-0.27 × 0.22)
          =  0.0065        +  (-0.0594)
          = -0.053
```

### Paso 2c — Score negativa 1 (`perro` → `gato`)

```python
score_neg1 = dot(v_t, W_context[1])
           = dot([0.13, -0.27], [0.30, -0.10])
           = (0.13 × 0.30) + (-0.27 × -0.10)
           =  0.039         +   0.027
           =  0.066
```

### Paso 2d — Score negativa 2 (`perro` → `camina`)

```python
score_neg2 = dot(v_t, W_context[2])
           = dot([0.13, -0.27], [-0.15, 0.08])
           = (0.13 × -0.15) + (-0.27 × 0.08)
           = -0.0195         +  (-0.0216)
           = -0.041
```

### Resumen de scores

| Par                    | Score    |
|------------------------|----------|
| `score_pos`  (`come`)   | `-0.053` |
| `score_neg1` (`gato`)   | ` 0.066` |
| `score_neg2` (`camina`) | `-0.041` |

---

## 3. 📊 Sigmoid — Convertir Scores en Probabilidades

### Softmax vs Sigmoid — ¿cuál usar y por qué?

Hay dos formas de convertir scores en probabilidades. Ambas son válidas; se usan en contextos distintos.

#### Opción A — Softmax (modelo original Word2Vec)

Toma **todos** los scores del vocabulario y los convierte en una distribución de probabilidad que suma 1.

```
softmax(zᵢ) = e^{zᵢ} / Σⱼ e^{zⱼ}
```

Con nuestros scores `z = [-0.053, 0.066, -0.041, ...]` (uno por cada palabra del vocabulario):

```python
# Vocabulario de 4 palabras — en la practica son 30.000+
e^{-0.053} ≈ 0.948
e^{ 0.066} ≈ 1.068
e^{-0.041} ≈ 0.960
e^{ 0.010} ≈ 1.010     # "perro", score inventado

S = 0.948 + 1.068 + 0.960 + 1.010 = 3.986

P(come)   = 0.948 / 3.986 ≈ 0.238
P(gato)   = 1.068 / 3.986 ≈ 0.268
P(camina) = 0.960 / 3.986 ≈ 0.241
P(perro)  = 1.010 / 3.986 ≈ 0.253
```

> [!WARNING]
> **Problema:** con un vocabulario real de 50.000 palabras hay que calcular
> `e^{zᵢ}` para **todas** ellas en cada par de entrenamiento.
> Con millones de pares por época, esto es computacionalmente prohibitivo.

---

#### Opción B — Sigmoid + Negative Sampling *(la que usamos)*

En vez de comparar contra todo el vocabulario, se compara el par real contra
**K pares falsos** elegidos al azar. Cada comparación es binaria: real (`1`) o falso (`0`).

```
σ(x) = 1 / (1 + e^{-x})      # un solo numero → probabilidad entre 0 y 1
```

```python
# Solo 3 calculos en vez de 50.000
σ(score_pos)  = σ(-0.053) ≈ 0.487   # queremos → 1.0
σ(score_neg1) = σ( 0.066) ≈ 0.516   # queremos → 0.0
σ(score_neg2) = σ(-0.041) ≈ 0.490   # queremos → 0.0
```

> [!NOTE]
> **¿Se pierde calidad?** No significativamente. Mikolov et al. (2013) demostraron
> que con `K = 5–20` muestras negativas los embeddings son igual de buenos
> que con softmax completo, pero el entrenamiento es **~100x más rápido**.

#### Comparación

| | Softmax | Sigmoid + NS |
|---|---|---|
| Cálculos por par | `V` (todo el vocab) | `K + 1` (típico: 6–21) |
| Probabilidades | Suman 1 (distribución) | Independientes (binarias) |
| Escala | Lento con vocab grande | Rápido siempre |
| Calidad embeddings | Alta | Alta (con K ≥ 5) |
| **¿Cuál usamos?** | Explicación conceptual | **En el código** |

---

### `σ(score_pos)` — par positivo (`come`)

```python
σ(-0.053) = 1 / (1 + e^{ 0.053})
          = 1 / (1 + 1.054)
          = 1 / 2.054
          = 0.487
```

### `σ(score_neg1)` — negativa 1 (`gato`)

```python
σ( 0.066) = 1 / (1 + e^{-0.066})
          = 1 / (1 + 0.936)
          = 1 / 1.936
          = 0.516
```

### `σ(score_neg2)` — negativa 2 (`camina`)

```python
σ(-0.041) = 1 / (1 + e^{ 0.041})
          = 1 / (1 + 1.042)
          = 1 / 2.042
          = 0.490
```

### Interpretación

| Par       | σ       | El modelo cree... | Queremos |
|-----------|---------|-------------------|----------|
| `come`    | `0.487` | 48.7% real        | → `1.0` ✅ (ES real) |
| `gato`    | `0.516` | 51.6% real        | → `0.0` ❌ (es falso) |
| `camina`  | `0.490` | 49.0% real        | → `0.0` ❌ (es falso) |

---

## 4. 📉 Pérdida (Loss)

```
Loss = -log(σ_pos) - log(1 - σ_neg1) - log(1 - σ_neg2)
```

| Término          | Cálculo                              | Valor   |
|------------------|--------------------------------------|---------|
| Par positivo     | `-log(0.487)`                        | `0.719` |
| Negativa 1       | `-log(1 - 0.516)` = `-log(0.484)`    | `0.726` |
| Negativa 2       | `-log(1 - 0.490)` = `-log(0.510)`    | `0.673` |
| **Loss total**   | `0.719 + 0.726 + 0.673`              | **`2.118`** |

> [!IMPORTANT]
> A menor Loss, mejor está aprendiendo el modelo. Con pesos perfectos, Loss → 0.

---

## 5. 🔙 Backward Pass — Gradientes

Los gradientes indican **en qué dirección y cuánto** ajustar cada vector.
Se calculan con la regla de la cadena sobre el Loss.

### `grad_v_t` — ajuste para `W_embed[3]` (`"perro"`)

```
∂L/∂v_t = (σ_pos - 1) × W_context[0]
         + σ_neg1      × W_context[1]
         + σ_neg2      × W_context[2]
```

```python
# Término par positivo:
(0.487 - 1) × [ 0.05,  0.22] = -0.513 × [ 0.05,  0.22] = [-0.026, -0.113]

# Término negativa 1:
0.516 × [ 0.30, -0.10] = [ 0.155, -0.052]

# Término negativa 2:
0.490 × [-0.15,  0.08] = [-0.074,  0.039]

# Suma componente a componente:
[-0.026 + 0.155 + (-0.074),  -0.113 + (-0.052) + 0.039]
= [ 0.056, -0.125]   # ← grad_v_t
```

### `grad_context_pos` — ajuste para `W_context[0]` (`"come"`)

```python
∂L/∂W_context[0] = (σ_pos - 1) × v_t
                 = (0.487 - 1)  × [0.13, -0.27]
                 = -0.513       × [0.13, -0.27]
                 = [-0.067,  0.139]   # ← grad_context_pos
```

### `grad_context_neg1` — ajuste para `W_context[1]` (`"gato"`)

```python
∂L/∂W_context[1] = σ_neg1 × v_t
                 = 0.516   × [0.13, -0.27]
                 = [ 0.067, -0.139]   # ← grad_context_neg1
```

### `grad_context_neg2` — ajuste para `W_context[2]` (`"camina"`)

```python
∂L/∂W_context[2] = σ_neg2 × v_t
                 = 0.490   × [0.13, -0.27]
                 = [ 0.064, -0.132]   # ← grad_context_neg2
```

---

## 6. 🔄 Actualización de Pesos (SGD)

```
nueva_fila = fila_actual  -  lr × gradiente        lr = 0.1
```

### `W_embed[3]` — `"perro"`

| | Valor |
|---|---|
| **Antes**     | `[ 0.130, -0.270]` |
| Gradiente     | `[ 0.056, -0.125]` |
| `- 0.1 ×`     | `[-0.006,  0.013]` |
| **Después**   | `[ 0.124, -0.258]` |

### `W_context[0]` — `"come"` (par positivo)

| | Valor |
|---|---|
| **Antes**     | `[ 0.050,  0.220]` |
| Gradiente     | `[-0.067,  0.139]` |
| `- 0.1 ×`     | `[ 0.007, -0.014]` |
| **Después**   | `[ 0.057,  0.206]` |

### `W_context[1]` — `"gato"` (negativa 1)

| | Valor |
|---|---|
| **Antes**     | `[ 0.300, -0.100]` |
| Gradiente     | `[ 0.067, -0.139]` |
| `- 0.1 ×`     | `[-0.007,  0.014]` |
| **Después**   | `[ 0.293, -0.086]` |

### `W_context[2]` — `"camina"` (negativa 2)

| | Valor |
|---|---|
| **Antes**     | `[-0.150,  0.080]` |
| Gradiente     | `[ 0.064, -0.132]` |
| `- 0.1 ×`     | `[-0.006,  0.013]` |
| **Después**   | `[-0.156,  0.093]` |

---

## 7. ✅ Verificación — ¿Mejoró el modelo?

### Score (`perro` → `come`) antes vs después

```python
# ANTES
dot([0.130, -0.270], [0.050,  0.220])
= (0.130 × 0.050) + (-0.270 × 0.220)
= 0.0065 + (-0.0594)
= -0.053

# DESPUÉS
dot([0.124, -0.258], [0.057,  0.206])
= (0.124 × 0.057) + (-0.258 × 0.206)
= 0.0071 + (-0.0531)
= -0.046
```

| | Score |
|---|---|
| Antes del paso  | `-0.053` |
| Después del paso | `-0.046` |
| Cambio          | `+0.007` ↑ |

> [!IMPORTANT]
> El score subió de `-0.053` a `-0.046`.
> Aún es negativo porque el modelo empieza muy lejos,
> pero **avanza en la dirección correcta** con cada par.
> Después de miles de pares, el score `(perro, come)` será alto
> y el de `(perro, gato_aleatorio)` será bajo. **Eso es Word2Vec.**

---

## 🗺️ Resumen del Ciclo (un paso)

| # | Operación | Qué produce |
|---|-----------|-------------|
| 1 | **Lookup** `v_t = W_embed[target]` | Vector del target |
| 2 | **Scores** `dot(v_t, W_context[c])` para c positivo y negativos | Un número por par |
| 3 | **Sigmoid** `σ(score)` para cada par | Probabilidad 0–1 |
| 4 | **Loss** `-log(σ_pos) - Σ log(1 - σ_neg)` | Escalar de error |
| 5 | **Gradientes** `∂L/∂v_t` y `∂L/∂v_c` por cada c | Dirección de ajuste |
| 6 | **SGD** `W -= lr × gradiente` | Pesos actualizados |
| 7 | **Repetir** con el siguiente par del corpus | Loss baja por época |
