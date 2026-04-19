# 📖 Word2Vec — Skip-gram: Ejemplo Numérico Completo

> Un solo paso de entrenamiento, de principio a fin.
> Los mismos números se usan en **todos** los pasos. Nada aparece de la nada.

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

> [!NOTE]
> En la práctica `W_embed ~ Normal(0, 0.01)` y `W_context = zeros`.
> Aquí usamos valores no-cero para que los gradientes sean ilustrativos.

### `W_embed` — Matriz de palabras objetivo

| Fila | Palabra    | Vector            | Uso              |
|------|------------|-------------------|------------------|
| `0`  | `"come"`   | `[ 0.05,  0.22]`  |                  |
| `1`  | `"gato"`   | `[ 0.42,  0.11]`  |                  |
| `2`  | `"camina"` | `[-0.09,  0.33]`  |                  |
| `3`  | `"perro"`  | `[ 0.13, -0.27]`  | ← **usaremos esta** |

### `W_context` — Matriz de palabras contexto

| Fila | Palabra    | Vector            | Uso              |
|------|------------|-------------------|------------------|
| `0`  | `"come"`   | `[ 0.05,  0.22]`  | ← par positivo   |
| `1`  | `"gato"`   | `[ 0.30, -0.10]`  | ← negativa 1     |
| `2`  | `"camina"` | `[-0.15,  0.08]`  | ← negativa 2     |
| `3`  | `"perro"`  | `[ 0.01, -0.05]`  |                  |

---

## 2. ⚡ Forward Pass — Calcular Scores

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

> [!NOTE]
> **¿Por qué sigmoid y no softmax?**
> Negative Sampling convierte el problema en clasificaciones binarias independientes:
> par real = `1`, par falso = `0`. Sigmoid devuelve la probabilidad de que el par sea real.

```
σ(x) = 1 / (1 + e^{-x})
```

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
