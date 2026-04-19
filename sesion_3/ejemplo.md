================================================================
 Word2Vec — Skip-gram: Ejemplo Numerico Completo
 Un solo paso de entrenamiento, de principio a fin
================================================================

Los mismos numeros se usan en TODOS los pasos.
Nada aparece de la nada.


================================================================
 0. PUNTO DE PARTIDA
================================================================

Vocabulario (4 palabras):
  come   -> indice 0
  gato   -> indice 1
  camina -> indice 2
  perro  -> indice 3

Par de entrenamiento de este paso:
  target  : "perro"  (indice 3)  <- palabra que damos al modelo
  contexto: "come"   (indice 0)  <- palabra que deberia predecir

Muestras negativas (palabras al azar, label = 0):
  negativa 1: "gato"   (indice 1)
  negativa 2: "camina" (indice 2)

Hiperparametros:
  D  = 2    (dimension del embedding)
  lr = 0.1  (learning rate)


================================================================
 1. INICIALIZACION DE MATRICES
================================================================

Dos matrices de dimension (4 x 2), una fila por palabra:

W_embed (palabra objetivo):
  fila 0  "come"   ->  [ 0.05,  0.22]
  fila 1  "gato"   ->  [ 0.42,  0.11]
  fila 2  "camina" ->  [-0.09,  0.33]
  fila 3  "perro"  ->  [ 0.13, -0.27]  <- usaremos esta

W_context (palabra contexto):
  fila 0  "come"   ->  [ 0.05,  0.22]  <- par positivo
  fila 1  "gato"   ->  [ 0.30, -0.10]  <- negativa 1
  fila 2  "camina" ->  [-0.15,  0.08]  <- negativa 2
  fila 3  "perro"  ->  [ 0.01, -0.05]

Nota: en la practica W_embed ~ Normal(0, 0.01) y W_context = zeros.
      Aqui usamos valores no-cero para que los gradientes
      del ejemplo sean mas ilustrativos.


================================================================
 2. FORWARD PASS — Calcular scores
================================================================

Paso 2a — Lookup del vector target

  v_t = W_embed[3]           <- extraer fila 3 directamente
      = [0.13, -0.27]

  (Matematicamente: v_t = W_embed * one_hot([0,0,0,1])
   En codigo siempre se usa el lookup: es equivalente y mas rapido.)

---

Paso 2b — Score del par POSITIVO  (perro -> come)

  score_pos = dot(v_t, W_context[0])
            = dot([0.13, -0.27], [0.05, 0.22])
            = (0.13 x 0.05) + (-0.27 x 0.22)
            =  0.0065        +  (-0.0594)
            = -0.053

---

Paso 2c — Score negativa 1  (perro -> gato)

  score_neg1 = dot(v_t, W_context[1])
             = dot([0.13, -0.27], [0.30, -0.10])
             = (0.13 x 0.30) + (-0.27 x -0.10)
             =  0.039         +   0.027
             =  0.066

---

Paso 2d — Score negativa 2  (perro -> camina)

  score_neg2 = dot(v_t, W_context[2])
             = dot([0.13, -0.27], [-0.15, 0.08])
             = (0.13 x -0.15) + (-0.27 x 0.08)
             = -0.0195         +  (-0.0216)
             = -0.041

Resumen de scores calculados:
  score_pos  (come)   = -0.053
  score_neg1 (gato)   =  0.066
  score_neg2 (camina) = -0.041


================================================================
 3. SIGMOID — Convertir scores en probabilidades
================================================================

Por que sigmoid y no softmax?
  Negative Sampling convierte el problema en clasificaciones
  binarias independientes: par real = 1, par falso = 0.
  Sigmoid devuelve la probabilidad de que el par sea real.

  sigma(x) = 1 / (1 + e^{-x})

---

sigma_pos  = sigma(-0.053)
           = 1 / (1 + e^{ 0.053})
           = 1 / (1 + 1.054)
           = 1 / 2.054
           = 0.487

  -> modelo cree 48.7% de probabilidad de que (perro,come) sea real
  -> queremos llegar a 1.0  (ES un par real del corpus)

---

sigma_neg1 = sigma(0.066)
           = 1 / (1 + e^{-0.066})
           = 1 / (1 + 0.936)
           = 1 / 1.936
           = 0.516

  -> modelo cree 51.6% de probabilidad de que (perro,gato) sea real
  -> queremos llegar a 0.0  (es par FALSO)

---

sigma_neg2 = sigma(-0.041)
           = 1 / (1 + e^{ 0.041})
           = 1 / (1 + 1.042)
           = 1 / 2.042
           = 0.490

  -> modelo cree 49.0% de probabilidad de que (perro,camina) sea real
  -> queremos llegar a 0.0  (es par FALSO)


================================================================
 4. PERDIDA (Loss)
================================================================

  Loss = -log(sigma_pos)
       - log(1 - sigma_neg1)
       - log(1 - sigma_neg2)

Termino 1 — par positivo:
  -log(sigma_pos)
  = -log(0.487)
  = 0.719

Termino 2 — negativa 1:
  1 - sigma_neg1 = 1 - 0.516 = 0.484
  -log(0.484) = 0.726

Termino 3 — negativa 2:
  1 - sigma_neg2 = 1 - 0.490 = 0.510
  -log(0.510) = 0.673

  Loss = 0.719 + 0.726 + 0.673 = 2.118

  A menor Loss, mejor esta aprendiendo el modelo.
  Con pesos perfectos Loss tenderia a 0.


================================================================
 5. BACKWARD PASS — Gradientes
================================================================

Los gradientes indican en que direccion y cuanto ajustar cada vector.
Se calculan con la regla de la cadena sobre el Loss.

---

grad_v_t  (ajuste para W_embed[3] "perro"):

  = (sigma_pos - 1) * W_context[0]
  +  sigma_neg1     * W_context[1]
  +  sigma_neg2     * W_context[2]

  Termino par positivo:
    (0.487 - 1) * [ 0.05,  0.22]
    = -0.513    * [ 0.05,  0.22]
    = [-0.026, -0.113]

  Termino negativa 1:
    0.516 * [ 0.30, -0.10]
    = [ 0.155, -0.052]

  Termino negativa 2:
    0.490 * [-0.15,  0.08]
    = [-0.074,  0.039]

  Suma componente a componente:
    [-0.026 + 0.155 + (-0.074),   -0.113 + (-0.052) + 0.039]
    = [ 0.056, -0.125]           <- grad_v_t

---

grad_context_pos  (ajuste para W_context[0] "come"):

  = (sigma_pos - 1) * v_t
  = (0.487 - 1)     * [0.13, -0.27]
  = -0.513          * [0.13, -0.27]
  = [-0.067,  0.139]              <- grad_context_pos

---

grad_context_neg1  (ajuste para W_context[1] "gato"):

  = sigma_neg1 * v_t
  = 0.516      * [0.13, -0.27]
  = [ 0.067, -0.139]              <- grad_context_neg1

---

grad_context_neg2  (ajuste para W_context[2] "camina"):

  = sigma_neg2 * v_t
  = 0.490      * [0.13, -0.27]
  = [ 0.064, -0.132]              <- grad_context_neg2


================================================================
 6. ACTUALIZACION DE PESOS  (SGD)
================================================================

  nueva_fila = fila_actual  -  lr * gradiente
  lr = 0.1

---

W_embed[3] "perro":
  antes    ->  [ 0.130, -0.270]
  gradiente    [ 0.056, -0.125]
  - 0.1 *    = [-0.006,  0.013]
  despues  ->  [ 0.124, -0.258]

---

W_context[0] "come":
  antes    ->  [ 0.050,  0.220]
  gradiente    [-0.067,  0.139]
  - 0.1 *    = [ 0.007, -0.014]
  despues  ->  [ 0.057,  0.206]

---

W_context[1] "gato":
  antes    ->  [ 0.300, -0.100]
  gradiente    [ 0.067, -0.139]
  - 0.1 *    = [-0.007,  0.014]
  despues  ->  [ 0.293, -0.086]

---

W_context[2] "camina":
  antes    ->  [-0.150,  0.080]
  gradiente    [ 0.064, -0.132]
  - 0.1 *    = [-0.006,  0.013]
  despues  ->  [-0.156,  0.093]


================================================================
 7. VERIFICACION — MEJORO EL MODELO?
================================================================

Score (perro -> come) ANTES del paso:
  dot([0.130, -0.270], [0.050,  0.220])
  = (0.130 x 0.050) + (-0.270 x 0.220)
  = 0.0065 + (-0.0594)
  = -0.053

Score (perro -> come) DESPUES del paso:
  dot([0.124, -0.258], [0.057,  0.206])
  = (0.124 x 0.057) + (-0.258 x 0.206)
  = 0.0071 + (-0.0531)
  = -0.046

  El score subio de -0.053 a -0.046.
  Aun es negativo porque el modelo empieza muy lejos,
  pero avanza en la direccion correcta con cada par.
  Despues de miles de pares el score (perro, come) sera alto
  y el de (perro, gato_aleatorio) sera bajo.
  Eso es lo que hace Word2Vec.


================================================================
 RESUMEN DEL CICLO (un paso)
================================================================

  1. Lookup       v_t = W_embed[target]
  2. Scores       dot(v_t, W_context[c])  para c positivo y negativos
  3. Sigmoid      sigma(score) para cada par
  4. Loss         -log(sigma_pos) - suma log(1 - sigma_neg)
  5. Gradientes   dL/dv_t  y  dL/dv_c  por cada c
  6. SGD          W -= lr * gradiente
  7. Repetir      con el siguiente par del corpus
