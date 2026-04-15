# ================================================================
# Lab 01 — El Perceptrón Original (1958)
# Escenario: ¿Voy al concierto?
#
# ARCHIVO DE SOLUCIÓN — solo para el instructor
# ================================================================

# ---------------------------------------------------------------
# Paso 1: Entradas (x)
# Cada variable es binaria: 1 = Sí, 0 = No
# ---------------------------------------------------------------
x1 = 1   # ¿Tengo dinero?          → Sí
x2 = 0   # ¿Van mis amigos?        → No
x3 = 1   # ¿Va a llover esta noche? → Sí

# ---------------------------------------------------------------
# Paso 2: Pesos (w) y sesgo (bias)
# Los pesos representan la importancia de cada factor.
# El bias es una constante que eleva o baja el umbral de decisión.
# ---------------------------------------------------------------
w1   =  6    # El dinero es muy importante
w2   =  2    # Los amigos importan, pero no deciden solos
w3   = -4    # La lluvia es un factor negativo
bias = -3    # Sesgo negativo: eres un poco perezoso para salir

# ---------------------------------------------------------------
# Paso 3: Suma ponderada (z)
# z = (x1 * w1) + (x2 * w2) + (x3 * w3) + bias
# ---------------------------------------------------------------
z = (x1 * w1) + (x2 * w2) + (x3 * w3) + bias
# z = (1*6) + (0*2) + (1*-4) + (-3)
# z = 6 + 0 - 4 - 3 = -1

# ---------------------------------------------------------------
# Paso 4: Función de activación (función de paso)
# Si z > 0  →  salida = 1  (Vas al concierto)
# Si z <= 0 →  salida = 0  (Te quedas en casa)
# ---------------------------------------------------------------
if z > 0:
    salida = 1
else:
    salida = 0

# ---------------------------------------------------------------
# Paso 5: Resultado
# ---------------------------------------------------------------
print(f"Suma ponderada z = {z}")
print(f"Salida del perceptrón = {salida}")

if salida == 1:
    print("Decisión: Vas al concierto")
else:
    print("Decisión: Te quedas en casa")
