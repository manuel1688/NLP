import json
from collections import Counter
import numpy as np

# Cargar dataset
with open("corpus_sentimiento_reviews.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer etiquetas de categoría
labels = [item["categoria"] for item in data["reviews"]]

# Conteo por clase
conteo = Counter(labels)

print("Distribución de clases:")
for clase, count in conteo.items():
    print(f"{clase}: {count}")

valores = np.array(list(conteo.values()))

mean = valores.mean()
std = valores.std()

print(f"\nMedia: {mean}")
print(f"Desviación estándar: {std}")

# # Coeficiente de variación: relación entre la desviación estándar y la media
# coef_var = std / mean

# # Regla simple de balanceo usando coef_var
# if std == 0:
#     print("✔ Dataset perfectamente balanceado")
# elif coef_var < 0.1:
#     print("✔ Bien balanceado (variación baja)")
# elif coef_var < 0.3:
#     print("⚠ Aceptable pero con algo de desbalance")
# else:
#     print("❌ Dataset desbalanceado")
