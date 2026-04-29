

from sklearn.datasets import fetch_20newsgroups
import numpy as np

print("Categorías disponibles en 20 Newsgroups:")
all_data = fetch_20newsgroups(subset='all')
print(all_data.target_names)
print()

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
print(f"Cantidad de textos en train: {len(train.data)}")
print(f"Cantidad de textos en test: {len(test.data)}\n")

print("Ejemplo de texto completo (train[0]):\n")
print(train.data[0])
print(f"\nCategoría: {train.target_names[train.target[0]]}\n")

print("Otro ejemplo (train[10]):\n")
print(train.data[10])
print(f"\nCategoría: {train.target_names[train.target[10]]}\n")

longitudes = [len(text) for text in train.data]
print(f"Longitud promedio de los textos de train: {np.mean(longitudes):.1f} caracteres")
print(f"Longitud mínima: {np.min(longitudes)}")
print(f"Longitud máxima: {np.max(longitudes)}\n")

train_clean = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
print("Ejemplo de texto limpio (sin headers, footers, quotes):\n")
print(train_clean.data[0])

# Contar y mostrar la cantidad de textos por categoría en el set de entrenamiento
# - train.target es una lista de índices numéricos que indican la categoría de cada texto
# - Counter(train.target) cuenta cuántos textos hay de cada categoría (por índice)
# - Se imprime el nombre de la categoría y la cantidad de textos para cada una
from collections import Counter
conteo = Counter(train.target)
print("\nCantidad de textos por categoría en train:")
for idx, count in conteo.items():
    print(f"{train.target_names[idx]}: {count}")
