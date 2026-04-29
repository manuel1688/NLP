
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter


textos = [
    "excelente pelicula me encanto",
    "muy buena pelicula recomendada",
    "pelicula aburrida y lenta",
    "gran actuación y guion en la película",
    "no me gustó la película, muy predecible",
    "película divertida y emocionante",
    "historia original y bien contada",
    "efectos especiales impresionantes",
    "el final de la película fue inesperado",
    "no recomendaría esta película",
    "restaurante con comida deliciosa",
    "servicio rapido y buena comida",
    "mala atencion en el restaurante",
    "el mejor restaurante de la ciudad",
    "comida fría y cara en el restaurante",
    "excelente atención y ambiente agradable",
    "menú variado y platos exquisitos",
    "demasiada espera para la comida",
    "porciones pequeñas y precios altos",
    "volvería a este restaurante",
    "producto de mala calidad",
    "excelente producto lo recomiendo",
    "no funciona el producto",
    "producto llegó antes de lo esperado",
    "producto defectuoso y sin garantía",
    "muy satisfecho con el producto",
    "el producto superó mis expectativas",
    "no era lo que esperaba",
    "producto fácil de usar",
    "no volvería a comprar este producto",
    "hotel muy limpio y comodo",
    "excelente ubicacion del hotel",
    "hotel sucio y mal servicio",
    "personal amable en el hotel",
    "ruidoso y caro el hotel",
    "habitaciones amplias y confortables",
    "desayuno incluido y variado",
    "no pude dormir por el ruido",
    "servicio al cliente deficiente",
    "volvería a hospedarme en este hotel"
]

etiquetas = [
    "peliculas", "peliculas", "peliculas", "peliculas", "peliculas",
    "peliculas", "peliculas", "peliculas", "peliculas", "peliculas",
    "restaurantes", "restaurantes", "restaurantes", "restaurantes", "restaurantes",
    "restaurantes", "restaurantes", "restaurantes", "restaurantes", "restaurantes",
    "productos", "productos", "productos", "productos", "productos",
    "productos", "productos", "productos", "productos", "productos",
    "hoteles", "hoteles", "hoteles", "hoteles", "hoteles",
    "hoteles", "hoteles", "hoteles", "hoteles", "hoteles"
]


print("Distribución original:", Counter(etiquetas))


print("\nSplit con STRATIFY:")

X_train, X_test, y_train, y_test = train_test_split(
    textos,
    etiquetas,
    test_size=0.3,
    stratify=etiquetas,
    random_state=42
)

print("Distribución train:", Counter(y_train))
print("Distribución test:", Counter(y_test))

