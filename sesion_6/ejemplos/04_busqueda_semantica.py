# ===============================
# Demo Interactiva: Búsqueda Semántica con Transformers
# ===============================
# Objetivo: Mostrar cómo un transformer "entiende" el significado de textos
# sin necesidad de entrenamiento adicional.

# --- DEPENDENCIAS ---
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Cargar dataset y preparar corpus ---
print("Cargando corpus de noticias...")
corpus = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
print(f"{len(corpus.data)} documentos cargados de {len(corpus.target_names)} categorías.\n")

# --- 2. Generar embeddings del corpus ---
print("Generando embeddings semánticos (esto puede tardar un poco)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================================================
#    LÍNEA CLAVE #1: Transformar TODO el corpus a vectores semánticos
# ============================================================================
corpus_embeddings = encoder.encode(corpus.data, show_progress_bar=True, convert_to_numpy=True)
#    Convierte 11,000+ textos en matriz de vectores (11000 x 384)
#    Cada fila = representación numérica del significado de un documento
#    Esto se hace UNA SOLA VEZ al inicio (es costoso computacionalmente)

print(f"Embeddings listos: {corpus_embeddings.shape}\n")

# --- 3. Función de búsqueda semántica ---
def buscar_similares(consulta, top_k=5):
    """Encuentra los textos más similares a la consulta"""
    
    # ============================================================================
    # LÍNEA CLAVE #2: Convertir la consulta del usuario a vector
    # ============================================================================
    query_embedding = encoder.encode([consulta], convert_to_numpy=True)
    #    Transforma "space exploration" → [0.12, -0.45, 0.78, ... 384 números]
    #    Usa el MISMO transformer para que esté en el mismo "espacio semántico"
    #    Debe ser lista: [consulta] porque encode() espera múltiples textos
    
    # ============================================================================
    #    LÍNEA CLAVE #3: Calcular similitud entre consulta y TODOS los documentos
    # ============================================================================
    similitudes = cosine_similarity(query_embedding, corpus_embeddings)[0]
    #    Compara 1 consulta vs 11,000 docs → devuelve array de 11,000 scores
    #    cosine_similarity calcula: cos(θ) = (A·B) / (||A|| × ||B||)
    #    Resultado: [0.588, 0.123, 0.508, ...] scores entre -1 y 1
    #    [0] extrae la primera (y única) fila porque query_embedding es 1 vector
    
    # ============================================================================
    #    LÍNEA CLAVE #4: Ordenar documentos de mayor a menor similitud
    # ============================================================================
    indices_ordenados = np.argsort(similitudes)[::-1][:top_k]
    #    argsort() devuelve ÍNDICES ordenados de menor a mayor
    #    [::-1] invierte el orden (mayor a menor similitud)
    #    [:top_k] toma solo los primeros 5 resultados
    #    Ejemplo: [4523, 891, 7234, 102, 6789] ← índices de los docs más similares
    
    # ============================================================================
    #    LÍNEA CLAVE #5: Extraer los scores correspondientes a los mejores resultados
    # ============================================================================
    return indices_ordenados, similitudes[indices_ordenados]
    #    Usa los índices para obtener los valores de similitud originales
    #    Ejemplo: indices=[4523, 891] → scores=[0.588, 0.508]
    #    Esto permite mostrar el porcentaje de similitud al usuario

# --- 4. Interfaz interactiva ---
print("=" * 70)
print("BUSCADOR SEMÁNTICO DE NOTICIAS")
print("=" * 70)
print("\nEscribe una consulta y encontraré noticias relacionadas por significado.")
print("Ejemplos: 'space exploration', 'computer graphics', 'religion debate'\n")
print("Escribe 'salir' para terminar.\n")

while True:
    consulta = input("Tu consulta: ").strip()
    
    if consulta.lower() in ['salir', 'exit', 'quit', '']:
        print("\nHasta luego.")
        break
    
    # Buscar documentos similares
    indices, scores = buscar_similares(consulta, top_k=5)
    
    print(f"\nTop 5 resultados para: '{consulta}'")
    print("-" * 70)
    
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        categoria = corpus.target_names[corpus.target[idx]]
        texto = corpus.data[idx][:200]  # Primeros 200 caracteres
        
        print(f"\n{i}. Categoría: {categoria} | Similitud: {score:.3f}")
        print(f"   {texto}...")
    
    print("\n" + "=" * 70 + "\n")

# --- BONUS: Demo de clasificación zero-shot ---
print("\nBONUS: Clasificación Zero-Shot")
print("=" * 70)
print("¿Cómo clasificaría un transformer sin entrenamiento?\n")

# Definir descripciones de algunas categorías
descripciones_categorias = {
    'sci.space': 'articles about space exploration, astronomy, and celestial objects',
    'comp.graphics': 'discussions about computer graphics, visualization, and rendering',
    'talk.religion.misc': 'religious debates and philosophical discussions',
    'rec.sport.baseball': 'baseball games, players, and statistics',
    'sci.med': 'medical research, health topics, and diseases'
}

# Generar embeddings de las descripciones
cat_embeddings = encoder.encode(list(descripciones_categorias.values()), convert_to_numpy=True)

texto_ejemplo = input("Escribe un texto para clasificar (o Enter para ejemplo): ").strip()
if not texto_ejemplo:
    texto_ejemplo = "The telescope captured amazing images of distant galaxies and nebulae"
    print(f"Usando ejemplo: '{texto_ejemplo}'")

# Clasificar por similitud
texto_embedding = encoder.encode([texto_ejemplo], convert_to_numpy=True)
similitudes_cat = cosine_similarity(texto_embedding, cat_embeddings)[0]

print("\nSimilitud con cada categoría:")
for (cat, desc), sim in zip(descripciones_categorias.items(), similitudes_cat):
    barra = "|" * int(sim * 50)
    print(f"{cat:25} {sim:.3f} {barra}")

prediccion = list(descripciones_categorias.keys())[np.argmax(similitudes_cat)]
print(f"\nPredicción: {prediccion}")