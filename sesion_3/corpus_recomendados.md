# 📚 Corpus Recomendados para Word2Vec

## 🇪🇸 Corpus en Español

### Para Principiantes (Pequeños — pruebas rápidas)

| Corpus | Tamaño | Cómo obtenerlo | Ideal para |
|--------|--------|-----------------|------------|
| **Corpus propio del curso** | ~1 KB | Definido inline en el script | Verificar que el pipeline funciona paso a paso |
| **NLTK cess_esp** | ~500 KB | `import nltk; nltk.download('cess_esp')` | Primeros experimentos con texto real en español |
| **Wikicorpus (muestra)** | ~5 MB | Subset de Wikipedia ES con [wikiextractor](https://github.com/attardi/wikiextractor) | Vocabulario variado sin esperar horas |

### Intermedios (Para embeddings con resultados visibles)

| Corpus | Tamaño | Cómo obtenerlo | Ideal para |
|--------|--------|-----------------|------------|
| **SBWCE** (Spanish Billion Words) | ~1.5 GB | [crscardellino.ar/SBWCE](https://crscardellino.ar/SBWCE/download.html) | Embeddings de calidad; analogías funcionales |
| **Wikipedia ES dump** | ~2 GB (texto limpio) | `wget` del dump + wikiextractor | Corpus general grande |
| **OpenSubtitles ES** | ~500 MB | [OPUS](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Lenguaje coloquial e informal |

### Avanzados (Embeddings pre-entrenados listos para usar)

| Recurso | Dimensión | Cómo cargarlo |
|---------|-----------|----------------|
| **FastText ES** (Facebook) | 300d | `fasttext.load_model('cc.es.300.bin')` — [descarga](https://fasttext.cc/docs/en/crawl-vectors.html) |
| **Word2Vec SBWCE** | 300d | Archivo `.bin` desde [crscardellino.ar](https://crscardellino.ar/SBWCE/download.html) → `gensim.models.KeyedVectors.load_word2vec_format('sbwce.bin', binary=True)` |

---

## 🇬🇧 Corpus en Inglés

### Para Principiantes

| Corpus | Tamaño | Cómo obtenerlo | Ideal para |
|--------|--------|-----------------|------------|
| **Text8** | ~100 MB | `import gensim.downloader as api; corpus = api.load('text8')` | Estándar de facto para tutoriales de Word2Vec |
| **NLTK Brown corpus** | ~1.5 MB | `import nltk; nltk.download('brown')` | Texto anotado, pruebas rápidas |

### Intermedios

| Corpus | Tamaño | Cómo obtenerlo | Ideal para |
|--------|--------|-----------------|------------|
| **WikiText-103** | ~500 MB | HuggingFace `datasets` o descarga directa | Benchmark estándar para modelos de lenguaje |
| **1 Billion Word Benchmark** | ~4 GB | `tensorflow_datasets` o descarga directa | Escala real, resultados publicables |

### Avanzados (Pre-entrenados)

| Recurso | Dimensión | Cómo cargarlo |
|---------|-----------|----------------|
| **Google News Word2Vec** | 300d, 3M palabras | `gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)` |
| **GloVe** (Stanford) | 50d / 100d / 200d / 300d | [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) → convertir con `glove2word2vec` |
| **FastText EN** | 300d | `fasttext.load_model('cc.en.300.bin')` — [descarga](https://fasttext.cc/docs/en/crawl-vectors.html) |

---

## Snippets de carga rápida

```python
# Text8 (inglés, principiantes)
import gensim.downloader as api
corpus = api.load('text8')  # lista de listas de tokens

# NLTK cess_esp (español, principiantes)
import nltk
nltk.download('cess_esp')
from nltk.corpus import cess_esp
sentences = cess_esp.sents()  # lista de listas de tokens

# Google News Word2Vec (inglés, pre-entrenado)
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# FastText ES (español, pre-entrenado)
import fasttext
ft = fasttext.load_model('cc.es.300.bin')
vector = ft.get_word_vector('computadora')  # funciona con OOV

# Word2Vec SBWCE (español, pre-entrenado)
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin.gz', binary=True)
```
