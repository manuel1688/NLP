# Lab 02 — Bigram Model (completar)

Curso: Diplomado en Inteligencia Artificial  
Nivel: Introductorio  
Duración estimada: 40–60 minutos

---

## Objetivo

Completar las funciones de un modelo de lenguaje de bigramas para:

1. Contar pares de palabras consecutivas.
2. Calcular probabilidades condicionales simples.
3. Generar una frase corta usando esas probabilidades.

El corpus, la tokenización y la generación ya están dados. Tu trabajo es implementar los algoritmos de conteo y probabilidad.

No usarás librerías de NLP. Solo Python puro.

---

## Conceptos clave

Un **bigrama** es un par de palabras consecutivas. Un **trigrama** es un triplete.

| Tipo | Ejemplo | Notación |
|---|---|---|
| Bigrama | (agrega, sal) | (w₁, w₂) |
| Trigrama | (agrega, sal, al) | (w₁, w₂, w₃) |

La probabilidad de una palabra dado lo anterior:

$$P(w_{i+1} \mid w_i) = \frac{\text{conteo}(w_i,\ w_{i+1})}{\text{conteo}(w_i)}$$

---

## Corpus

El corpus ya está en el archivo como una variable Python. Aquí lo verás en su forma cruda y con fronteras de oración.

### Oraciones originales

```
agrega sal al agua
agrega sal y pimienta
agrega las cebollas al caldo
agrega las zanahorias al caldo
calienta el agua en la olla
calienta el aceite en la sartén
mezcla la harina con el agua
mezcla la harina con la leche
corta las cebollas en trozos
corta las zanahorias en trozos
hierve el agua por cinco minutos
hierve el caldo por diez minutos
sirve el plato caliente
```

### Con tokens de frontera (`<s>` y `</s>`)

```
<s>  agrega    sal         al          agua       </s>
<s>  agrega    sal         y           pimienta   </s>
<s>  agrega    las         cebollas    al         caldo     </s>
<s>  agrega    las         zanahorias  al         caldo     </s>
<s>  calienta  el          agua        en         la        olla      </s>
<s>  calienta  el          aceite      en         la        sartén    </s>
<s>  mezcla    la          harina      con        el        agua      </s>
<s>  mezcla    la          harina      con        la        leche     </s>
<s>  corta     las         cebollas    en         trozos    </s>
<s>  corta     las         zanahorias  en         trozos    </s>
<s>  hierve    el          agua        por        cinco     minutos   </s>
<s>  hierve    el          caldo       por        diez      minutos   </s>
<s>  sirve     el          plato       caliente   </s>
```

---

## Tabla de conteo: Bigramas

Cada fila es un par consecutivo (w₁, w₂) y cuántas veces aparece en el corpus.

| w₁            | w₂            | Conteo |
|---------------|---------------|--------|
| `<s>`         | agrega        | 4      |
| `<s>`         | calienta      | 2      |
| `<s>`         | corta         | 2      |
| `<s>`         | hierve        | 2      |
| `<s>`         | mezcla        | 2      |
| `<s>`         | sirve         | 1      |
| aceite        | en            | 1      |
| agrega        | las           | 2      |
| agrega        | sal           | 2      |
| agua          | `</s>`        | 2      |
| agua          | en            | 1      |
| agua          | por           | 1      |
| al            | agua          | 1      |
| al            | caldo         | 2      |
| caldo         | `</s>`        | 2      |
| caldo         | por           | 1      |
| calienta      | el            | 2      |
| caliente      | `</s>`        | 1      |
| cebollas      | al            | 1      |
| cebollas      | en            | 1      |
| cinco         | minutos       | 1      |
| con           | el            | 1      |
| con           | la            | 1      |
| corta         | las           | 2      |
| diez          | minutos       | 1      |
| el            | aceite        | 1      |
| el            | agua          | 3      |
| el            | caldo         | 1      |
| el            | plato         | 1      |
| en            | la            | 2      |
| en            | trozos        | 2      |
| harina        | con           | 2      |
| hierve        | el            | 2      |
| la            | harina        | 2      |
| la            | leche         | 1      |
| la            | olla          | 1      |
| la            | sartén        | 1      |
| las           | cebollas      | 2      |
| las           | zanahorias    | 2      |
| leche         | `</s>`        | 1      |
| mezcla        | la            | 2      |
| minutos       | `</s>`        | 2      |
| olla          | `</s>`        | 1      |
| pimienta      | `</s>`        | 1      |
| plato         | caliente      | 1      |
| por           | cinco         | 1      |
| por           | diez          | 1      |
| sal           | al            | 1      |
| sal           | y             | 1      |
| sartén        | `</s>`        | 1      |
| sirve         | el            | 1      |
| trozos        | `</s>`        | 2      |
| y             | pimienta      | 1      |
| zanahorias    | al            | 1      |
| zanahorias    | en            | 1      |

---

## Tabla de conteo: Trigramas

Cada fila es un triplete consecutivo (w₁, w₂, w₃).

| w₁            | w₂            | w₃            | Conteo |
|---------------|---------------|---------------|--------|
| `<s>`         | agrega        | las           | 2      |
| `<s>`         | agrega        | sal           | 2      |
| `<s>`         | calienta      | el            | 2      |
| `<s>`         | corta         | las           | 2      |
| `<s>`         | hierve        | el            | 2      |
| `<s>`         | mezcla        | la            | 2      |
| `<s>`         | sirve         | el            | 1      |
| aceite        | en            | la            | 1      |
| agrega        | las           | cebollas      | 1      |
| agrega        | las           | zanahorias    | 1      |
| agrega        | sal           | al            | 1      |
| agrega        | sal           | y             | 1      |
| agua          | en            | la            | 1      |
| agua          | por           | cinco         | 1      |
| al            | agua          | `</s>`        | 1      |
| al            | caldo         | `</s>`        | 2      |
| caldo         | por           | diez          | 1      |
| calienta      | el            | aceite        | 1      |
| calienta      | el            | agua          | 1      |
| cebollas      | al            | caldo         | 1      |
| cebollas      | en            | trozos        | 1      |
| cinco         | minutos       | `</s>`        | 1      |
| con           | el            | agua          | 1      |
| con           | la            | leche         | 1      |
| corta         | las           | cebollas      | 1      |
| corta         | las           | zanahorias    | 1      |
| diez          | minutos       | `</s>`        | 1      |
| el            | aceite        | en            | 1      |
| el            | agua          | `</s>`        | 1      |
| el            | agua          | en            | 1      |
| el            | agua          | por           | 1      |
| el            | caldo         | por           | 1      |
| el            | plato         | caliente      | 1      |
| en            | la            | olla          | 1      |
| en            | la            | sartén        | 1      |
| en            | trozos        | `</s>`        | 2      |
| harina        | con           | el            | 1      |
| harina        | con           | la            | 1      |
| hierve        | el            | agua          | 1      |
| hierve        | el            | caldo         | 1      |
| la            | harina        | con           | 2      |
| la            | leche         | `</s>`        | 1      |
| la            | olla          | `</s>`        | 1      |
| la            | sartén        | `</s>`        | 1      |
| las           | cebollas      | al            | 1      |
| las           | cebollas      | en            | 1      |
| las           | zanahorias    | al            | 1      |
| las           | zanahorias    | en            | 1      |
| mezcla        | la            | harina        | 2      |
| plato         | caliente      | `</s>`        | 1      |
| por           | cinco         | minutos       | 1      |
| por           | diez          | minutos       | 1      |
| sal           | al            | agua          | 1      |
| sal           | y             | pimienta      | 1      |
| sirve         | el            | plato         | 1      |
| y             | pimienta      | `</s>`        | 1      |
| zanahorias    | al            | caldo         | 1      |
| zanahorias    | en            | trozos        | 1      |

---

## Tu trabajo

Abre el archivo `lab_bigram_model_template.py`. Encontrarás el corpus y las funciones auxiliares ya implementadas. Solo debes completar las dos funciones marcadas con `# TODO`.

### Función 1: `construir_conteos`

Itera sobre cada oración del corpus ya tokenizada con fronteras. Para cada posición `i`:

- Suma 1 al conteo del unigrama `tokens[i]`.
- Suma 1 al conteo del bigrama `(tokens[i], tokens[i+1])`.

Usa las tablas de arriba para verificar tus resultados.

### Función 2: `calcular_probabilidades`

Para cada bigrama `(actual, siguiente)` con su conteo, aplica:

$$P(\text{siguiente} \mid \text{actual}) = \frac{\text{conteo\_bigramas}[(\text{actual},\ \text{siguiente})]}{\text{conteo\_unigramas}[\text{actual}]}$$

Guarda cada resultado en el diccionario `probabilidades`.

---

## Salida esperada

```
=== Probabilidades solicitadas ===
P(las | agrega)  = 0.5
P(el | hierve)   = 1.0
P(con | harina)  = 1.0

=== Frase generada (greedy) ===
agrega sal al caldo
```

---

## Preguntas de reflexión

1. ¿Por qué P(el | hierve) = 1.0? ¿Qué implica eso sobre las oraciones del corpus?
2. ¿Qué pasa si consultas P(agua | sirve)? ¿Por qué?
3. Agrega la oración `"agrega las especias al caldo"` al corpus. ¿Cómo cambia P(las | agrega)?
4. ¿Por qué un modelo de trigramas necesita más datos que uno de bigramas para ser confiable?
