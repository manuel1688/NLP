# Visualizaciones de Evaluación

---

## Matriz de Confusión

Muestra exactamente dónde se equivoca el modelo.

|  | Predijo: positivo | Predijo: negativo |
|---|---|---|
| **Real: positivo** | TP (acierto) | FN (perdió un positivo) |
| **Real: negativo** | FP (falsa alarma) | TN (acierto) |

- **TP** — predijo positivo y era positivo
- **TN** — predijo negativo y era negativo
- **FP** — predijo positivo pero era negativo
- **FN** — predijo negativo pero era positivo

> No engaña: muestra la verdad completa sin resumirla en un número.

---

## 📈 Curva ROC

Muestra el trade-off entre detectar positivos reales (TPR) y generar falsas alarmas (FPR) al variar el umbral de decisión.

| Umbral | FPR (falsos positivos) | TPR / Recall |
|--------|------------------------|--------------|
| 1.0    | 0.00                   | 0.00         |
| 0.8    | 0.05                   | 0.60         |
| 0.5    | 0.10                   | 0.85         |
| 0.3    | 0.25                   | 0.95         |
| 0.0    | 1.00                   | 1.00         |

- Mientras más cerca de la **esquina superior izquierda** (FPR=0, TPR=1), mejor
- **AUC** (área bajo la curva): 1.0 = perfecto, 0.5 = aleatorio
- ⚠️ Puede ser optimista cuando las clases están muy desbalanceadas

---

## 📉 Curva Precision-Recall

Mejor que ROC cuando hay desbalance de clases (ej: 90% negativos, 10% positivos).

| Umbral | Precision | Recall |
|--------|-----------|--------|
| 0.9    | 0.98      | 0.40   |
| 0.7    | 0.90      | 0.70   |
| 0.5    | 0.82      | 0.85   |
| 0.3    | 0.65      | 0.95   |
| 0.1    | 0.50      | 1.00   |

- Subir el umbral → más precision, menos recall
- Bajarlo → más recall, más falsos positivos
- **AP** (Average Precision): área bajo esta curva; más útil que AUC en NLP

---

## ¿Cuándo usar cada una?

| Visualización | Úsala cuando... | Ventaja clave |
|---|---|---|
| **Matriz de confusión** | Quieres ver exactamente qué tipos de errores comete | No resume, muestra la verdad completa |
| **Curva ROC** | Las clases están balanceadas y quieres comparar modelos | Compara el trade-off TPR/FPR a todos los umbrales |
| **Curva Precision-Recall** | Hay desbalance de clases o el coste de falsos positivos es alto | Más honesta que ROC en escenarios reales de NLP |
