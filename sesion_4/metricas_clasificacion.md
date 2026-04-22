# Métricas de Clasificación

| Métrica | Qué mide (simple) | ¿Puede engañar? | ¿Por qué? |
|---|---|---|---|
| **Accuracy** | Qué tanto acierta en general | ⚠️ Sí | Si hay muchas más de una clase (ej: 90% positivos), puede verse alto aunque el modelo sea malo |
| **Precision** | Qué tan confiable es cuando dice "positivo" | ⚠️ A veces | Puede ser alta aunque ignore muchos positivos reales |
| **Recall** | Qué tanto detecta lo importante | ⚠️ A veces | Puede ser alto aunque genere muchos falsos positivos |
| **F1-score** | Balance entre precision y recall | ⚠️ Poco | Puede ocultar si precision y recall están muy desbalanceados |
| **Matriz de confusión** | Dónde se equivoca exactamente | ❌ No | No resume en un número, pero muestra la verdad completa |

## 🧠 Lectura rápida

- Si quieres una sola métrica → **F1**
- Si quieres entender errores → **matriz de confusión**
- Si ves accuracy muy alto → **sospecha primero**
