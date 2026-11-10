# Cambios respecto a la versión original

## Revisión técnica y editorial — 2026-11-10

| Problema original | Corrección |
|---|---|
| Uso de NumPy sin importación | Importaciones completas y ejecución desde un kernel limpio |
| Escalado y PCA ajustados antes de separar o validar | Preprocesamiento dentro de pipelines ajustados solo con entrenamiento |
| Validación no estratificada y cambios en la distribución de evaluación | Prueba estratificada fija y CV estratificada de cinco particiones |
| Conclusiones sobre sesgo sin sustento y resultados inconsistentes | Métricas regeneradas, referencia mayoritaria, errores por clase y límites explícitos |

Se añaden pruebas, dependencias fijadas, fuentes documentadas y resultados de ejecución. Las cifras nuevas corresponden al protocolo y los datasets identificados en esta revisión.

**Trazabilidad:** [cuaderno original](https://github.com/hectortorres26/clasificacion-generos-musicales/blob/4dffafd3f5e8f511ca13eded6ec59a2362b12e84/clasificador_canciones.ipynb) · [comparación completa](https://github.com/hectortorres26/clasificacion-generos-musicales/compare/4dffafd3f5e8f511ca13eded6ec59a2362b12e84...main) · [historial](https://github.com/hectortorres26/clasificacion-generos-musicales/commits/main).
