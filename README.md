# Clasificación de géneros musicales: Hip-Hop y Rock

Proyecto de **Hector Americo Torres Huamani** · Python y análisis de datos.

[Ver cuaderno](clasificador_canciones.ipynb) · [Abrir original en Colab](https://colab.research.google.com/drive/1ln1OIF-b0fEr9UUaqHJU3BBw7o84OUPW) · [Portafolio](https://www.datascienceportfol.io/hectortorres26/projects/1)

## Objetivo y contenido

Clasificar canciones de Hip-Hop y Rock a partir de características de audio de Echonest.

### Trabajo incluido

Unión de datos por `track_id`, análisis de correlaciones, estandarización, reducción de dimensionalidad con PCA, árbol de decisión, matriz de confusión, comparación con una muestra balanceada y validación cruzada.

### Datos

Archivos esperados: `songs.csv` y `echonest-metrics.json`. El cuaderno los lee desde Google Drive. La URL original de distribución y la licencia no aparecen en el cuaderno; deben documentarse antes de redistribuir los datasets.

### Resultados históricos guardados

| Experimento | Accuracy | F1 macro | Casos de prueba |
|---|---:|---:|---:|
| Muestra original | 0,84 | 0,75 | 1201 |
| Muestra balanceada | 0,74 | 0,74 | 455 |

Son salidas del cuaderno original, no resultados recalculados durante esta publicación. Los conjuntos de prueba tienen composiciones distintas; no permiten concluir por sí solos que una variante sea superior. Otras salidas del cuaderno no concuerdan con el primer informe, lo que sugiere ejecuciones en distinto orden.

### Revisión técnica

- Se usa `np.cumsum` y `np.mean` sin importar NumPy como `np`.
- La correlación incluye una etiqueta de texto; conviene seleccionar columnas numéricas.
- El escalador y PCA se ajustan antes de separar entrenamiento y prueba, tanto en la muestra original como en la balanceada: la evaluación tiene fuga de información.
- La validación cruzada recibe una proyección PCA calculada previamente con todos los datos, por lo que comparte el mismo problema.
- `KFold(10)` no estratifica y los datos balanceados se concatenan por clase; conviene evaluar con `StratifiedKFold` y mezcla reproducible.
- No se puede afirmar que el balanceo haya eliminado todo sesgo ni que la validación actual garantice generalización.

### Próximo paso recomendado

Separar primero un conjunto de prueba estratificado y mantener `StandardScaler`, PCA y el clasificador dentro de un `Pipeline`. Evaluar ese pipeline en cada partición y tratar el desbalance solo dentro del entrenamiento. Reportar F1 macro, métricas por clase y comparación con un clasificador base.

## Estado de esta publicación

Se conserva el cuaderno original, incluidas sus salidas históricas. La revisión del 9 de noviembre de 2026 fue estática: lectura del código, salidas y estructura. No se ejecutó el análisis con los datasets ni se corrigió silenciosamente el trabajo original. Los problemas anteriores deben resolverse antes de considerar reproducibles los resultados.

## Cómo explorar el proyecto

1. Abra el enlace de Colab o el archivo `.ipynb` en GitHub para consultar el trabajo.
2. Para trabajar localmente, cree un entorno virtual con `python -m venv .venv` y actívelo.
3. Instale las dependencias con `python -m pip install -r requirements.txt`.
4. Obtenga los datos desde su fuente autorizada. Ajuste las rutas `/content/drive/MyDrive/DATA_PRACTICA/` al directorio donde los haya guardado.
5. En un entorno local, omita las celdas de montaje de Google Drive, si las hay.
6. Resuelva los puntos de la revisión técnica y reinicie el kernel antes de ejecutar todas las celdas en orden.

`requirements.txt` enumera las librerías utilizadas; las versiones del entorno original no están registradas y no se ha validado una combinación exacta de versiones. Los datasets no se incluyen en este repositorio.

## Autoría y procedencia

Publicado a partir del cuaderno enlazado en el portafolio del autor. Las fuentes y licencias de los datos son independientes de la autoría del análisis. Esta publicación no añade una licencia de redistribución al código ni a los datasets.
