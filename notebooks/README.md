# Notebooks - Tesis Doctoral

Este directorio contiene los notebooks Jupyter y scripts Python utilizados para el análisis de datos y modelado hidrológico de la tesis doctoral:

**"Modelos de datos basados en aproximaciones físicas para los procesos hidrológicos a escala de cuenca"**

Autor: Víctor Galán Vaquerizo
Director: Dr. Sergio Zubelzu
Universidad Politécnica de Madrid

## Estructura

```
notebooks/
├── 1-data-preparation.ipynb              # Preparación de datos
├── 2_1-forced-clustering.ipynb           # Clustering físico (CN y GA)
├── 3-predictive-models.ipynb             # Modelos predictivos principales
├── 3_4-predictive-models-single-cluster.ipynb  # Baseline autorregresivo
├── run_models.py                         # Script standalone
└── supplementary/
    └── 6.1-ebro-complete-table-with-metrics.ipynb  # Análisis tiempo series
```

## Notebooks Principales

### 1. Data Preparation (`1-data-preparation.ipynb`)

**Propósito**: Preparación y limpieza de datos de eventos hidrológicos de 19 cuencas del Ebro.

**Inputs**:
- Archivos CSV con datos de eventos (precipitación, caudal, escorrentía)
- 8 variables por cuenca: Duration, Volume, Maximum (P/Q/R), Times

**Outputs**:
- `events_dataset.csv`: Dataset consolidado con todos los eventos
- Archivos JSON individuales por cuenca (datos limpios sin outliers)

**Metodología**:
- Detección y eliminación de outliers usando método IQR (Interquartile Range)
- Creación de variables derivadas (ratios entre variables)
- Normalización de nombres de cuencas

**Relación con tesis**: Sección 4 (Metodología) - Preparación de datos

---

### 2. Physics-Based Clustering (`2_1-forced-clustering.ipynb`)

**Propósito**: Agrupación de cuencas usando criterios físicos de hidrología.

**Inputs**:
- Características físicas de las cuencas
- Parámetros hidrológicos calculados

**Metodología**:
- **Criterio CN (Curve Number)**: 8 variables físicas → 3 clusters
  - Mountain (2 cuencas)
  - Medium (6 cuencas)
  - Plains (11 cuencas)

- **Criterio GA (Green-Ampt)**: 12 variables hidráulicas → 2 clusters
  - Basado en permeabilidad del suelo

**Outputs**:
- Asignaciones de cluster predefinidas (CN_cluster0-2, GA_cluster0-1)
- Visualizaciones PCA con loadings
- Dendrogramas de clustering jerárquico
- Figuras de calidad publicación (300 DPI)

**Relación con tesis**: Sección 5.1.2 - Clustering basado en aproximaciones físicas

---

### 3. Predictive Models (`3-predictive-models.ipynb`)

**Propósito**: Modelos de machine learning para predicción de variables hidrológicas usando clustering físico.

**Inputs**:
- `events_dataset.csv` (del notebook 1)
- Asignaciones de cluster (del notebook 2_1)

**Modelos implementados**:
- Linear Regression
- Ridge
- Lasso
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)

**Configuraciones**:
- 8 configuraciones de modelo (M1-M8)
- Variables objetivo: ymin, yav, ymax, qmin, qav, qmax, Dif_y, Dif_q
- 6 scopes: Global + CN_clusters (3) + GA_clusters (2)
- **Total**: 336 modelos (8 configs × 6 scopes × 7+ algoritmos)

**Métricas**:
- R² (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- NSE (Nash-Sutcliffe Efficiency)
- KGE (Kling-Gupta Efficiency)
- PBIAS (Percent Bias)
- MAPE (Mean Absolute Percentage Error)

**Outputs**:
- Tablas de resultados por modelo y scope
- Métricas de desempeño comparativas
- Análisis de importancia de variables

**Relación con tesis**: Sección 5.1 - Modelo para predicción de caudales en cuencas pequeñas aforadas (Tablas 3, 4, y resultados principales con **1,825 configuraciones**)

---

### 4. Single-Cluster Baseline (`3_4-predictive-models-single-cluster.ipynb`)

**Propósito**: Modelo baseline autorregresivo sin clustering para comparación.

**Metodología**:
- Todas las cuencas en un único cluster
- Mismos algoritmos y configuraciones que notebook 3
- Sirve como baseline para evaluar el valor añadido del clustering físico

**Relación con tesis**: Sección 5.1 - Análisis comparativo para validar beneficio del clustering

---

### 5. Standalone Execution Script (`run_models.py`)

**Propósito**: Script Python standalone para ejecutar modelos fuera de Jupyter.

**Características**:
- Permite ejecución batch de modelos
- Útil para computación en servidor sin interfaz Jupyter
- Incluye mismas funcionalidades que notebook 3
- Soporta XGBoost y CatBoost

**Uso**:
```bash
python run_models.py
```

---

## Análisis Suplementario

### 6. Time Series Models (`supplementary/6.1-ebro-complete-table-with-metrics.ipynb`)

**Propósito**: Análisis especializado con modelos de series temporales.

**Modelos**:
- LSTM (Long Short-Term Memory)
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)

**Configuraciones**:
- 72 configuraciones de series temporales
- 3 modelos × 72 configs = **216 combinaciones**
- Total ejecutado: **504 modelos** (168 por tipo de modelo)

**Metodología**:
- Preparación de datos secuenciales
- Early stopping para LSTM
- Time series split para validación
- Evaluación con métricas hidrológicas

**Outputs**:
- `notebook_6.1_time_series_models_[timestamp]-without-outliers.csv`
- Resultados comparativos entre modelos temporales

**Relación con tesis**: Análisis complementario explorando modelos temporales avanzados

---

## Orden de Ejecución Recomendado

1. **`1-data-preparation.ipynb`** - Generar dataset limpio
2. **`2_1-forced-clustering.ipynb`** - Crear asignaciones de clusters
3. **`3-predictive-models.ipynb`** - Entrenar modelos con clustering
4. **`3_4-predictive-models-single-cluster.ipynb`** - Entrenar baseline sin clustering
5. **`supplementary/6.1-ebro-complete-table-with-metrics.ipynb`** - (Opcional) Análisis tiempo series

## Dependencias

```python
# Python 3.9+
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.2
xgboost >= 1.7
lightgbm >= 3.3
catboost >= 1.1
matplotlib >= 3.5
seaborn >= 0.12
tqdm
scipy
```

Para modelos de series temporales (notebook 6.1):
```python
tensorflow >= 2.10  # Para LSTM
statsmodels >= 0.13  # Para ARIMA/SARIMA
```

## Datos

### Cuencas Estudiadas (19)

Las 19 cuencas pequeñas aforadas de la cuenca del Ebro incluyen:
- Bailín en Sabiñánigo
- Nela en Terán
- Larraun en Lekunberri
- [... y 16 cuencas adicionales]

### Variables

**Variables de precipitación (P)**:
- Duration: Duración del evento
- MaximoP: Precipitación máxima
- VolumenP: Volumen total de precipitación

**Variables de caudal (Q)**:
- MaximoQ: Caudal máximo
- VolumenQ: Volumen total de caudal
- Tiempos: Tiempo desde inicio de lluvia hasta pico de caudal

**Variables de escorrentía (R)**:
- MaximoR: Escorrentía máxima
- VolumenR: Volumen total de escorrentía

**Variables derivadas**:
- MaximoR_VolumenP: Ratio entre escorrentía máxima y precipitación total
- VolumenR_VolumenP: Coeficiente de escorrentía

## Notas Importantes

### Evolución del Código

Estos notebooks representan la **versión final** del análisis usado en la tesis doctoral. Versiones preliminares (nb_example/) han sido excluidas por claridad.

### Anexo 2 de la Tesis

Aunque la tesis menciona un "Anexo 2" con código de modelos precipitación-escorrentía, este anexo aparece incompleto en el documento Word. Los notebooks en este directorio constituyen la **implementación completa** de ese anexo.

### Scripts Faltantes

Ver `TODO.md` en el directorio raíz para información sobre scripts de las Secciones 5.2 (tiempo de tránsito) y 5.3 (sensores IoT) que están pendientes de obtener.

## Contacto

Para preguntas sobre estos notebooks:
- Víctor Galán Vaquerizo
- Director: Dr. Sergio Zubelzu (sergio.zubelzu@upm.es)
- Universidad Politécnica de Madrid

## Licencia

Este código es parte de una tesis doctoral en la Universidad Politécnica de Madrid.
