# Inventory of Thesis Scripts and Notebooks

**PhD Thesis:** "Modelos de datos basados en aproximaciones físicas para los procesos hidrológicos a escala de cuenca"

**Author:** Víctor Galán Vaquerizo
**Director:** Dr. Sergio Zubelzu
**Universidad Politécnica de Madrid**

**Last Updated:** 2026-10-20

---

## Table of Contents

- [Section 5.1: Flow Prediction (Event-Based Analysis)](#section-51-flow-prediction-event-based-analysis)
- [Section 5.3: Soil Moisture and Time Series](#section-53-soil-moisture-and-time-series)
- [QGIS Scripts (Anexo 1)](#qgis-scripts-anexo-1)
- [Execution Workflow](#execution-workflow)

---

## Section 5.1: Flow Prediction (Event-Based Analysis)

**Location:** `notebooks/section_5.1_flow_prediction_events/`

**Objective:** Event-based prediction of streamflow in 19 small gauged basins of the Ebro River basin using physics-based clustering and machine learning.

**Datasets:** 19 cuencas (basins), event-based hydrological data

---

### 1-data-preparation.ipynb

**Description:** Prepares and cleans event-based hydrological data from 19 basins, performs outlier detection, and creates consolidated dataset.

**Section:** 5.1

**Inputs:**
- Raw CSV files with event data per basin (8 variables):
  - `AgregadosPorCuencasDuracionP.csv` - Storm duration
  - `AgregadosPorCuencasVolumenP.csv` - Precipitation volume
  - `AgregadosPorCuencasMaximoP.csv` - Maximum precipitation intensity
  - `AgregadosPorCuencasMaximoQ.csv` - Peak discharge
  - `AgregadosPorCuencasMaximoR.csv` - Maximum runoff
  - `AgregadosPorCuencasVolumenQ.csv` - Discharge volume
  - `AgregadosPorCuencasVolumenR.csv` - Runoff volume
  - `AgregadosPorCuencasTiempos.csv` - Time to peak

**Outputs:**
- `events_dataset.csv` - Consolidated wide-format dataset with all events
- Individual JSON files per basin with clean data (outliers removed)
- Derived variables (ratios between variables)

**Dependencies:** None (execute first)

**Execution Order:** 1

**Key Methods:**
- IQR (Interquartile Range) method for outlier detection
- Wide-to-long format transformation
- Basin name normalization

---

### 2_1-forced-clustering.ipynb

**Description:** Physics-based clustering of basins using Curve Number (CN) and Green-Ampt (GA) hydrological criteria.

**Section:** 5.1

**Inputs:**
- Basin physical properties from `cuencas_unificadas.csv`:
  - **CN variables (8):** CN_ponderado, CN_DT, n, DT_n, area_cuenca, longitud_total, z_min, z_max
  - **GA variables (12):** mean_ks, DT_ks, mean_thetas, DT_thetas, mean_thetar, DT_thetar, mean_thau, DT_thau, area_cuenca, longitud_total, z_min, z_max

**Outputs:**
- **CN clustering:** 3 clusters
  - Mountain (2 basins)
  - Medium (6 basins)
  - Plains (11 basins)
- **GA clustering:** 2 clusters based on soil permeability
- PCA visualizations with loadings
- Hierarchical clustering dendrograms
- Publication-quality figures (300 DPI)

**Dependencies:** None (can run independently)

**Execution Order:** 2 (or parallel with step 1)

**Key Methods:**
- Hierarchical clustering with Ward linkage
- Principal Component Analysis (PCA)
- Cluster assignments: `CN_cluster0-2`, `GA_cluster0-1`

---

### 3-predictive-models.ipynb

**Description:** Comprehensive machine learning models for predicting hydrological variables using physics-based clustering.

**Section:** 5.1

**Inputs:**
- `events_dataset.csv` (from notebook 1)
- Cluster assignments (from notebook 2_1)

**Outputs:**
- Results tables with performance metrics for **336 model configurations**:
  - 8 model configurations (M1-M8) × 6 scopes × 7+ algorithms
- Performance metrics per model:
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - NSE (Nash-Sutcliffe Efficiency)
  - KGE (Kling-Gupta Efficiency)
  - PBIAS (Percent Bias)
  - MAPE (Mean Absolute Percentage Error)
- Variable importance analysis

**Dependencies:**
- Notebook 1 (data preparation)
- Notebook 2_1 (cluster assignments)

**Execution Order:** 3

**Algorithms Implemented:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)

**Target Variables:**
- ymin, yav, ymax (discharge quantiles)
- qmin, qav, qmax (runoff quantiles)
- Dif_y, Dif_q (differences)

**Scopes:**
- Global (all basins)
- CN_cluster0, CN_cluster1, CN_cluster2
- GA_cluster0, GA_cluster1

---

### 3_4-predictive-models-single-cluster.ipynb

**Description:** Baseline autoregressive model with all basins in a single cluster (no physics-based clustering).

**Section:** 5.1

**Inputs:**
- `events_dataset.csv` (from notebook 1)

**Outputs:**
- Baseline performance metrics
- Comparison metrics to evaluate benefit of physics-based clustering

**Dependencies:**
- Notebook 1 (data preparation)

**Execution Order:** 4 (can run in parallel with notebook 3)

**Purpose:** Serves as baseline to demonstrate the value added by physics-based clustering (CN/GA).

---

### run_models.py

**Description:** Standalone Python script for executing predictive models outside Jupyter environment.

**Section:** 5.1

**Inputs:**
- Same as notebook 3 (events_dataset.csv, cluster assignments)

**Outputs:**
- Same as notebook 3 (model results, metrics)

**Dependencies:**
- Data preparation must be completed
- Cluster assignments available

**Execution Order:** Alternative to notebook 3 for batch execution

**Usage:**
```bash
python run_models.py
```

**Key Features:**
- Supports XGBoost and CatBoost
- Useful for server execution without Jupyter
- Same functionality as notebook 3

---

### supplementary/6.1-ebro-complete-table-with-metrics.ipynb

**Description:** Time series analysis using LSTM, ARIMA, and SARIMA models.

**Section:** 5.1 (Supplementary Analysis)

**Inputs:**
- Event-based time series data
- Sequential event ordering per basin

**Outputs:**
- `notebook_6.1_time_series_models_[timestamp]-without-outliers.csv`
- Performance metrics for **504 time series models**:
  - 72 configurations × 3 model types (LSTM, ARIMA, SARIMA)
  - 168 models per type

**Dependencies:**
- Notebook 1 (data preparation)

**Execution Order:** 5 (supplementary, can run independently after step 1)

**Models:**
- **LSTM (Long Short-Term Memory):** Deep learning for sequential patterns
- **ARIMA (AutoRegressive Integrated Moving Average):** Classical time series
- **SARIMA (Seasonal ARIMA):** Seasonal time series patterns

**Key Features:**
- Time series split for validation
- Early stopping for LSTM
- Comprehensive configuration space exploration

---

## Section 5.3: Soil Moisture and Time Series

**Location:** `notebooks/section_5.3_soil_moisture_timeseries/`

**Objective:** Integration of SMAP satellite soil moisture data with climate and physical basin properties for daily streamflow prediction.

**Datasets:** 20 cuencas (vs 19 in 5.1), daily time series 2020-2025, **73 variables including SMAP**

---

### 0.2-ebro-data-cleaning.ipynb

**Description:** Data cleaning and integration of SMAP satellite soil moisture data with climate and physical properties.

**Section:** 5.3

**Inputs:**
- **Climate archive:** Daily climate data per basin (2020-2025)
  - Temperature, precipitation, discharge (qmax, qav, qmin)
- **Physical properties archive:** Basin characteristics
  - Area, elevation, land use, soil properties
- **SMAP archive:** Soil moisture from SMAP satellite
  - Root zone and surface soil moisture measurements

**Outputs:**
- **Merged dataset with 73 variables:**
  - Climate data (~20 columns)
  - Physical properties (~39 columns)
  - Humidity data (~4 columns)
  - **SMAP data (10 columns):**
    - sm_rootzone_av, sm_rootzone_dif, sm_rootzone_max, sm_rootzone_min
    - sm_surface_av, sm_surface_dif, sm_surface_max, sm_surface_min
    - q0_r, q0_s
- Clean JSON files per basin (outliers removed, NaN handled)

**Dependencies:** None (execute first)

**Execution Order:** 1

**Data Quality Corrections:**
- **Bailín basin:** Discharge values divided by 1000 (unit correction)
- **Anomalous values:** Discharge >1000 m³/s → NaN (preserves data integrity)
- Dropna operations on all data sources

**Key Features:**
- Inner join merge strategy (only dates with all data sources)
- ~1777 days of overlapping data after merge
- 20 basins processed (includes Trueba and Zidacos, missing in Section 5.1)

---

### 1-ebro-correlations.ipynb

**Description:** Comprehensive correlation analysis between climate, physical, SMAP, and discharge variables.

**Section:** 5.3

**Inputs:**
- Merged dataset from notebook 0.2 (73 variables)

**Outputs:**
- Correlation matrices and heatmaps
- Key variable correlation summaries
- Scatter plots for significant correlations

**Dependencies:**
- Notebook 0.2 (data cleaning)

**Execution Order:** 2

**Key Analyses:**
- Climate vs discharge correlations
- SMAP soil moisture vs discharge
- Physical properties vs hydrological response
- Multivariate correlation patterns

---

### 2-ebro-clustering.ipynb

**Description:** Clustering analysis incorporating SMAP soil moisture variables.

**Section:** 5.3

**Inputs:**
- Merged dataset from notebook 0.2
- Basin physical properties + SMAP variables

**Outputs:**
- Cluster assignments incorporating soil moisture
- PCA visualizations
- Cluster characteristic analysis

**Dependencies:**
- Notebook 0.2 (data cleaning)

**Execution Order:** 3 (can run in parallel with notebook 1)

**Key Features:**
- Extended clustering with SMAP variables
- Comparison with Section 5.1 clustering (CN/GA)

---

### 3-ebro-seasonal-analysis.ipynb

**Description:** Analysis of seasonal patterns in discharge, precipitation, and soil moisture.

**Section:** 5.3

**Inputs:**
- Merged daily time series from notebook 0.2

**Outputs:**
- Seasonal statistics per basin
- Monthly/seasonal correlation patterns
- Seasonal cluster visualizations

**Dependencies:**
- Notebook 0.2 (data cleaning)

**Execution Order:** 4

**Key Analyses:**
- Seasonal discharge patterns
- Seasonal soil moisture dynamics
- Climate-discharge relationships by season

---

### 4.0-ebro-predictive-modeling.ipynb

**Description:** Predictive models for daily discharge using climate and SMAP data.

**Section:** 5.3

**Inputs:**
- Merged daily time series (73 variables)
- SMAP soil moisture data

**Outputs:**
- Model performance metrics
- Feature importance analysis
- Predictions vs observations plots

**Dependencies:**
- Notebook 0.2 (data cleaning)
- Notebook 2 (cluster assignments) - optional

**Execution Order:** 5

**Models:**
- Random Forest, XGBoost, LightGBM
- Linear models (Ridge, Lasso)
- Support Vector Regression

---

### 4.1-ebro-cluster-based-advanced-models.ipynb

**Description:** Advanced predictive models using cluster-specific training with SMAP integration.

**Section:** 5.3

**Inputs:**
- Merged dataset from notebook 0.2
- Cluster assignments from notebook 2

**Outputs:**
- Cluster-specific model performance
- Comparison: global vs cluster-based models
- Feature importance per cluster

**Dependencies:**
- Notebook 0.2 (data cleaning)
- Notebook 2 (clustering)

**Execution Order:** 6

**Key Features:**
- Cluster-specific hyperparameters
- Enhanced performance vs global models
- SMAP variables as key predictors

---

### 5-ebro-tables-values.ipynb

**Description:** Generation of summary tables with basin properties and model results for thesis.

**Section:** 5.3

**Inputs:**
- Merged dataset
- Model results from notebooks 4.0, 4.1

**Outputs:**
- Formatted tables for thesis document
- Summary statistics per basin
- Comparative performance tables

**Dependencies:**
- Notebooks 0.2, 4.0, 4.1

**Execution Order:** 7

**Purpose:** Generates tables for inclusion in thesis manuscript.

---

### 6.1-ebro-complete-table-with-metrics.ipynb

**Description:** Comprehensive time series modeling with LSTM, ARIMA, and SARIMA including SMAP data.

**Section:** 5.3

**Inputs:**
- Daily time series from notebook 0.2
- SMAP soil moisture as additional features

**Outputs:**
- Time series model performance metrics
- CSV with complete results table
- Comparison across model types

**Dependencies:**
- Notebook 0.2 (data cleaning)

**Execution Order:** 8

**Models:**
- LSTM with SMAP features
- ARIMA with exogenous SMAP variables
- SARIMA with seasonal patterns

**Configuration:** 72 configurations per model type

---

### 6.1.1-JOIN_notebook_6_csv_output.ipynb

**Description:** Joins and consolidates output CSV files from notebook 6.1 time series models.

**Section:** 5.3

**Inputs:**
- Multiple CSV outputs from notebook 6.1 runs
- `notebook_6.1_time_series_models_*.csv` files

**Outputs:**
- Consolidated CSV with all time series results
- Unified performance comparison table

**Dependencies:**
- Notebook 6.1 (must be executed multiple times)

**Execution Order:** 9

**Purpose:** Aggregates results from different notebook 6.1 executions into single table.

---

### 7-rf-hyperparameter-tuning.ipynb

**Description:** Hyperparameter tuning for Random Forest models using grid search.

**Section:** 5.3

**Inputs:**
- Merged dataset from notebook 0.2

**Outputs:**
- Optimal Random Forest hyperparameters
- Tuning results CSV
- Performance comparison: default vs tuned

**Dependencies:**
- Notebook 0.2 (data cleaning)

**Execution Order:** 10 (can run in parallel with other notebooks)

**Hyperparameters Tuned:**
- n_estimators, max_depth, min_samples_split, min_samples_leaf
- Grid search with cross-validation

---

### 8-rf-predictions-and-analysis_PAPER_FIGURES.ipynb

**Description:** Final analysis notebook with publication-ready figures for paper/thesis.

**Section:** 5.3

**Inputs:**
- Merged dataset
- Tuned model parameters from notebook 7

**Outputs:**
- **Publication-quality figures (300+ DPI)**
- Prediction vs observation plots
- Residual analysis
- Feature importance plots
- Performance comparison charts

**Dependencies:**
- Notebook 0.2 (data cleaning)
- Notebook 7 (hyperparameter tuning) - optional

**Execution Order:** 11 (final notebook for results presentation)

**Key Features:**
- High-resolution figures for publication
- Comprehensive model evaluation
- Statistical significance tests
- Final results synthesis

---

### scripts/ (Python Standalone Scripts)

**Location:** `notebooks/section_5.3_soil_moisture_timeseries/scripts/`

**Purpose:** Standalone Python scripts for batch execution, testing, and specialized analyses.

---

#### 7-rf-hyperparameter-tuning.py

**Description:** Standalone version of notebook 7 for RF hyperparameter tuning.

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 7-rf-hyperparameter-tuning.py
```

**Inputs:** Same as notebook 7

**Outputs:** Tuned hyperparameters CSV

---

#### 7-rf-hyperparameter-tuning_2.py

**Description:** Alternative version of RF tuning script with different parameter grid.

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 7-rf-hyperparameter-tuning_2.py
```

**Differences:** Expanded parameter search space

---

#### 7.1-multi-algorithm-hyperparameter-tuning.py

**Description:** Extended tuning script for multiple algorithms (RF, XGBoost, SVM).

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 7.1-multi-algorithm-hyperparameter-tuning.py
```

**Algorithms:** Random Forest, XGBoost, SVM

**Outputs:** Tuned parameters for all three algorithms

---

#### 8-rf-predictions-and-analysis.py

**Description:** Standalone version of RF predictions and analysis.

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 8-rf-predictions-and-analysis.py
```

---

#### 8-rf-predictions-and-analysis_2.py

**Description:** Alternative version with modified analysis workflow.

**Section:** 5.3 (scripts)

---

#### 8-rf-predictions-and-analysis_3.py

**Description:** Third version with additional evaluation metrics.

**Section:** 5.3 (scripts)

---

#### 8.1-multi-algorithm-predictions-and-analysis.py

**Description:** Multi-algorithm predictions (RF, XGBoost, SVM) with comprehensive analysis.

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 8.1-multi-algorithm-predictions-and-analysis.py
```

**Algorithms:** RF, XGBoost, SVM

**Outputs:** Comparative performance analysis

---

#### 8.1-multi-algorithm-predictions-and-analysis_PAPER_FIGURES.py

**Description:** Paper-ready figures version of multi-algorithm analysis.

**Section:** 5.3 (scripts)

**Usage:**
```bash
python 8.1-multi-algorithm-predictions-and-analysis_PAPER_FIGURES.py
```

**Outputs:** High-resolution publication figures

---

#### CuencasTesisVictor_20251209.py

**Description:** Utility script for basin-specific processing and analysis.

**Section:** 5.3 (scripts)

**Date:** 2025-12-09

**Purpose:** Basin data management and processing utilities

---

#### run_model_1110.py

**Description:** Model execution script (version from November 10).

**Section:** 5.3 (scripts)

**Usage:**
```bash
python run_model_1110.py
```

---

#### aplicar_modificaciones_restantes_v3.py

**Description:** Script for applying remaining data modifications (version 3).

**Section:** 5.3 (scripts)

**Purpose:** Data correction and transformation pipeline

---

#### test_error_debugging.py

**Description:** Debugging script for error diagnosis in models.

**Section:** 5.3 (scripts)

**Purpose:** Development and debugging tool

---

#### test_variables.py

**Description:** Script for testing variable transformations and validations.

**Section:** 5.3 (scripts)

**Purpose:** Data quality testing and validation

---

#### verificar_configuracion.py

**Description:** Configuration verification script.

**Section:** 5.3 (scripts)

**Purpose:** Validates model configuration and environment setup

---

## QGIS Scripts (Anexo 1)

**Location:** `scripts/qgis/`

---

### anexo1_territorial_processing.py

**Description:** QGIS script for territorial data processing and basin characterization.

**Section:** Anexo 1

**Inputs:**
- QGIS project layers:
  - Land use polygons (SIOSE classification)
  - Stream network lines
  - Digital Elevation Model (DEM/MDT)
  - Soil texture rasters (sand, silt, clay)
  - Basin boundary vector layer

**Outputs:**
- **6 raster layers:**
  - `param_ks.tif` - Saturated hydraulic conductivity
  - `param_thetas.tif` - Saturated water content
  - `param_thetar.tif` - Residual water content
  - `param_thau.tif` - Tau parameter
  - `param_swcfc.tif` - Soil water content at field capacity
  - `param_swcpwp.tif` - Soil water content at permanent wilting point
- **CSV file:** Basin properties summary with all calculated parameters

**Dependencies:** QGIS environment with required layers loaded

**Execution:** Run from QGIS Python console

**Key Functions:**

1. **CNs(SIOSE):** Calculates Curve Number from SIOSE land use code
   - Input: SIOSE land use classification code
   - Output: Curve Number (CN) value

2. **SoilTexture(sand, silt, clay):** Determines soil texture class using USDA triangle
   - Input: Sand, silt, clay percentages
   - Output: Soil texture class (Sand, Loam, Clay, etc.)

3. **soil_param(soil):** Calculates van Genuchten hydraulic parameters
   - Input: Soil texture class
   - Output: ks, thetas, thetar, thau, swcfc, swcpwp

4. **Neuman(ks, thetas, thetar, alpha, m):** Calculates tau parameter using Neuman method
   - Input: Hydraulic parameters
   - Output: Tau value

**Processing Steps:**
1. Load QGIS project layers
2. Clip rasters to basin boundary
3. Calculate weighted Curve Number from land use
4. Extract soil texture from raster data
5. Determine soil type (USDA classification)
6. Calculate hydraulic parameters pixel-by-pixel
7. Compute basin-wide statistics (mean, std dev)
8. Transform coordinates EPSG:25830 → EPSG:4326
9. Export results to GeoTIFF rasters and CSV

**Hydrological Parameters Calculated:**
- **ks:** Saturated hydraulic conductivity (cm/day)
- **thetas:** Saturated water content (-)
- **thetar:** Residual water content (-)
- **thau:** Tau parameter (cm²/day)
- **swcfc:** Soil water content at field capacity (-)
- **swcpwp:** Soil water content at permanent wilting point (-)

**Geographic Processing:**
- Coordinate transformation: EPSG:25830 (UTM) → EPSG:4326 (WGS84)
- Raster clipping to basin boundaries
- Statistical aggregation (mean, standard deviation)

---

## Execution Workflow

### Section 5.1: Event-Based Flow Prediction

**Recommended Execution Order:**

```
1. 1-data-preparation.ipynb              [~15-30 min]
   ↓
2. 2_1-forced-clustering.ipynb           [~10-20 min]
   ↓
3. 3-predictive-models.ipynb             [~1-3 hours] - 336 models
   ↓
4. 3_4-predictive-models-single-cluster.ipynb [~30-60 min] - baseline
   ↓
5. supplementary/6.1-ebro-complete-table-with-metrics.ipynb [~2-4 hours] - LSTM/ARIMA/SARIMA
```

**Alternative:** Use `run_models.py` instead of notebook 3 for batch execution.

**Parallel Options:**
- Notebooks 2_1 and 3_4 can run in parallel with notebook 3
- Notebook 5 (supplementary) can run independently after notebook 1

---

### Section 5.3: Soil Moisture and Time Series

**Recommended Execution Order:**

```
1. 0.2-ebro-data-cleaning.ipynb          [~20-40 min] - SMAP integration
   ↓
2. 1-ebro-correlations.ipynb             [~15-30 min]
   ↓
3. 2-ebro-clustering.ipynb               [~15-30 min]
   ↓
4. 3-ebro-seasonal-analysis.ipynb        [~20-40 min]
   ↓
5. 4.0-ebro-predictive-modeling.ipynb    [~30-60 min]
   ↓
6. 4.1-ebro-cluster-based-advanced-models.ipynb [~45-90 min]
   ↓
7. 5-ebro-tables-values.ipynb            [~10-20 min]
   ↓
8. 6.1-ebro-complete-table-with-metrics.ipynb [~2-4 hours] - Time series models
   ↓
9. 6.1.1-JOIN_notebook_6_csv_output.ipynb [~5-10 min]
   ↓
10. 7-rf-hyperparameter-tuning.ipynb     [~1-2 hours] - Grid search
   ↓
11. 8-rf-predictions-and-analysis_PAPER_FIGURES.ipynb [~30-60 min] - Final figures
```

**Parallel Options:**
- Notebooks 1, 2, 3, 4 can run in parallel after notebook 0.2
- Notebook 7 (hyperparameter tuning) can run anytime after notebook 0.2
- Notebook 10 can use default parameters if notebook 7 is skipped

**Script Execution:**
All scripts in `scripts/` directory can be executed standalone:
```bash
cd notebooks/section_5.3_soil_moisture_timeseries/scripts/
python [script_name].py
```

---

### QGIS Processing (Anexo 1)

**Execution:**

1. Open QGIS project with required layers
2. Load script in Python Console:
   ```python
   exec(open('scripts/qgis/anexo1_territorial_processing.py').read())
   ```
3. Follow prompts to save CSV output
4. Review generated raster layers in QGIS

**Required Layers:**
- Land use polygons (SIOSE)
- Stream network
- Digital Elevation Model
- Soil texture rasters (sand, silt, clay)
- Basin boundary vector

**Execution Time:** ~10-30 minutes depending on basin size

---

## Summary Statistics

| Category | Count | Lines of Code (approx) |
|----------|-------|----------------------|
| **Section 5.1 Notebooks** | 6 | ~11,000 |
| **Section 5.3 Notebooks** | 11 | ~53,000 |
| **Section 5.3 Scripts** | 14 | ~3,000 |
| **QGIS Scripts** | 1 | ~550 |
| **TOTAL** | **32** | **~67,550** |

---

## Models Implemented

| Section | Model Type | Count |
|---------|-----------|-------|
| 5.1 Event-Based | ML algorithms (9 types) × configurations | 336 |
| 5.1 Time Series | LSTM, ARIMA, SARIMA | 504 |
| 5.3 Daily Prediction | RF, XGBoost, SVM, others | Variable |
| 5.3 Time Series | LSTM, ARIMA, SARIMA with SMAP | 216 |
| **TOTAL MODELS** | | **1000+** |

---

## Key Datasets

### Section 5.1
- **Basins:** 19
- **Format:** Event-based (discrete hydrological events)
- **Variables:** 8 per event (Duration, Volume, Maximum for P/Q/R, Times)
- **Events:** ~hundreds per basin

### Section 5.3
- **Basins:** 20 (includes Trueba and Zidacos)
- **Format:** Daily time series
- **Period:** 2020-2025 (~1777 days with complete data)
- **Variables:** 73 total (climate + physical + humidity + SMAP)
- **SMAP Variables:** 10 (soil moisture root zone, surface, and derived discharge)

---

## Dependencies and Requirements

### Python Libraries

**Core:**
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.5
- seaborn >= 0.12

**Machine Learning:**
- scikit-learn >= 1.2
- xgboost >= 1.7
- lightgbm >= 3.3
- catboost >= 1.1

**Deep Learning (Section 5.1/5.3 Time Series):**
- tensorflow >= 2.10

**Time Series (ARIMA/SARIMA):**
- statsmodels >= 0.13

**Utilities:**
- tqdm
- scipy >= 1.8

**Geospatial (QGIS Script):**
- qgis
- gdal
- osgeo

### Environment

- Python 3.9+
- Jupyter Notebook or JupyterLab
- QGIS 3.x (for Anexo 1 script)

---

## Notes

### Data Quality
- **Section 5.3** includes critical data corrections:
  - Bailín basin discharge units (÷1000)
  - Anomalous discharge values (>1000 m³/s → NaN)

### File Versions
- Files with `_PAPER_FIGURES` suffix contain publication-ready visualizations
- Numbered versions (\_2, \_3, etc.) represent development iterations
- Final versions prioritized in this repository

### Execution Environment
- Notebooks designed for sequential execution within each section
- Some notebooks can run in parallel (noted in workflow)
- Scripts provide alternative CLI execution option

---

**Generated:** 2026-10-20
**Repository:** https://github.com/[username]/phd-physics-guided-hydrology

