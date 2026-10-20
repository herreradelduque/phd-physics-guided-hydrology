# Physics-Guided Hydrology - PhD Thesis Code Repository

Python code for physics-guided, data-driven models of catchment-scale hydrological processes.

**Tesis Doctoral:** "Modelos de datos basados en aproximaciones físicas para los procesos hidrológicos a escala de cuenca"

**Autor:** Víctor Galán Vaquerizo
**Director:** Dr. Sergio Zubelzu
**Universidad Politécnica de Madrid**

---

## Repository Structure

This repository contains all the computational scripts used in the doctoral thesis, organized by thesis sections:

```
phd-physics-guided-hydrology/
├── notebooks/
│   ├── section_5.1_flow_prediction_events/      # Sección 5.1: Predicción de caudales
│   │   ├── 1-data-preparation.ipynb
│   │   ├── 2_1-forced-clustering.ipynb
│   │   ├── 3-predictive-models.ipynb
│   │   ├── 3_4-predictive-models-single-cluster.ipynb
│   │   ├── run_models.py
│   │   └── supplementary/
│   │       └── 6.1-ebro-complete-table-with-metrics.ipynb
│   │
│   └── section_5.3_soil_moisture_timeseries/    # Sección 5.3: Humedad del suelo
│       ├── 0.2-ebro-data-cleaning.ipynb
│       ├── 1-ebro-correlations.ipynb
│       ├── 2-ebro-clustering.ipynb
│       ├── 3-ebro-seasonal-analysis.ipynb
│       ├── 4.0-ebro-predictive-modeling.ipynb
│       ├── 4.1-ebro-cluster-based-advanced-models.ipynb
│       ├── 5-ebro-tables-values.ipynb
│       ├── 6.1-ebro-complete-table-with-metrics.ipynb
│       ├── 6.1.1-JOIN_notebook_6_csv_output.ipynb
│       ├── 7-rf-hyperparameter-tuning.ipynb
│       ├── 8-rf-predictions-and-analysis_PAPER_FIGURES.ipynb
│       └── scripts/                             # Python standalone scripts
│           ├── 7-rf-hyperparameter-tuning.py
│           ├── 7-rf-hyperparameter-tuning_2.py
│           ├── 7.1-multi-algorithm-hyperparameter-tuning.py
│           ├── 8-rf-predictions-and-analysis.py
│           ├── 8-rf-predictions-and-analysis_2.py
│           ├── 8-rf-predictions-and-analysis_3.py
│           ├── 8.1-multi-algorithm-predictions-and-analysis.py
│           ├── 8.1-multi-algorithm-predictions-and-analysis_PAPER_FIGURES.py
│           ├── CuencasTesisVictor_20251209.py
│           ├── run_model_1110.py
│           ├── aplicar_modificaciones_restantes_v3.py
│           ├── test_error_debugging.py
│           ├── test_variables.py
│           └── verificar_configuracion.py
│
└── scripts/
    └── qgis/
        └── anexo1_territorial_processing.py    # Anexo 1: Script QGIS

```

---

## Thesis Sections Coverage

### ✅ Sección 5.1: Predicción de caudales en cuencas pequeñas aforadas

**Objective:** Event-based prediction of streamflow in 19 small gauged basins of the Ebro River basin.

**Key Features:**
- Event-based hydrological analysis (precipitation → runoff)
- Physics-based clustering using CN (Curve Number) and GA (Green-Ampt) criteria
- Multiple ML algorithms: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost, SVR, MLP
- 8 model configurations × 6 scopes × 7+ algorithms = **336 models**
- Time series models: LSTM, ARIMA, SARIMA (**504 additional models**)

**Notebooks:** 6 files (5 main + 1 supplementary)

**Data:** 19 cuencas, event-based variables (Duration, Volume, Maximum for P/Q/R)

---

### ✅ Sección 5.3: Sensores IoT de Humedad del Suelo

**Objective:** Integration of SMAP satellite soil moisture data with climate and physical basin properties for streamflow prediction.

**Key Features:**
- Integration of SMAP (Soil Moisture Active Passive) satellite data
- 20 sub-basins (vs 19 in Section 5.1)
- Daily time series data (2020-2025)
- **73 variables** including:
  - Climate data (temperature, precipitation, discharge)
  - Basin physical properties (area, elevation, land use)
  - **SMAP soil moisture** (10 variables):
    - Root zone: sm_rootzone_av, sm_rootzone_dif, sm_rootzone_max, sm_rootzone_min
    - Surface: sm_surface_av, sm_surface_dif, sm_surface_max, sm_surface_min
    - Discharge: q0_r, q0_s
- Correlation analysis, seasonal patterns, clustering
- Advanced predictive models with hyperparameter tuning
- **Paper-ready figures** (8-rf-predictions-and-analysis_PAPER_FIGURES.ipynb)

**Notebooks:** 11 files + 14 Python scripts

**Data:** 20 cuencas, daily time series, SMAP integration

**Important:** Includes data quality corrections:
- Bailín basin discharge units correction (÷1000)
- Anomalous discharge values (>1000 m³/s) → NaN

---

### ⚠️ Sección 5.2: Análisis de Tiempo de Tránsito

**Status:** PENDING - Scripts not yet included

**Note:** High-resolution (15-minute) transit time analysis scripts are pending. Contact Sergio Zubelzu (sergio.zubelzu@upm.es) for these scripts.

---

## QGIS Script (Anexo 1)

**Location:** `scripts/qgis/anexo1_territorial_processing.py`

**Purpose:** Territorial data processing for basin characterization

**Functions:**
- Curve Number (CN) calculation from SIOSE land use data
- Soil texture classification (USDA triangle)
- Van Genuchten hydraulic parameters calculation
- Pixel-by-pixel soil parameter estimation
- Raster processing (sand, silt, clay) → hydraulic parameters
- Geographic extent transformation (EPSG:25830 → EPSG:4326)
- CSV export of basin properties

**Outputs:**
- 6 raster layers: ks, thetas, thetar, thau, swcfc, swcpwp
- CSV with basin statistics and properties

---

## Key Technologies

**Python Libraries:**
- Data: `pandas`, `numpy`
- ML: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- Deep Learning: `tensorflow` (LSTM)
- Time Series: `statsmodels` (ARIMA/SARIMA)
- Visualization: `matplotlib`, `seaborn`
- Geospatial: `qgis`, `gdal`, `osgeo`

**Development Environment:**
- Python 3.9+
- Jupyter Notebooks
- QGIS for spatial processing

---

## File Naming Convention

**Event-based models (Section 5.1):**
- `1-data-preparation.ipynb` - Data cleaning and preparation
- `2_1-forced-clustering.ipynb` - Physics-based clustering (CN/GA)
- `3-predictive-models.ipynb` - Main predictive models
- `3_4-predictive-models-single-cluster.ipynb` - Baseline (no clustering)

**Time series models (Section 5.3):**
- `0.X-ebro-data-cleaning.ipynb` - Data cleaning with SMAP integration
- `1-ebro-correlations.ipynb` - Correlation analysis
- `2-ebro-clustering.ipynb` - Clustering with soil moisture
- `3-ebro-seasonal-analysis.ipynb` - Seasonal patterns
- `4.X-ebro-predictive-modeling.ipynb` - Predictive models
- `6.X-ebro-complete-table-with-metrics.ipynb` - LSTM/ARIMA/SARIMA
- `7-rf-hyperparameter-tuning.ipynb` - Hyperparameter optimization
- `8-rf-predictions-and-analysis_PAPER_FIGURES.ipynb` - Publication figures

**Suffixes:**
- `_PAPER_FIGURES` - Final version with publication-ready figures
- `_OPTIMIZED` - Optimized version of the model
- Numbers (e.g., `_2`, `_3`) - Development versions

---

## Citation

If you use this code, please cite the doctoral thesis:

```
Galán Vaquerizo, V. (2026). Modelos de datos basados en aproximaciones físicas
para los procesos hidrológicos a escala de cuenca.
Tesis Doctoral, Universidad Politécnica de Madrid.
```

---

## License

See LICENSE file for details.

---

## Contact

**Víctor Galán Vaquerizo**
Universidad Politécnica de Madrid

**Director:** Dr. Sergio Zubelzu
Email: sergio.zubelzu@upm.es

---

## Acknowledgments

This research was conducted as part of a PhD program at Universidad Politécnica de Madrid under the supervision of Dr. Sergio Zubelzu.
