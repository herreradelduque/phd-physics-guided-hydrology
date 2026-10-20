#!/usr/bin/env python
# coding: utf-8

# # Notebook 7: Random Forest Hyperparameter Tuning
# 
# ## Objetivo
# 
# Este notebook realiza **ajuste fino de hiperparámetros** (hyperparameter tuning) para los 8 modelos Random Forest que han resultado ser los mejores modelos causales.
# 
# **Modelos a optimizar:**
# 1. **ymin** (Modelo 1110) - Eq 6 con humedad promedio
# 2. **yav** (Modelo 1123) - Eq 7 con humedad promedio
# 3. **ymax** (Modelo 1112) - Eq 6 con humedad promedio
# 4. **Dif_y** (Modelo 1214) - Eq 6 variables sintéticas
# 5. **qmin** (Modelo 1131) - Eq 9a
# 6. **qav** (Modelo 1132) - Eq 9a
# 7. **qmax** (Modelo 1133) - Eq 9a
# 8. **Dif_q** (Modelo 1221) - Eq 9a variables sintéticas
# 
# **Metodología:**
# - Grid Search con cross-validation
# - Comparación: días aleatorios vs rangos temporales
# - Tipificación de variables (StandardScaler)
# - Análisis de importancia de variables
# 
# **Autor:** Víctor Galán  
# **Fecha:** 2026-03-14

# ## 1. Imports y Configuración

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import warnings
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Scipy for statistical calculations
from scipy import stats

warnings.filterwarnings('ignore')

print("=" * 80)
print("NOTEBOOK 7: RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 80)
print(f"✅ Imports loaded successfully")
print(f"   numpy version: {np.__version__}")
print(f"   pandas version: {pd.__version__}")
print(f"   sklearn version: {__import__('sklearn').__version__}")


# ## 2. Funciones de Métricas Hidrológicas
# 
# Reutilizamos las mismas métricas del notebook 6.

# In[2]:


def calculate_nse(y_true, y_pred):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    NSE = 1 - [Σ(y_true - y_pred)²] / [Σ(y_true - mean(y_true))²]
    Range: -∞ to 1 (perfect fit)
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - (numerator / denominator)

def calculate_kge(y_true, y_pred):
    """
    Calculate Kling-Gupta Efficiency (KGE).
    KGE = 1 - sqrt[(r-1)² + (α-1)² + (β-1)²]
    """
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def calculate_pbias(y_true, y_pred):
    """
    Calculate Percent Bias (PBIAS).
    PBIAS = [Σ(y_true - y_pred) / Σ(y_true)] × 100
    """
    if np.sum(y_true) == 0:
        return np.nan

    return (np.sum(y_true - y_pred) / np.sum(y_true)) * 100

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all metrics at once.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse = calculate_nse(y_true, y_pred)
    kge = calculate_kge(y_true, y_pred)
    pbias = calculate_pbias(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'NSE': nse,
        'KGE': kge,
        'PBIAS_%': pbias,
        'MAPE_%': mape
    }

print("✅ Metric functions defined")


# ## 3. Carga de Datos

# In[3]:


# Load merged data (sin outliers)
base_path = "../../data/clean/merged_data/" # SERGIO: sustituye este path por el tuyo, donde descomprimas el zip merged_data.zip

if not os.path.exists(base_path):
    raise FileNotFoundError(f"Path {base_path} does not exist. Run 0-ebro-data-cleaning.ipynb first.")

join_dict = {}
for filename in os.listdir(base_path):
    if filename.endswith("-without-outliers.json"):
        basin_name = filename.replace("-without-outliers.json", "")
        df = pd.read_json(os.path.join(base_path, filename), orient="records")

        # Ensure Date column exists and is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        join_dict[basin_name] = df

print("=" * 80)
print("DATA LOADING")
print("=" * 80)
print(f"✅ Loaded {len(join_dict)} basins")
print(f"📊 Basins: {sorted(list(join_dict.keys()))}")

# Sample basin for verification
sample_basin = list(join_dict.keys())[0]
print(f"\n📋 Sample basin: {sample_basin}")
print(f"   Shape: {join_dict[sample_basin].shape}")
print(f"   Columns: {len(join_dict[sample_basin].columns)}")


# ## 4. Crear Variables Sintéticas

# In[4]:


# Create synthetic variables
for basin_name, df in join_dict.items():
    # Temperature difference
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['Dif_t'] = df['tmax'] - df['tmin']

    # Humidity difference
    if 'Hrmax' in df.columns and 'Hrmin' in df.columns:
        df['Dif_HR'] = df['Hrmax'] - df['Hrmin']

    # Flow difference
    if 'qmax' in df.columns and 'qmin' in df.columns:
        df['Dif_q'] = df['qmax'] - df['qmin']

    # Water level difference
    if 'ymax' in df.columns and 'ymin' in df.columns:
        df['Dif_y'] = df['ymax'] - df['ymin']

    # Soil moisture differences (SMAP)
    if 'sm_rootzone_max' in df.columns and 'sm_rootzone_min' in df.columns:
        df['Dif_qr'] = df['sm_rootzone_max'] - df['sm_rootzone_min']

    if 'sm_surface_max' in df.columns and 'sm_surface_min' in df.columns:
        df['Dif_qs'] = df['sm_surface_max'] - df['sm_surface_min']

    join_dict[basin_name] = df

print("✅ Synthetic variables created")


# ## 5. Mapeo de Variables y Definición de Modelos

# In[5]:


# Variable mapping: Paper variables -> Real column names
var_mapping = {
    # Climate
    'P': 'P',
    'tav': 'tav',
    'tmax': 'tmax',
    'tmin': 'tmin',
    'Hrav': 'Hrav',
    'Hrmax': 'Hrmax',
    'Hrmin': 'Hrmin',
    'wv': 'wv',
    'wvmax': 'wvmax',
    'SolarRad': 'SolarRad',
    'et0': 'et0',

    # SMAP soil moisture
    'Av_qr': 'sm_rootzone_av',
    'Av_qs': 'sm_surface_av',
    'q0r': 'q0_r',  # ⚠️ CORREGIDO: con guión bajo
    'q0s': 'q0_s',  # ⚠️ CORREGIDO: con guión bajo
    'Dif_qr': 'Dif_qr',
    'Dif_qs': 'Dif_qs',

    # Synthetic
    'Dif_t': 'Dif_t',
    'Dif_HR': 'Dif_HR',
    'Dif_q': 'Dif_q',
    'Dif_y': 'Dif_y',

    # Basin physical properties - ⚠️ CORREGIDOS
    'area_catch': 'area_cuenca',
    'chan_length': 'longitud_total',
    'CN_we': 'CN_ponderado',
    'CN_DT': 'CN_DT',
    'z_min': 'z_min',
    'z_max': 'z_max',

    # Soil properties
    'DT_ks': 'DT_ks',
    'mean_ks': 'mean_ks',
    'mean_thetas': 'mean_thetas',
    'DT_thetas': 'DT_thetas',
    'mean_thetar': 'mean_thetar',
    'DT_thetar': 'DT_thetar',
    'mean_thau': 'mean_thau',
    'DT_thau': 'DT_thau',
    'mean_swcfc': 'mean_swcfc',
    'DT_swcfc': 'DT_swcfc',
    'mean_swcpwp': 'mean_swcpwp',
    'DT_swcpwp': 'DT_swcpwp',

    # Targets
    'ymin': 'ymin',
    'yav': 'yav',
    'ymax': 'ymax',
    'qmin': 'qmin',
    'qav': 'qav',
    'qmax': 'qmax'
}

def map_variables(var_list):
    """Map paper variable names to real column names"""
    return [var_mapping.get(v, v) for v in var_list]

print("✅ Variable mapping defined")


# In[6]:


# Define the 8 models to optimize
models_to_tune = [
    # ymin - Modelo 1110 (Eq 6 promedio)
    {
        'id': 1110,
        'name': 'ymin',
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'humidity': 'Con los valores de humedad promedio',
        'target': 'ymin',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # yav - Modelo 1123 (Eq 7 promedio)
    {
        'id': 1123,
        'name': 'yav',
        'equation': 'Eq 7',
        'type': 'Variables primarias',
        'humidity': 'Con los valores de humedad promedio',
        'target': 'yav',
        'features': ['P', 'Av_qr', 'Av_qs', 'CN_we', 'CN_DT',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # ymax - Modelo 1112 (Eq 6 promedio)
    {
        'id': 1112,
        'name': 'ymax',
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'humidity': 'Con los valores de humedad promedio',
        'target': 'ymax',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # Dif_y - Modelo 1214 (Eq 6 sintéticas)
    {
        'id': 1214,
        'name': 'Dif_y',
        'equation': 'Eq 6',
        'type': 'Variables sintéticas',
        'humidity': 'Con los valores de humedad promedio',
        'target': 'Dif_y',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # qmin - Modelo 1131 (Eq 9a)
    {
        'id': 1131,
        'name': 'qmin',
        'equation': 'Eq 9a',
        'type': 'Variables primarias',
        'humidity': 'N/A',
        'target': 'qmin',
        'features': ['P', 'Dif_qr', 'Dif_qs', 'tav', 'tmax', 'tmin', 'Hrav', 'Hrmax', 'Hrmin',
                     'wv', 'wvmax', 'SolarRad', 'mean_swcfc', 'DT_swcfc', 'mean_swcpwp', 'DT_swcpwp',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # qav - Modelo 1132 (Eq 9a)
    {
        'id': 1132,
        'name': 'qav',
        'equation': 'Eq 9a',
        'type': 'Variables primarias',
        'humidity': 'N/A',
        'target': 'qav',
        'features': ['P', 'Dif_qr', 'Dif_qs', 'tav', 'tmax', 'tmin', 'Hrav', 'Hrmax', 'Hrmin',
                     'wv', 'wvmax', 'SolarRad', 'mean_swcfc', 'DT_swcfc', 'mean_swcpwp', 'DT_swcpwp',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # qmax - Modelo 1133 (Eq 9a)
    {
        'id': 1133,
        'name': 'qmax',
        'equation': 'Eq 9a',
        'type': 'Variables primarias',
        'humidity': 'N/A',
        'target': 'qmax',
        'features': ['P', 'Dif_qr', 'Dif_qs', 'tav', 'tmax', 'tmin', 'Hrav', 'Hrmax', 'Hrmin',
                     'wv', 'wvmax', 'SolarRad', 'mean_swcfc', 'DT_swcfc', 'mean_swcpwp', 'DT_swcpwp',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # Dif_q - Modelo 1221 (Eq 9a sintéticas)
    {
        'id': 1221,
        'name': 'Dif_q',
        'equation': 'Eq 9a',
        'type': 'Variables sintéticas',
        'humidity': 'N/A',
        'target': 'Dif_q',
        'features': ['P', 'Dif_qr', 'Dif_qs', 'Dif_t', 'Dif_HR', 'wv', 'wvmax', 'SolarRad',
                     'mean_swcfc', 'DT_swcfc', 'mean_swcpwp', 'DT_swcpwp',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    }
]

print(f"✅ Defined {len(models_to_tune)} models for hyperparameter tuning")
for model in models_to_tune:
    print(f"   - {model['id']}: {model['name']} ({model['equation']}, {len(model['features'])} features)")


# ## 6. Grid de Hiperparámetros
# 
# Definimos el grid de búsqueda para Random Forest.

# In[7]:


# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Calculate total combinations
total_combinations = 1
for param, values in param_grid.items():
    total_combinations *= len(values)

print("=" * 80)
print("HYPERPARAMETER GRID")
print("=" * 80)
print(f"Total combinations per model: {total_combinations}")
print("\nParameter ranges:")
for param, values in param_grid.items():
    print(f"  - {param}: {values}")


# ## 7. Función de Entrenamiento con Grid Search

# In[8]:


def train_with_grid_search(X, y, param_grid, cv_strategy='random', n_folds=5,
                          use_scaling=True, verbose=0):
    """
    Train Random Forest with Grid Search and hyperparameter tuning.

    Parameters:
    -----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like or Series
        Target variable
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings
    cv_strategy : str
        'random' = KFold with shuffle (días aleatorios)
        'temporal' = TimeSeriesSplit (rangos temporales completos)
    n_folds : int
        Number of folds for cross-validation
    use_scaling : bool
        Whether to scale features using StandardScaler
    verbose : int
        Verbosity level

    Returns:
    --------
    results : dict
        Dictionary with best parameters, metrics, and trained model
    """

    # Drop NaN values
    if isinstance(X, pd.DataFrame):
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx].values
        y_clean = y[valid_idx].values
    else:
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

    if len(X_clean) < 100:
        raise ValueError(f"Insufficient data: {len(X_clean)} samples")

    # Scale features if requested
    if use_scaling:
        scaler = StandardScaler()
        X_clean = scaler.fit_transform(X_clean)
    else:
        scaler = None

    # Define cross-validation strategy
    if cv_strategy == 'random':
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif cv_strategy == 'temporal':
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    # Create base Random Forest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True
    )

    # Fit grid search
    grid_search.fit(X_clean, y_clean)

    # Get best model
    best_model = grid_search.best_estimator_

    # Make predictions with best model (on entire dataset)
    y_pred = best_model.predict(X_clean)

    # Calculate metrics
    metrics = calculate_all_metrics(y_clean, y_pred)

    # Extract feature importances
    feature_importances = best_model.feature_importances_

    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # Convert back to positive RMSE
        'best_model': best_model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'n_samples': len(X_clean),
        'cv_results': grid_search.cv_results_
    }

print("✅ Grid search training function defined")


# ## 8. Preparar Dataset Global

# In[9]:


# Combine all basins into global dataset
all_basins_data = []

for basin_name, df in join_dict.items():
    df_copy = df.copy()
    df_copy['Basin'] = basin_name
    all_basins_data.append(df_copy)

global_df = pd.concat(all_basins_data, ignore_index=True)

# Sort by date for temporal split
if 'Date' in global_df.columns:
    global_df = global_df.sort_values('Date').reset_index(drop=True)

print("=" * 80)
print("GLOBAL DATASET PREPARATION")
print("=" * 80)
print(f"✅ Global dataset created")
print(f"   Total observations: {len(global_df):,}")
print(f"   Basins: {len(join_dict)}")
print(f"   Columns: {len(global_df.columns)}")


# ## 9. Ejecutar Grid Search - MODELO DE PRUEBA (ymin 1110)
# 
# Empezamos con el primer modelo para evaluar tiempo de computación.

# In[ ]:


# Select first model for testing
test_model = models_to_tune[0]  # ymin 1110

print("=" * 80)
print(f"TESTING: Model {test_model['id']} - {test_model['name']}")
print("=" * 80)
print(f"Equation: {test_model['equation']}")
print(f"Type: {test_model['type']}")
print(f"Features: {len(test_model['features'])}")
print(f"Target: {test_model['target']}")

# Map variables
target_col = var_mapping.get(test_model['target'], test_model['target'])
feature_cols = map_variables(test_model['features'])

print(f"\nMapped target: {target_col}")
print(f"Mapped features: {feature_cols}")

# Prepare data
X = global_df[feature_cols]
y = global_df[target_col]

print(f"\nData shape: X={X.shape}, y={y.shape}")
print(f"Missing values: X={X.isna().sum().sum()}, y={y.isna().sum()}")

# IMPORTANT: Measure time for grid search
import time

print("\n" + "=" * 80)
print("STARTING GRID SEARCH (this may take a while...)")
print("=" * 80)

start_time = time.time()

# Run grid search with RANDOM split (días aleatorios)
print("\n🔄 Testing strategy: Random split (días aleatorios)...")
results_random = train_with_grid_search(
    X, y,
    param_grid=param_grid,
    cv_strategy='random',
    n_folds=5,
    use_scaling=True,
    verbose=1
)

elapsed_random = time.time() - start_time
print(f"\n✅ Random split completed in {elapsed_random/60:.2f} minutes")

# Run grid search with TEMPORAL split (rangos completos)
print("\n🔄 Testing strategy: Temporal split (rangos temporales completos)...")
start_temporal = time.time()

results_temporal = train_with_grid_search(
    X, y,
    param_grid=param_grid,
    cv_strategy='temporal',
    n_folds=5,
    use_scaling=True,
    verbose=1
)

elapsed_temporal = time.time() - start_temporal
print(f"\n✅ Temporal split completed in {elapsed_temporal/60:.2f} minutes")

# Compare results
print("\n" + "=" * 80)
print("COMPARISON: RANDOM vs TEMPORAL")
print("=" * 80)

print("\n📊 RANDOM SPLIT (Días aleatorios):")
print(f"   Best RMSE (CV): {results_random['best_score']:.6f}")
print(f"   Best params: {results_random['best_params']}")
print(f"   Metrics (full dataset):")
for metric, value in results_random['metrics'].items():
    print(f"      {metric}: {value:.6f}")

print("\n📊 TEMPORAL SPLIT (Rangos temporales):")
print(f"   Best RMSE (CV): {results_temporal['best_score']:.6f}")
print(f"   Best params: {results_temporal['best_params']}")
print(f"   Metrics (full dataset):")
for metric, value in results_temporal['metrics'].items():
    print(f"      {metric}: {value:.6f}")

print("\n⏱️  COMPUTATION TIME:")
print(f"   Random split: {elapsed_random/60:.2f} min")
print(f"   Temporal split: {elapsed_temporal/60:.2f} min")
print(f"   Total: {(elapsed_random + elapsed_temporal)/60:.2f} min")

# Estimated time for all 8 models
estimated_total = ((elapsed_random + elapsed_temporal) / 2) * 8
print(f"\n📈 ESTIMATED TIME for all 8 models: {estimated_total/60:.2f} min ({estimated_total/3600:.2f} hours)")


# ## 10. Análisis de Importancia de Variables (Modelo de Prueba)

# In[ ]:


# Plot feature importance for test model
def plot_feature_importance(feature_names, importances, title, save_path=None):
    """
    Plot feature importances.
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()

# Create output directory
output_dir = "7-results/feature_importance/"
os.makedirs(output_dir, exist_ok=True)

# Plot for random split
plot_feature_importance(
    feature_cols,
    results_random['feature_importances'],
    f"Feature Importance - {test_model['name']} (Random Split)",
    save_path=os.path.join(output_dir, f"{test_model['id']}_{test_model['name']}_random_importance.png")
)

# Plot for temporal split
plot_feature_importance(
    feature_cols,
    results_temporal['feature_importances'],
    f"Feature Importance - {test_model['name']} (Temporal Split)",
    save_path=os.path.join(output_dir, f"{test_model['id']}_{test_model['name']}_temporal_importance.png")
)


# ## 11. DECISIÓN: Continuar con los 7 modelos restantes?
# 
# **Evalúa el tiempo estimado arriba antes de continuar.**
# 
# Si el tiempo es razonable (<2 horas por modelo), ejecuta la celda siguiente.

# In[ ]:


# CONFIGURACIÓN: Cambiar a True para ejecutar todos los modelos
RUN_ALL_MODELS = False  # ← CAMBIAR A True PARA EJECUTAR

if not RUN_ALL_MODELS:
    print("⚠️  RUN_ALL_MODELS = False")
    print("   Cambia a True en la celda anterior para ejecutar los 7 modelos restantes.")
else:
    print("✅ RUN_ALL_MODELS = True. Ejecutando todos los modelos...")


# ## 12. Ejecutar Grid Search para los 7 modelos restantes

# In[ ]:


if RUN_ALL_MODELS:
    # Storage for all results
    all_results = []

    # Loop through remaining models (skip first one, already done)
    for model_spec in tqdm(models_to_tune[1:], desc="Training models"):
        print("\n" + "=" * 80)
        print(f"Model {model_spec['id']} - {model_spec['name']}")
        print("=" * 80)

        # Map variables
        target_col = var_mapping.get(model_spec['target'], model_spec['target'])
        feature_cols = map_variables(model_spec['features'])

        # Prepare data
        X = global_df[feature_cols]
        y = global_df[target_col]

        # Train with both strategies
        print("\n🔄 Training with random split...")
        results_random = train_with_grid_search(
            X, y,
            param_grid=param_grid,
            cv_strategy='random',
            n_folds=5,
            use_scaling=True,
            verbose=0
        )

        print("\n🔄 Training with temporal split...")
        results_temporal = train_with_grid_search(
            X, y,
            param_grid=param_grid,
            cv_strategy='temporal',
            n_folds=5,
            use_scaling=True,
            verbose=0
        )

        # Store results
        all_results.append({
            'model_id': model_spec['id'],
            'model_name': model_spec['name'],
            'equation': model_spec['equation'],
            'target': model_spec['target'],
            'n_features': len(feature_cols),
            'random': results_random,
            'temporal': results_temporal,
            'feature_cols': feature_cols
        })

        # Plot feature importance
        plot_feature_importance(
            feature_cols,
            results_random['feature_importances'],
            f"Feature Importance - {model_spec['name']} (Random Split)",
            save_path=os.path.join(output_dir, f"{model_spec['id']}_{model_spec['name']}_random_importance.png")
        )

        print(f"\n✅ Model {model_spec['id']} completed")

    print("\n" + "=" * 80)
    print("ALL MODELS COMPLETED")
    print("=" * 80)
else:
    print("⏭️  Skipping execution of remaining models.")


# ## 13. Consolidar Resultados en CSV

# In[ ]:


if RUN_ALL_MODELS:
    # Create summary DataFrame
    summary_data = []

    for result in all_results:
        # Random split results
        summary_data.append({
            'Model_ID': result['model_id'],
            'Model_Name': result['model_name'],
            'Equation': result['equation'],
            'Target': result['target'],
            'N_Features': result['n_features'],
            'Split_Strategy': 'Random',
            'n_estimators': result['random']['best_params']['n_estimators'],
            'max_depth': result['random']['best_params']['max_depth'],
            'min_samples_split': result['random']['best_params']['min_samples_split'],
            'min_samples_leaf': result['random']['best_params']['min_samples_leaf'],
            'max_features': result['random']['best_params']['max_features'],
            'bootstrap': result['random']['best_params']['bootstrap'],
            'CV_RMSE': result['random']['best_score'],
            **{f"Test_{k}": v for k, v in result['random']['metrics'].items()},
            'n_samples': result['random']['n_samples']
        })

        # Temporal split results
        summary_data.append({
            'Model_ID': result['model_id'],
            'Model_Name': result['model_name'],
            'Equation': result['equation'],
            'Target': result['target'],
            'N_Features': result['n_features'],
            'Split_Strategy': 'Temporal',
            'n_estimators': result['temporal']['best_params']['n_estimators'],
            'max_depth': result['temporal']['best_params']['max_depth'],
            'min_samples_split': result['temporal']['best_params']['min_samples_split'],
            'min_samples_leaf': result['temporal']['best_params']['min_samples_leaf'],
            'max_features': result['temporal']['best_params']['max_features'],
            'bootstrap': result['temporal']['best_params']['bootstrap'],
            'CV_RMSE': result['temporal']['best_score'],
            **{f"Test_{k}": v for k, v in result['temporal']['metrics'].items()},
            'n_samples': result['temporal']['n_samples']
        })

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #csv_path = f"../../data/model_results/tuned_rf_models_{timestamp}.csv"
    csv_path = f"./tuned_rf_models_{timestamp}.csv" # SERGIO: Cambia este path si quieres guardarlo en otra carpeta
    summary_df.to_csv(csv_path, index=False)

    print("=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"✅ Saved to: {csv_path}")
    print(f"   Rows: {len(summary_df)}")
    print(f"   Columns: {len(summary_df.columns)}")

    # Display summary
    print("\n📊 SUMMARY:")
    display(summary_df[['Model_ID', 'Model_Name', 'Split_Strategy', 'CV_RMSE', 'Test_NSE', 'Test_KGE']].head(20))
else:
    print("⏭️  Skipping results consolidation.")


# ## 14. Conclusiones y Próximos Pasos
# 
# Este notebook ha completado:
# 
# 1. ✅ **Hyperparameter tuning** para 8 modelos Random Forest
# 2. ✅ **Comparación** de estrategias de partición (días aleatorios vs rangos temporales)
# 3. ✅ **Tipificación** de variables con StandardScaler
# 4. ✅ **Análisis de importancia** de variables
# 
# **Próximos pasos:**
# - Ver notebook `8-rf-predictions-and-analysis.ipynb` para:
#   - Generar predicciones por cuenca
#   - Crear CSVs de predicciones (filas=cuencas, columnas=fechas)
#   - Calcular matrices de confusión
#   - Generar visualizaciones
# 
# **Reporte técnico:**
# - Ver `7-results/HYPERPARAMETER_TUNING_REPORT.md` para documentación completa
