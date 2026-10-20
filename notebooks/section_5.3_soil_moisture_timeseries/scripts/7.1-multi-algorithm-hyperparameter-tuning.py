#!/usr/bin/env python
# coding: utf-8

# # Notebook 7: Multi-Algorithm Hyperparameter Tuning
#
# ## Objetivo
#
# Este notebook realiza **ajuste fino de hiperparámetros** para los 8 modelos 
# que han resultado ser los mejores según análisis del director de tesis.
#
# **Modelos a optimizar:**
# 1. **ymin** - Random Forest (Eq 6) - 15 features
# 2. **yav** - Random Forest (Eq 6) - 15 features
# 3. **ymax** - Random Forest (Eq 6) - 15 features
# 4. **qmin** - Random Forest (Eq 6) - 15 features
# 5. **qav** - Random Forest (Eq 6) - 15 features
# 6. **qmax** - Random Forest (Eq 6) - 15 features
# 7. **Dif_q** - SVM_RBF (Eq 7) - 9 features
# 8. **Dif_y** - XGBoost (Eq 6) - 15 features
#
# **Algoritmos:**
# - Random Forest: 6 modelos (tree-based ensemble)
# - XGBoost: 1 modelo (gradient boosting)
# - SVM_RBF: 1 modelo (support vector machine con kernel RBF)
#
# **Ecuaciones:**
# - Eq 6: Variables primarias con propiedades de suelo detalladas (15 features)
# - Eq 7: Variables primarias con CN ponderado (9 features, más simple)
#
# **Metodología:**
# - Grid Search con cross-validation
# - Comparación: días aleatorios (KFold) vs rangos temporales (TimeSeriesSplit)
# - Tipificación de variables (StandardScaler) - CRÍTICO para SVM
# - Split train/test 80/20 ANTES del Grid Search (previene overfitting)
# - Análisis de importancia de variables (nativa para RF/XGBoost, permutation para SVM)
#
# **Autor:** Víctor Galán
# **Fecha:** 2026-03-29

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
from sklearn.svm import SVR  # Para SVM_RBF
from sklearn.inspection import permutation_importance  # Para feature importance de SVM

# XGBoost
import xgboost as xgb

# Scipy for statistical calculations
from scipy import stats

warnings.filterwarnings('ignore')

print("=" * 80)
print("NOTEBOOK 7: MULTI-ALGORITHM HYPERPARAMETER TUNING")
print("=" * 80)
print(f"✅ Imports loaded successfully")
print(f"   numpy version: {np.__version__}")
print(f"   pandas version: {pd.__version__}")
print(f"   sklearn version: {__import__('sklearn').__version__}")
print(f"   xgboost version: {xgb.__version__}")


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


# Define the 8 models to optimize (UPDATED según director de tesis)
models_to_tune = [
    # ymin - RF + Eq 6 (sin cambios)
    {
        'id': 1110,
        'name': 'ymin',
        'algorithm': 'RF',
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'target': 'ymin',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # yav - RF + Eq 6 (CAMBIO: antes era Eq 7)
    {
        'id': 1123,
        'name': 'yav',
        'algorithm': 'RF',
        'equation': 'Eq 6',  # ← CAMBIO: antes Eq 7
        'type': 'Variables primarias',
        'target': 'yav',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO
    },
    # ymax - RF + Eq 6 (sin cambios)
    {
        'id': 1112,
        'name': 'ymax',
        'algorithm': 'RF',
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'target': 'ymax',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # Dif_y - XGBoost + Eq 6 (CAMBIO: antes era RF)
    {
        'id': 1214,
        'name': 'Dif_y',
        'algorithm': 'XGBoost',  # ← CAMBIO: antes RF
        'equation': 'Eq 6',
        'type': 'Variables sintéticas',
        'target': 'Dif_y',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # qmin - RF + Eq 6 (CAMBIO: antes Eq 9a)
    {
        'id': 1131,
        'name': 'qmin',
        'algorithm': 'RF',
        'equation': 'Eq 6',  # ← CAMBIO: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qmin',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO
    },
    # qav - RF + Eq 6 (CAMBIO: antes Eq 9a)
    {
        'id': 1132,
        'name': 'qav',
        'algorithm': 'RF',
        'equation': 'Eq 6',  # ← CAMBIO: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qav',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO
    },
    # qmax - RF + Eq 6 (CAMBIO: antes Eq 9a)
    {
        'id': 1133,
        'name': 'qmax',
        'algorithm': 'RF',
        'equation': 'Eq 6',  # ← CAMBIO: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qmax',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO
    },
    # Dif_q - SVM_RBF + Eq 7 (CAMBIO: antes RF + Eq 9a)
    {
        'id': 1221,
        'name': 'Dif_q',
        'algorithm': 'SVM_RBF',  # ← CAMBIO: antes RF
        'equation': 'Eq 7',  # ← CAMBIO: antes Eq 9a
        'type': 'Variables sintéticas',
        'target': 'Dif_q',
        'features': ['P', 'Av_qr', 'Av_qs', 'CN_we', 'CN_DT',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO
    }
]

print(f"✅ Defined {len(models_to_tune)} models for hyperparameter tuning")
for model in models_to_tune:
    print(f"   - {model['id']}: {model['name']} ({model['algorithm']}, {model['equation']}, {len(model['features'])} features)")


# ## 6. Grid de Hiperparámetros
#
# Definimos los grids de búsqueda para Random Forest, XGBoost y SVM_RBF.

# In[7]:


# Hyperparameter grid for Random Forest
# ✅ OPTIMIZADO: Reducido de 648 a 96 combinaciones para acelerar (6x más rápido)
# Mantiene exploración razonable de parámetros clave
param_grid_rf = {
    'n_estimators': [100, 200, 300],      # Mantener 3 (importante)
    'max_depth': [10, None],              # Reducir a 2 (con límite y sin límite)
    'min_samples_split': [2, 10],         # Reducir a 2 (bajo y alto)
    'min_samples_leaf': [1, 4],           # Reducir a 2 (bajo y alto)
    'max_features': ['sqrt', None],       # Reducir a 2 (sqrt y todas las features)
    'bootstrap': [True, False]            # Mantener 2
}
# Total: 3×2×2×2×2×2 = 96 combinaciones

# Hyperparameter grid for XGBoost
# Exploración equilibrada de parámetros clave
param_grid_xgboost = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
# Total: 3×3×2×2×2 = 72 combinaciones

# Hyperparameter grid for SVM_RBF
# Basado en valores del notebook 6 (C=10.0, epsilon=0.1, gamma='scale')
# Exploración alrededor de estos valores
param_grid_svm = {
    'C': [1.0, 10.0, 100.0],           # Parámetro de regularización
    'epsilon': [0.01, 0.1, 0.2],       # Margen de tolerancia
    'gamma': ['scale', 'auto', 0.01, 0.1]  # Coeficiente del kernel RBF
}
# Total: 3 × 3 × 4 = 36 combinaciones

# Calculate total combinations per algorithm
total_rf = np.prod([len(v) for v in param_grid_rf.values()])
total_xgb = np.prod([len(v) for v in param_grid_xgboost.values()])
total_svm = np.prod([len(v) for v in param_grid_svm.values()])

print("=" * 80)
print("HYPERPARAMETER GRIDS")
print("=" * 80)
print(f"\n🌲 Random Forest: {total_rf} combinations")
for param, values in param_grid_rf.items():
    print(f"  - {param}: {values}")

print(f"\n🚀 XGBoost: {total_xgb} combinations")
for param, values in param_grid_xgboost.items():
    print(f"  - {param}: {values}")

print(f"\n🎯 SVM_RBF: {total_svm} combinations")
for param, values in param_grid_svm.items():
    print(f"  - {param}: {values}")


# ## 7. Función de Entrenamiento con Grid Search (Multi-Algorithm)

# In[8]:


def train_model_with_grid_search(X, y, algorithm='RF', param_grid=None, 
                                  cv_strategy='random', n_folds=5, 
                                  use_scaling=True, verbose=0):
    """
    Entrena modelo (RF, XGBoost o SVM_RBF) con Grid Search.
    
    Parámetros:
    -----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like or Series
        Target variable
    algorithm : str
        'RF' para Random Forest, 'XGBoost' para XGBoost, 'SVM_RBF' para SVM con kernel RBF
    param_grid : dict
        Grid específico del algoritmo
    cv_strategy : str
        'random' = KFold con shuffle (días aleatorios)
        'temporal' = TimeSeriesSplit (rangos temporales)
    n_folds : int
        Número de folds para cross-validation
    use_scaling : bool
        Si True, aplica StandardScaler
        IMPORTANTE: SVM_RBF REQUIERE scaling (True)
        RF y XGBoost no lo requieren pero no afecta
    verbose : int
        Nivel de verbosidad
    
    Returns:
    --------
    results : dict
        Diccionario con mejores parámetros, métricas y modelo entrenado
    """
    
    # 1. VALIDACIÓN DE DATOS
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

    # 2. SPLIT TRAIN/TEST (previene overfitting)
    # 80% train, 20% test - Grid Search solo verá los datos de TRAIN
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean,
        test_size=0.2,
        random_state=42
    )

    # 3. SCALING (CRÍTICO para SVM)
    if use_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # ✅ Fit solo en TRAIN
        X_test = scaler.transform(X_test)        # ✅ Transform (no fit!) en TEST
    else:
        scaler = None
        if algorithm == 'SVM_RBF':
            warnings.warn("⚠️ SVM_RBF requires scaling! Setting use_scaling=True is strongly recommended.")

    # 4. CROSS-VALIDATION STRATEGY
    if cv_strategy == 'random':
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif cv_strategy == 'temporal':
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    # 5. SELECCIÓN DE MODELO BASE según algoritmo
    if algorithm == 'RF':
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    elif algorithm == 'XGBoost':
        base_model = xgb.XGBRegressor(
            random_state=42, 
            n_jobs=-1,
            tree_method='auto'  # 'gpu_hist' si hay GPU
        )
    
    elif algorithm == 'SVM_RBF':
        base_model = SVR(kernel='rbf')  # kernel fijo, C/epsilon/gamma en grid
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'RF', 'XGBoost', or 'SVM_RBF'")

    # 6. GRID SEARCH
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # 7. PREDICCIONES Y MÉTRICAS en TEST set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = calculate_all_metrics(y_test, y_pred)

    # 8. FEATURE IMPORTANCES
    # RF y XGBoost tienen .feature_importances_
    # SVM NO tiene, usar permutation importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    else:
        # SVM: calcular permutation importance
        perm_importance = permutation_importance(
            best_model, X_test, y_test, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        feature_importances = perm_importance.importances_mean

    # 9. RETURN
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

print("✅ Multi-algorithm grid search training function defined")


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


# ## 9. Función de Visualización de Feature Importance

# In[10]:


def plot_feature_importance(feature_names, importances, title, save_path=None):
    """
    Plot feature importances.
    Funciona para RF, XGBoost (atributo nativo) y SVM (permutation importance)
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
        print(f"   ✅ Saved to: {save_path}")

    plt.show()

# Create output directory
output_dir = "7-results/feature_importance/"
os.makedirs(output_dir, exist_ok=True)

print("✅ Feature importance plotting function defined")


# ## 10. Ejecutar Grid Search para TODOS LOS MODELOS
#
# Entrenamiento directo de los 8 modelos sin modo de prueba.

# In[11]:


# Storage for all results
all_results = []

print("=" * 80)
print("STARTING FULL TRAINING - 8 MODELS")
print("=" * 80)
print(f"Models: 6 RF + 1 XGBoost + 1 SVM_RBF")
print(f"Strategies: Random + Temporal per model")
print(f"Estimated time: 2-3 hours (depends on hardware)")
print("=" * 80)

import time
total_start = time.time()

for model_spec in tqdm(models_to_tune, desc="Training models"):
    print("\n" + "=" * 80)
    print(f"Model {model_spec['id']} - {model_spec['name']} ({model_spec['algorithm']})")
    print("=" * 80)

    # Map variables
    target_col = var_mapping.get(model_spec['target'], model_spec['target'])
    feature_cols = map_variables(model_spec['features'])

    # Prepare data
    X = global_df[feature_cols]
    y = global_df[target_col]

    # Seleccionar grid según algoritmo
    if model_spec['algorithm'] == 'RF':
        param_grid = param_grid_rf
    elif model_spec['algorithm'] == 'XGBoost':
        param_grid = param_grid_xgboost
    elif model_spec['algorithm'] == 'SVM_RBF':
        param_grid = param_grid_svm

    print(f"   Algorithm: {model_spec['algorithm']}")
    print(f"   Equation: {model_spec['equation']}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

    # Train with RANDOM split
    print(f"\n🔄 Training {model_spec['algorithm']} with random split...")
    start_random = time.time()
    results_random = train_model_with_grid_search(
        X, y,
        algorithm=model_spec['algorithm'],
        param_grid=param_grid,
        cv_strategy='random',
        n_folds=5,
        use_scaling=True,  # CRÍTICO: True para SVM
        verbose=0
    )
    elapsed_random = time.time() - start_random
    print(f"   ✅ Random split completed in {elapsed_random/60:.2f} min")
    print(f"      Best CV RMSE: {results_random['best_score']:.6f}")
    print(f"      Test NSE: {results_random['metrics']['NSE']:.4f}")
    print(f"      Test KGE: {results_random['metrics']['KGE']:.4f}")

    # Train with TEMPORAL split
    print(f"\n🔄 Training {model_spec['algorithm']} with temporal split...")
    start_temporal = time.time()
    results_temporal = train_model_with_grid_search(
        X, y,
        algorithm=model_spec['algorithm'],
        param_grid=param_grid,
        cv_strategy='temporal',
        n_folds=5,
        use_scaling=True,  # CRÍTICO: True para SVM
        verbose=0
    )
    elapsed_temporal = time.time() - start_temporal
    print(f"   ✅ Temporal split completed in {elapsed_temporal/60:.2f} min")
    print(f"      Best CV RMSE: {results_temporal['best_score']:.6f}")
    print(f"      Test NSE: {results_temporal['metrics']['NSE']:.4f}")
    print(f"      Test KGE: {results_temporal['metrics']['KGE']:.4f}")

    # Store results
    all_results.append({
        'model_id': model_spec['id'],
        'model_name': model_spec['name'],
        'algorithm': model_spec['algorithm'],
        'equation': model_spec['equation'],
        'target': model_spec['target'],
        'n_features': len(feature_cols),
        'random': results_random,
        'temporal': results_temporal,
        'feature_cols': feature_cols
    })

    # Plot feature importance
    print(f"\n📊 Generating feature importance plots...")
    plot_feature_importance(
        feature_cols,
        results_random['feature_importances'],
        f"Feature Importance - {model_spec['name']} ({model_spec['algorithm']}, Random Split)",
        save_path=os.path.join(output_dir, f"{model_spec['id']}_{model_spec['name']}_{model_spec['algorithm']}_random_importance.png")
    )

    print(f"\n✅ Model {model_spec['id']} completed (total: {(elapsed_random + elapsed_temporal)/60:.2f} min)")

total_elapsed = time.time() - total_start

print("\n" + "=" * 80)
print("ALL MODELS COMPLETED")
print("=" * 80)
print(f"⏱️  Total time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)")
print(f"📊 Models trained: {len(all_results)}")


# ## 11. Consolidar Resultados en CSV

# In[12]:


# Create summary DataFrame
summary_data = []

for result in all_results:
    # RANDOM split results
    random_row = {
        'Model_ID': result['model_id'],
        'Model_Name': result['model_name'],
        'Algorithm': result['algorithm'],
        'Equation': result['equation'],
        'Target': result['target'],
        'N_Features': result['n_features'],
        'Split_Strategy': 'Random',
        'CV_RMSE': result['random']['best_score'],
        **{f"Test_{k}": v for k, v in result['random']['metrics'].items()},
        'n_samples': result['random']['n_samples']
    }
    
    # Agregar hiperparámetros como columnas dinámicas
    for param_name, param_value in result['random']['best_params'].items():
        if param_value is None:
            param_value = 'None'
        random_row[f'HP_{param_name}'] = param_value
    
    summary_data.append(random_row)

    # TEMPORAL split results
    temporal_row = {
        'Model_ID': result['model_id'],
        'Model_Name': result['model_name'],
        'Algorithm': result['algorithm'],
        'Equation': result['equation'],
        'Target': result['target'],
        'N_Features': result['n_features'],
        'Split_Strategy': 'Temporal',
        'CV_RMSE': result['temporal']['best_score'],
        **{f"Test_{k}": v for k, v in result['temporal']['metrics'].items()},
        'n_samples': result['temporal']['n_samples']
    }
    
    for param_name, param_value in result['temporal']['best_params'].items():
        if param_value is None:
            param_value = 'None'
        temporal_row[f'HP_{param_name}'] = param_value
    
    summary_data.append(temporal_row)

# Create DataFrame
summary_df = pd.DataFrame(summary_data)

# Reordenar columnas para mejor lectura
metadata_cols = ['Model_ID', 'Model_Name', 'Algorithm', 'Equation', 
                 'Target', 'N_Features', 'Split_Strategy']
hp_cols = sorted([c for c in summary_df.columns if c.startswith('HP_')])
metric_cols = ['CV_RMSE'] + sorted([c for c in summary_df.columns if c.startswith('Test_')])
other_cols = [c for c in summary_df.columns if c not in metadata_cols + hp_cols + metric_cols]

summary_df = summary_df[metadata_cols + hp_cols + metric_cols + other_cols]

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"./tuned_models_rf_xgb_svm_{timestamp}.csv"  # SERGIO: Cambia este path si quieres guardarlo en otra carpeta
summary_df.to_csv(csv_path, index=False)

print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"✅ Saved to: {csv_path}")
print(f"   Rows: {len(summary_df)}")
print(f"   Columns: {len(summary_df.columns)}")

# Display summary
display_cols = ['Model_ID', 'Model_Name', 'Algorithm', 'Split_Strategy', 
                'CV_RMSE', 'Test_NSE', 'Test_KGE']
print("\n📊 SUMMARY:")
print(summary_df[display_cols].to_string())


# ## 12. Análisis Comparativo por Algoritmo

# In[13]:


print("=" * 80)
print("COMPARATIVE ANALYSIS BY ALGORITHM")
print("=" * 80)

# Agrupar por algoritmo y estrategia
for algo in ['RF', 'XGBoost', 'SVM_RBF']:
    algo_results = summary_df[summary_df['Algorithm'] == algo]
    if len(algo_results) > 0:
        print(f"\n🔹 {algo}:")
        print(f"   Models: {algo_results['Model_Name'].unique().tolist()}")
        print(f"   Mean CV RMSE: {algo_results['CV_RMSE'].mean():.6f}")
        print(f"   Mean Test NSE: {algo_results['Test_NSE'].mean():.4f}")
        print(f"   Mean Test KGE: {algo_results['Test_KGE'].mean():.4f}")

# Comparación Random vs Temporal
print("\n" + "=" * 80)
print("COMPARISON: RANDOM vs TEMPORAL SPLIT")
print("=" * 80)

for strategy in ['Random', 'Temporal']:
    strat_results = summary_df[summary_df['Split_Strategy'] == strategy]
    print(f"\n🔹 {strategy} Split:")
    print(f"   Mean CV RMSE: {strat_results['CV_RMSE'].mean():.6f}")
    print(f"   Mean Test NSE: {strat_results['Test_NSE'].mean():.4f}")
    print(f"   Mean Test KGE: {strat_results['Test_KGE'].mean():.4f}")


# ## 13. Conclusiones y Próximos Pasos
#
# Este notebook ha completado:
#
# 1. ✅ **Hyperparameter tuning** para 8 modelos (6 RF + 1 XGBoost + 1 SVM_RBF)
# 2. ✅ **Comparación** de estrategias de partición (random vs temporal)
# 3. ✅ **Comparación entre algoritmos**:
#    - RF: Robusto, rápido, buena interpretabilidad
#    - XGBoost: Potencial mejor performance, más parámetros
#    - SVM_RBF: Bueno para relaciones no lineales, requiere scaling
# 4. ✅ **Tipificación** de variables con StandardScaler (crítico para SVM)
# 5. ✅ **Feature importance** para todos los algoritmos:
#    - RF y XGBoost: Importancia nativa (reducción impureza/gain)
#    - SVM: Permutation importance (más costosa pero interpretable)
#
# **Resultados destacados:**
# - Mejores hiperparámetros por modelo y estrategia de CV
# - Métricas hidrológicas (NSE, KGE, PBIAS, RMSE, MAE, MAPE)
# - Comparativa de performance entre algoritmos
#
# **Observaciones clave:**
# - Comparar RF vs XGBoost para Dif_y (mismo Eq 6)
# - Analizar si SVM_RBF con Eq 7 (menos features) compite con RF
# - Evaluar trade-off tiempo de entrenamiento vs performance
#
# **Próximos pasos:**
# - Notebook `8-rf-predictions-and-analysis.ipynb` para:
#   - Generar predicciones por cuenca con modelos optimizados
#   - Crear CSVs de predicciones (filas=cuencas, columnas=fechas)
#   - Análisis de residuos por algoritmo
#   - Visualizaciones comparativas
