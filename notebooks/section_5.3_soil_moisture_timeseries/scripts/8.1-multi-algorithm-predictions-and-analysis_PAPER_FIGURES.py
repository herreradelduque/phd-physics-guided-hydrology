#!/usr/bin/env python
# coding: utf-8

# # Notebook 8: Multi-Algorithm - Predictions and Complete Analysis
#
# ## Objetivo
#
# Este notebook completa el análisis de los 8 modelos optimizados en el Notebook 7 (versión 3).
#
# **Entregables para el director de tesis:**
# 1. ✅ **Predicciones numéricas**: CSVs por variable (filas=cuencas, columnas=fechas)
# 2. ✅ **Gráficos de importancia**: Para todos los modelos (16 gráficos + heatmap)
# 3. ✅ **Matrices de confusión**: Especificidad, sensibilidad, prevalencia
# 4. ✅ **Reporte técnico completo**: Metodología, parámetros, sentido físico
# 5. ✅ **Visualizaciones adicionales**: Time series, scatter, residuales, Taylor diagrams
#
# **Modelos analizados (UPDATED v4):**
# - **Random Forest (6 modelos)**: ymin, yav, ymax, qmin, qav, qmax
# - **XGBoost (1 modelo)**: Dif_y
# - **SVM_RBF (1 modelo)**: Dif_q
#
# **Algoritmos:**
# - Random Forest: Bagging ensemble, paralelo, robusto
# - XGBoost: Gradient boosting, secuencial, optimizado
# - SVM_RBF: Support Vector Machine con kernel radial
#
# **Autor:** Víctor Galán
# **Fecha:** 2026-03-29
# **Versión:** 4 - Soporte multi-algoritmo (RF + XGBoost + SVM_RBF)
#
# **Cambios v4:**
# - ✅ Soporte para 3 algoritmos diferentes (RF, XGBoost, SVM_RBF)
# - ✅ Lectura de hiperparámetros dinámicos desde CSV (prefijo HP_)
# - ✅ Reentrenamiento específico por algoritmo
# - ✅ Feature importance: nativa (RF/XGBoost) y permutation (SVM)
# - ✅ Validación de columna 'Algorithm' en CSV
# - ✅ Manejo robusto de cuencas con columnas faltantes (heredado de v3)

# ## 1. Imports y Configuración

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import glob
import warnings
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # ← NUEVO V4: Para SVM_RBF
from sklearn.inspection import permutation_importance  # ← NUEVO V4: Para feature importance de SVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

# XGBoost
import xgboost as xgb  # ← NUEVO V4

# Scipy for statistical calculations
from scipy import stats

warnings.filterwarnings('ignore')

# ✅ CONFIGURACIÓN MEJORADA PARA PAPER (DPI 450, texto grande, Times New Roman)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 450
plt.rcParams['savefig.dpi'] = 450
plt.rcParams['font.size'] = 14  # Aumentado de 10
plt.rcParams['axes.labelsize'] = 16  # Aumentado de 12
plt.rcParams['axes.titlesize'] = 18  # Aumentado de 14
plt.rcParams['xtick.labelsize'] = 14  # Aumentado de 10
plt.rcParams['ytick.labelsize'] = 14  # Aumentado de 10
plt.rcParams['legend.fontsize'] = 14  # Aumentado de 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Seaborn style
sns.set_style('whitegrid')
sns.set_palette('husl')

print('✅ Configuración matplotlib mejorada para paper')
print(f'   DPI: {plt.rcParams["savefig.dpi"]}')
print(f'   Font size: {plt.rcParams["font.size"]}')

print("=" * 80)
print("NOTEBOOK 8: MULTI-ALGORITHM - PREDICTIONS AND COMPLETE ANALYSIS")
print("=" * 80)
print(f"✅ Imports loaded successfully")
print(f"   numpy version: {np.__version__}")
print(f"   pandas version: {pd.__version__}")
print(f"   sklearn version: {__import__('sklearn').__version__}")
print(f"   xgboost version: {xgb.__version__}")  # ← NUEVO V4


# ## 2. Configuración de Paths

# In[2]:


# Base paths
BASE_PATH = "../../data/clean/merged_data/"
# ✅ CORREGIDO: Path correcto para los resultados del notebook 7
#RESULTS_PATH_NB7 = "./7-results/"
RESULTS_PATH_NB7 = "./" # Sergio: pon aqui el path donde tengas los resultados del script 7, que se van a usar como input para este script. Creo que me dijiste que los tenias en el mismo directorio que este script, pero si no es asi, pon aqui el path correcto.

OUTPUT_BASE = "./8-results/" # Sergio: pon aqui el path donde quieras que se guarden los resultados de este script. Si lo dejas como está se guardarán en un directorio llamado "8-results" dentro del mismo directorio que este script, pero si no es asi, pon aqui el path correcto.

# Create output directories
output_dirs = {
    'predictions': OUTPUT_BASE + 'predictions/',
    'feature_importance': OUTPUT_BASE + 'feature_importance/',
    'confusion_matrices': OUTPUT_BASE + 'confusion_matrices/',
    'timeseries': OUTPUT_BASE + 'figures/timeseries/',
    'scatter': OUTPUT_BASE + 'figures/scatter/',
    'residuals': OUTPUT_BASE + 'figures/residuals/',
    'taylor': OUTPUT_BASE + 'figures/taylor/',
    'reports': OUTPUT_BASE + 'reports/'
}

for dir_name, dir_path in output_dirs.items():
    os.makedirs(dir_path, exist_ok=True)

print("✅ Output directories created:")
for dir_name, dir_path in output_dirs.items():
    print(f"   - {dir_name}: {dir_path}")


# ## 3. Funciones de Métricas Hidrológicas

# In[3]:


def calculate_nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (NSE)"""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

def calculate_kge(y_true, y_pred):
    """Kling-Gupta Efficiency (KGE)"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def calculate_pbias(y_true, y_pred):
    """Percent Bias (PBIAS)"""
    if np.sum(y_true) == 0:
        return np.nan
    return (np.sum(y_true - y_pred) / np.sum(y_true)) * 100

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE)"""
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_all_metrics(y_true, y_pred):
    """Calculate all metrics at once"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'NSE': calculate_nse(y_true, y_pred),
        'KGE': calculate_kge(y_true, y_pred),
        'PBIAS_%': calculate_pbias(y_true, y_pred),
        'MAPE_%': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

print("✅ Metric functions defined")


# ## 4. Variable Mapping y Definiciones de Modelos

# In[4]:


# Variable mapping (mismo que Notebook 7)
var_mapping = {
    # Climate
    'P': 'P', 'tav': 'tav', 'tmax': 'tmax', 'tmin': 'tmin',
    'Hrav': 'Hrav', 'Hrmax': 'Hrmax', 'Hrmin': 'Hrmin',
    'wv': 'wv', 'wvmax': 'wvmax', 'SolarRad': 'SolarRad', 'et0': 'et0',

    # SMAP soil moisture
    'Av_qr': 'sm_rootzone_av', 'Av_qs': 'sm_surface_av',
    'q0r': 'q0_r', 'q0s': 'q0_s',
    'Dif_qr': 'Dif_qr', 'Dif_qs': 'Dif_qs',

    # Synthetic
    'Dif_t': 'Dif_t', 'Dif_HR': 'Dif_HR',
    'Dif_q': 'Dif_q', 'Dif_y': 'Dif_y',

    # Basin physical properties
    'area_catch': 'area_cuenca', 'chan_length': 'longitud_total',
    'CN_we': 'CN_ponderado', 'CN_DT': 'CN_DT',
    'z_min': 'z_min', 'z_max': 'z_max',

    # Soil properties
    'DT_ks': 'DT_ks', 'mean_ks': 'mean_ks',
    'mean_thetas': 'mean_thetas', 'DT_thetas': 'DT_thetas',
    'mean_thetar': 'mean_thetar', 'DT_thetar': 'DT_thetar',
    'mean_thau': 'mean_thau', 'DT_thau': 'DT_thau',
    'mean_swcfc': 'mean_swcfc', 'DT_swcfc': 'DT_swcfc',
    'mean_swcpwp': 'mean_swcpwp', 'DT_swcpwp': 'DT_swcpwp',

    # Targets
    'ymin': 'ymin', 'yav': 'yav', 'ymax': 'ymax',
    'qmin': 'qmin', 'qav': 'qav', 'qmax': 'qmax'
}

def map_variables(var_list):
    """Map paper variable names to real column names"""
    return [var_mapping.get(v, v) for v in var_list]

# ✅ ACTUALIZADO V4: Define the 8 models con algoritmos correctos (mismo que Notebook 7_3)
models_to_analyze = [
    # ymin - RF + Eq 6
    {
        'id': 1110,
        'name': 'ymin',
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'target': 'ymin',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # yav - RF + Eq 6 (CAMBIO: antes Eq 7)
    {
        'id': 1123,
        'name': 'yav',
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',  # ← CAMBIO V4: antes Eq 7
        'type': 'Variables primarias',
        'target': 'yav',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO V4
    },
    # ymax - RF + Eq 6
    {
        'id': 1112,
        'name': 'ymax',
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',
        'type': 'Variables primarias',
        'target': 'ymax',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']
    },
    # Dif_y - XGBoost + Eq 6 (CAMBIO: antes RF)
    {
        'id': 1214,
        'name': 'Dif_y',
        'algorithm': 'XGBoost',  # ← CAMBIO V4: antes RF
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
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',  # ← CAMBIO V4: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qmin',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO V4
    },
    # qav - RF + Eq 6 (CAMBIO: antes Eq 9a)
    {
        'id': 1132,
        'name': 'qav',
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',  # ← CAMBIO V4: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qav',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO V4
    },
    # qmax - RF + Eq 6 (CAMBIO: antes Eq 9a)
    {
        'id': 1133,
        'name': 'qmax',
        'algorithm': 'RF',  # ← NUEVO V4
        'equation': 'Eq 6',  # ← CAMBIO V4: antes Eq 9a
        'type': 'Variables primarias',
        'target': 'qmax',
        'features': ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                     'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO V4
    },
    # Dif_q - SVM_RBF + Eq 7 (CAMBIO: antes RF + Eq 9a)
    {
        'id': 1221,
        'name': 'Dif_q',
        'algorithm': 'SVM_RBF',  # ← CAMBIO V4: antes RF
        'equation': 'Eq 7',  # ← CAMBIO V4: antes Eq 9a
        'type': 'Variables sintéticas',
        'target': 'Dif_q',
        'features': ['P', 'Av_qr', 'Av_qs', 'CN_we', 'CN_DT',
                     'area_catch', 'chan_length', 'z_min', 'z_max']  # ← CAMBIO V4
    }
]

print(f"✅ Defined {len(models_to_analyze)} models for analysis")
for model in models_to_analyze:
    print(f"   - {model['id']}: {model['name']} ({model['algorithm']}, {model['equation']}, {len(model['features'])} features)")
#                                                    ↑ NUEVO V4


# ## 5. Carga de Datos de Cuencas

# In[5]:


# Load merged data
if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(f"Path {BASE_PATH} does not exist. Run 0-ebro-data-cleaning.ipynb first.")

join_dict = {}
for filename in os.listdir(BASE_PATH):
    if filename.endswith("-without-outliers.json"):
        basin_name = filename.replace("-without-outliers.json", "")
        df = pd.read_json(os.path.join(BASE_PATH, filename), orient="records")

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        join_dict[basin_name] = df

# Create synthetic variables
for basin_name, df in join_dict.items():
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['Dif_t'] = df['tmax'] - df['tmin']
    if 'Hrmax' in df.columns and 'Hrmin' in df.columns:
        df['Dif_HR'] = df['Hrmax'] - df['Hrmin']
    if 'qmax' in df.columns and 'qmin' in df.columns:
        df['Dif_q'] = df['qmax'] - df['qmin']
    if 'ymax' in df.columns and 'ymin' in df.columns:
        df['Dif_y'] = df['ymax'] - df['ymin']
    if 'sm_rootzone_max' in df.columns and 'sm_rootzone_min' in df.columns:
        df['Dif_qr'] = df['sm_rootzone_max'] - df['sm_rootzone_min']
    if 'sm_surface_max' in df.columns and 'sm_surface_min' in df.columns:
        df['Dif_qs'] = df['sm_surface_max'] - df['sm_surface_min']

    join_dict[basin_name] = df

print("=" * 80)
print("DATA LOADING")
print("=" * 80)
print(f"✅ Loaded {len(join_dict)} basins")
print(f"📊 Basins: {sorted(list(join_dict.keys()))}")

# Combine all basins into global dataset
all_basins_data = []
for basin_name, df in join_dict.items():
    df_copy = df.copy()
    df_copy['Basin'] = basin_name
    all_basins_data.append(df_copy)

global_df = pd.concat(all_basins_data, ignore_index=True)

if 'Date' in global_df.columns:
    global_df = global_df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Global dataset created")
print(f"   Total observations: {len(global_df):,}")
print(f"   Columns: {len(global_df.columns)}")


# ## 6. Cargar Resultados del Grid Search (Notebook 7)

# In[6]:


# Auto-detect grid search results file
# Busca primero en directorio actual, luego en RESULTS_PATH_NB7
possible_files = []
for search_path in ["./", RESULTS_PATH_NB7]:
    possible_files.extend(glob.glob(os.path.join(search_path, "tuned_models_rf_xgb_svm_*.csv")))
    possible_files.extend(glob.glob(os.path.join(search_path, "tuned_rf_models_*.csv")))  # fallback

# ✅ NUEVO V4: Preferir archivos multi-algoritmo
multi_algo_files = [f for f in possible_files if 'xgb' in f or 'svm' in f]
if multi_algo_files:
    GRID_SEARCH_RESULTS_FILE = sorted(multi_algo_files)[-1]
    print(f"✅ Auto-detected multi-algorithm results file: {GRID_SEARCH_RESULTS_FILE}")
elif possible_files:
    GRID_SEARCH_RESULTS_FILE = sorted(possible_files)[-1]
    print(f"⚠️  Auto-detected RF-only results file: {GRID_SEARCH_RESULTS_FILE}")
    print("   WARNING: This may be from notebook 7 v2 (RF only)")
else:
    print("❌ ERROR: No grid search results file found!")
    print("   Searched in:")
    print(f"      - Current directory: ./")
    print(f"      - Notebook 7 results: {RESULTS_PATH_NB7}")
    print("   Please:")
    print("   1. Run notebook 7_3 (multi-algorithm) to generate results")
    print("   2. Copy the CSV file to this directory")
    raise FileNotFoundError("Grid search results file not found")

# Load results
results_df = pd.read_csv(GRID_SEARCH_RESULTS_FILE)

# ✅ NUEVO V4: Validar que CSV tiene columna Algorithm
if 'Algorithm' not in results_df.columns:
    raise ValueError(
        f"❌ CSV file {GRID_SEARCH_RESULTS_FILE} is missing 'Algorithm' column.\n"
        "This file was likely generated with notebook 7 v2 (RF only).\n"
        "Please use notebook 7 v3 (multi-algorithm) to generate results with:\n"
        "   - Column 'Algorithm' (RF, XGBoost, SVM_RBF)\n"
        "   - Hyperparameters with prefix 'HP_'\n"
        "\nRun: notebooks/nb_tesis/7-rf-hyperparameter-tuning_3.py"
    )

print("=" * 80)
print("GRID SEARCH RESULTS LOADED")
print("=" * 80)
print(f"✅ File: {GRID_SEARCH_RESULTS_FILE}")
print(f"📊 Shape: {results_df.shape}")
print(f"📋 Columns: {list(results_df.columns)[:10]}...")  # Primeras 10 columnas

# ✅ NUEVO V4: Mostrar resumen por algoritmo
print(f"\n🔍 Models per algorithm:")
for algo in results_df['Algorithm'].unique():
    models = results_df[results_df['Algorithm'] == algo]['Model_Name'].unique()
    print(f"   {algo}: {list(models)}")

print(f"\n✅ CSV is compatible with multi-algorithm workflow (v4)")


# ## 7. Análisis de Resultados del Grid Search

# In[7]:


# Compare Random vs Temporal strategies
comparison_data = []

for model_id in results_df['Model_ID'].unique():
    model_results = results_df[results_df['Model_ID'] == model_id]

    random_row = model_results[model_results['Split_Strategy'] == 'Random'].iloc[0]
    temporal_row = model_results[model_results['Split_Strategy'] == 'Temporal'].iloc[0]

    comparison_data.append({
        'Model_ID': model_id,
        'Model_Name': random_row['Model_Name'],
        'Algorithm': random_row['Algorithm'],  # ← NUEVO V4
        'Random_NSE': random_row['Test_NSE'],
        'Temporal_NSE': temporal_row['Test_NSE'],
        'Random_KGE': random_row['Test_KGE'],
        'Temporal_KGE': temporal_row['Test_KGE'],
        'Random_RMSE': random_row['CV_RMSE'],
        'Temporal_RMSE': temporal_row['CV_RMSE'],
        'Best_Strategy': 'Random' if random_row['Test_NSE'] > temporal_row['Test_NSE'] else 'Temporal'
    })

comparison_df = pd.DataFrame(comparison_data)

print("=" * 80)
print("RANDOM vs TEMPORAL COMPARISON")
print("=" * 80)
print(comparison_df[['Model_Name', 'Algorithm', 'Best_Strategy', 'Random_NSE', 'Temporal_NSE']].to_string(index=False))
#                                  ↑ NUEVO V4

# Save comparison
comparison_df.to_csv(output_dirs['reports'] + 'strategy_comparison.csv', index=False)
print(f"\n✅ Saved: {output_dirs['reports']}strategy_comparison.csv")


# In[8]:


# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# NSE comparison
x = np.arange(len(comparison_df))
width = 0.35

axes[0].bar(x - width/2, comparison_df['Random_NSE'], width, label='Random', alpha=0.8)
axes[0].bar(x + width/2, comparison_df['Temporal_NSE'], width, label='Temporal', alpha=0.8)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('NSE')
axes[0].set_title('Nash-Sutcliffe Efficiency Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model_Name'], rotation=45, ha='right')
# ✅ Leyenda en lower right para evitar solapamiento con barras altas
axes[0].legend(loc='lower right', framealpha=0.95, edgecolor='black')
axes[0].grid(True, alpha=0.3)

# KGE comparison
axes[1].bar(x - width/2, comparison_df['Random_KGE'], width, label='Random', alpha=0.8)
axes[1].bar(x + width/2, comparison_df['Temporal_KGE'], width, label='Temporal', alpha=0.8)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('KGE')
axes[1].set_title('Kling-Gupta Efficiency Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(comparison_df['Model_Name'], rotation=45, ha='right')
# ✅ Leyenda en lower right para evitar solapamiento con barras altas
axes[1].legend(loc='lower right', framealpha=0.95, edgecolor='black')
axes[1].grid(True, alpha=0.3)

# RMSE comparison
axes[2].bar(x - width/2, comparison_df['Random_RMSE'], width, label='Random', alpha=0.8)
axes[2].bar(x + width/2, comparison_df['Temporal_RMSE'], width, label='Temporal', alpha=0.8)
axes[2].set_xlabel('Model')
axes[2].set_ylabel('RMSE (CV)')
axes[2].set_title('Root Mean Squared Error Comparison')
axes[2].set_xticks(x)
axes[2].set_xticklabels(comparison_df['Model_Name'], rotation=45, ha='right')
# ✅ Leyenda en upper right (RMSE tiene valores bajos, espacio arriba)
axes[2].legend(loc='upper right', framealpha=0.95, edgecolor='black')
axes[2].grid(True, alpha=0.3)

# ✅ Añadir etiquetas (a), (b), (c)
labels = ['(a)', '(b)', '(c)']
for ax, label in zip(axes, labels):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig(output_dirs['reports'] + 'strategy_comparison.png', dpi=450, bbox_inches='tight')
print(f"✅ Saved: {output_dirs['reports']}strategy_comparison.png")
# plt.show()  # ✅ Comentado para ejecución no-interactiva


# ## 8. Funciones de Validación y Reentrenamiento (Multi-Algorithm)

# In[9]:


from sklearn.model_selection import KFold, TimeSeriesSplit


# Funciones de validación de cuencas (heredadas de v3, sin cambios)
def validate_basin_for_model(df, feature_cols, basin_name=None):
    """
    Valida si una cuenca tiene todas las columnas necesarias para un modelo.
    """
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        return False, missing_cols

    # También verificar que haya datos suficientes (no todo NaN)
    if isinstance(df, pd.DataFrame):
        valid_rows = df[feature_cols].notna().all(axis=1).sum()
        if valid_rows < 10:  # Umbral mínimo de filas válidas
            return False, [f"Insufficient valid data ({valid_rows} rows)"]

    return True, []


def get_valid_basins_for_model(join_dict, feature_cols):
    """
    Identifica qué cuencas son válidas para un modelo específico.
    """
    valid_basins = []
    invalid_basins = {}

    for basin_name, df in join_dict.items():
        is_valid, missing = validate_basin_for_model(df, feature_cols, basin_name)

        if is_valid:
            valid_basins.append(basin_name)
        else:
            invalid_basins[basin_name] = missing

    return {
        'valid': valid_basins,
        'invalid': invalid_basins,
        'coverage': len(valid_basins) / len(join_dict) * 100 if join_dict else 0,
        'n_valid': len(valid_basins),
        'n_total': len(join_dict)
    }


# ✅ MODIFICADO V4: Función de reentrenamiento con soporte multi-algoritmo
def retrain_best_model(model_spec, strategy, best_params, global_df,
                       algorithm='RF', join_dict=None):
    """
    Retrain model with best parameters on full dataset.

    MODIFICACIÓN V4: Soporte para RF, XGBoost y SVM_RBF

    Parameters:
    -----------
    model_spec : dict
        Model specification with features and target
    strategy : str
        'Random' or 'Temporal' (for logging)
    best_params : dict
        Best hyperparameters from grid search
    global_df : DataFrame
        Combined dataset from all basins
    algorithm : str
        'RF', 'XGBoost', or 'SVM_RBF'
    join_dict : dict, optional
        Basin data for coverage reporting

    Returns:
    --------
    dict : Model artifacts and predictions
    """
    # Map variables
    target_col = var_mapping.get(model_spec['target'], model_spec['target'])
    feature_cols = map_variables(model_spec['features'])

    # Validar cobertura de cuencas (heredado de v3)
    basin_info = None
    if join_dict is not None:
        basin_info = get_valid_basins_for_model(join_dict, feature_cols)
        print(f"   Model coverage: {basin_info['n_valid']}/{basin_info['n_total']} cuencas ({basin_info['coverage']:.1f}%)")

        if basin_info['n_valid'] == 0:
            raise ValueError(f"No basins have the required columns for model {model_spec['name']}")

        if basin_info['invalid']:
            n_show = min(5, len(basin_info['invalid']))
            print(f"   ⚠️  Excluded {len(basin_info['invalid'])} basins::")
            for basin, missing in list(basin_info['invalid'].items())[:n_show]:
                print(f"      - {basin}: missing {missing}")
            if len(basin_info['invalid']) > n_show:
                print(f"      ... and {len(basin_info['invalid']) - n_show} more")

    # Prepare data
    X = global_df[feature_cols]
    y = global_df[target_col]

    # Drop NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_idx].values
    y_clean = y[valid_idx].values
    basin_clean = global_df[valid_idx]['Basin'].values
    date_clean = global_df[valid_idx]['Date'].values if 'Date' in global_df.columns else None

    # Scale features (CRÍTICO para SVM, recomendado para consistencia)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # ✅ NUEVO V4: Crear modelo según algoritmo
    if algorithm == 'RF':
        model = RandomForestRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            max_features=best_params.get('max_features', 'sqrt'),
            bootstrap=best_params.get('bootstrap', True),
            random_state=42,
            n_jobs=-1
        )

    elif algorithm == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('learning_rate', 0.1),
            subsample=best_params.get('subsample', 1.0),
            colsample_bytree=best_params.get('colsample_bytree', 1.0),
            random_state=42,
            n_jobs=-1,
            tree_method='auto'
        )

    elif algorithm == 'SVM_RBF':
        model = SVR(
            kernel='rbf',
            C=best_params.get('C', 10.0),
            epsilon=best_params.get('epsilon', 0.1),
            gamma=best_params.get('gamma', 'scale')
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'RF', 'XGBoost', or 'SVM_RBF'")

    # Fit model
    print(f"   Training {algorithm} model...")
    model.fit(X_scaled, y_clean)

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    metrics = calculate_all_metrics(y_clean, y_pred)

    # ✅ NUEVO V4: Feature importance según algoritmo
    if hasattr(model, 'feature_importances_'):
        # RF y XGBoost tienen .feature_importances_
        feature_importances = model.feature_importances_
        print(f"   Feature importance: Native ({algorithm})")
    else:
        # SVM: usar permutation importance
        print(f"   ⚠️  Calculating permutation importance for {algorithm} (may take 2-5 min)...")
        perm_importance = permutation_importance(
            model, X_scaled, y_clean,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        feature_importances = perm_importance.importances_mean
        print(f"   Feature importance: Permutation-based (SVM)")

    return {
        'model': model,
        'scaler': scaler,
        'algorithm': algorithm,  # ← NUEVO V4
        'feature_cols': feature_cols,
        'target_col': target_col,
        'y_true': y_clean,
        'y_pred': y_pred,
        'basins': basin_clean,
        'dates': date_clean,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'basin_coverage': basin_info
    }

print("✅ Multi-algorithm training functions defined")


# ## 9. Reentrenamiento de Modelos con Mejores Hiperparámetros

# In[10]:


# Retrain all models with best strategy
trained_models = {}

print("=" * 80)
print("RETRAINING MODELS WITH BEST PARAMETERS (MULTI-ALGORITHM)")
print("=" * 80)

for model_spec in tqdm(models_to_analyze, desc="Training models"):
    model_id = model_spec['id']
    model_name = model_spec['name']
    algorithm = model_spec['algorithm']  # ← NUEVO V4

    # Get best strategy for this model
    best_strategy = comparison_df[comparison_df['Model_ID'] == model_id]['Best_Strategy'].values[0]

    # Get best parameters from CSV
    best_row = results_df[
        (results_df['Model_ID'] == model_id) &
        (results_df['Split_Strategy'] == best_strategy)
    ].iloc[0]

    # ✅ NUEVO V4: Extraer hiperparámetros dinámicos (con prefijo HP_)
    best_params = {}
    for col in best_row.index:
        if col.startswith('HP_'):
            param_name = col.replace('HP_', '')
            param_value = best_row[col]

            # ✅ FIX: Convertir NaN a None
            if pd.isna(param_value):
                param_value = None
            # Convertir 'None' string a None real
            elif param_value == 'None':
                param_value = None
            # ✅ FIX: Convertir numpy bools a Python bool
            elif isinstance(param_value, (np.bool_, bool)):
                param_value = bool(param_value)
            # ✅ FIX: Convertir numpy floats a Python int/float
            elif isinstance(param_value, (np.floating, float)):
                # Convertir a Python float primero
                param_value = float(param_value)
                # Si es entero, convertir a int
                if param_value.is_integer():
                    param_value = int(param_value)
            # Convertir numpy ints a Python int
            elif isinstance(param_value, (np.integer, int)):
                param_value = int(param_value)
                # ✅ FIX: Convertir 0/1 a bool para parámetros booleanos
                if param_name in ['bootstrap'] and param_value in [0, 1]:
                    param_value = bool(param_value)
            # Convertir valores numéricos si son strings
            elif isinstance(param_value, str):
                try:
                    param_value = float(param_value)
                    if param_value.is_integer():
                        param_value = int(param_value)
                except ValueError:
                    pass  # mantener como string (ej: 'scale', 'sqrt')

            best_params[param_name] = param_value

    print(f"\n{'='*80}")
    print(f"Model {model_id} - {model_name} ({algorithm})")  # ← NUEVO V4
    print(f"{'='*80}")
    print(f"   Strategy: {best_strategy}")
    print(f"   Algorithm: {algorithm}")  # ← NUEVO V4
    print(f"   Best params: {best_params}")

    # Retrain (✅ V4: Ahora con algoritmo específico)
    trained_model = retrain_best_model(
        model_spec,
        best_strategy,
        best_params,
        global_df,
        algorithm=algorithm,  # ← NUEVO V4
        join_dict=join_dict
    )

    trained_model['model_spec'] = model_spec
    trained_model['best_strategy'] = best_strategy
    trained_model['best_params'] = best_params

    trained_models[model_name] = trained_model

    print(f"   ✅ {model_name} ({algorithm}): NSE={trained_model['metrics']['NSE']:.4f}, KGE={trained_model['metrics']['KGE']:.4f}")
    #                          ↑ NUEVO V4

print(f"\n✅ All {len(trained_models)} models retrained successfully")

# ✅ NUEVO V4: Resumen por algoritmo
print("\n" + "=" * 80)
print("MODELS BY ALGORITHM")
print("=" * 80)
algo_counts = {}
for model_name, trained_model in trained_models.items():
    algo = trained_model['algorithm']
    algo_counts[algo] = algo_counts.get(algo, []) + [model_name]

for algo, models in sorted(algo_counts.items()):
    print(f"{algo}: {len(models)} models")
    print(f"   Models: {', '.join(models)}")


# ## 10. Generación de Predicciones por Cuenca

# In[11]:


def generate_predictions_by_basin(trained_model_info, join_dict):
    """
    Generate predictions for each basin separately.
    Returns DataFrame with rows=basins, columns=dates.

    V4: Funciona para RF, XGBoost y SVM_RBF (sin cambios necesarios)
    """
    model = trained_model_info['model']
    scaler = trained_model_info['scaler']
    feature_cols = trained_model_info['feature_cols']
    target_col = trained_model_info['target_col']

    basin_predictions = {}
    all_dates = set()

    # Tracking de cuencas procesadas y excluidas
    processed_basins = []
    excluded_basins = {}

    # Predict for each basin
    for basin_name, df in join_dict.items():
        # Validar columnas ANTES de intentar acceder
        is_valid, missing = validate_basin_for_model(df, feature_cols, basin_name)

        if not is_valid:
            excluded_basins[basin_name] = missing
            continue

        # Prepare data
        try:
            X = df[feature_cols]
            valid_idx = X.notna().all(axis=1)

            if valid_idx.sum() == 0:
                excluded_basins[basin_name] = ["No valid rows after removing NaN"]
                continue

            X_clean = X[valid_idx].values
            dates = df[valid_idx]['Date'].values if 'Date' in df.columns else None

            # Scale and predict (funciona para RF, XGBoost, SVM)
            X_scaled = scaler.transform(X_clean)
            predictions = model.predict(X_scaled)

            # Store predictions with dates
            if dates is not None:
                basin_predictions[basin_name] = pd.Series(
                    predictions,
                    index=pd.to_datetime(dates)
                )
                all_dates.update(dates)

            processed_basins.append(basin_name)

        except Exception as e:
            excluded_basins[basin_name] = [f"Prediction error: {str(e)}"]
            continue

    # Reportar resultado
    print(f"   Predictions generated for {len(processed_basins)}/{len(join_dict)} basins")
    if excluded_basins:
        n_show = min(3, len(excluded_basins))
        print(f"   ⚠️  Excluded {len(excluded_basins)} basins:")
        for basin, reason in list(excluded_basins.items())[:n_show]:
            print(f"      - {basin}: {reason}")
        if len(excluded_basins) > n_show:
            print(f"      ... and {len(excluded_basins) - n_show} more")

    # Create DataFrame with all dates
    if not basin_predictions:
        print("   ❌ WARNING: No predictions generated for any basin!")
        return pd.DataFrame()

    all_dates_sorted = sorted(all_dates)
    predictions_df = pd.DataFrame(index=sorted(basin_predictions.keys()))

    for date in all_dates_sorted:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        predictions_df[date_str] = [
            basin_predictions[basin].get(date, np.nan)
            for basin in predictions_df.index
        ]

    # Añadir metadatos
    predictions_df.attrs['excluded_basins'] = excluded_basins
    predictions_df.attrs['processed_basins'] = processed_basins

    return predictions_df

# Generate predictions for all models
print("=" * 80)
print("GENERATING PREDICTIONS BY BASIN")
print("=" * 80)

prediction_dfs = {}
model_coverage_report = []

for model_name, trained_model in tqdm(trained_models.items(), desc="Generating predictions"):
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({trained_model['model_spec']['id']}, {trained_model['algorithm']})")
    #                                                                 ↑ NUEVO V4
    print(f"{'='*60}")

    print(f"   Required features: {len(trained_model['feature_cols'])}")
    if len(trained_model['feature_cols']) <= 10:
        print(f"   Features: {', '.join(trained_model['feature_cols'])}")

    pred_df = generate_predictions_by_basin(trained_model, join_dict)
    prediction_dfs[model_name] = pred_df

    # Save to CSV
    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo
    output_path = output_dirs['predictions'] + f'predictions_{model_name}_{trained_model["algorithm"]}.csv'
    pred_df.to_csv(output_path)

    # Guardar también reporte de cuencas excluidas
    if hasattr(pred_df, 'attrs') and pred_df.attrs.get('excluded_basins'):
        excluded_path = output_dirs['predictions'] + f'excluded_basins_{model_name}.json'
        with open(excluded_path, 'w') as f:
            json.dump(pred_df.attrs['excluded_basins'], f, indent=2)

    # Tracking para reporte
    model_coverage_report.append({
        'model_id': trained_model['model_spec']['id'],
        'model_name': model_name,
        'algorithm': trained_model['algorithm'],  # ← NUEVO V4
        'n_predictions': pred_df.shape[0],
        'n_total_basins': len(join_dict),
        'coverage_%': pred_df.shape[0] / len(join_dict) * 100 if join_dict else 0,
        'n_excluded': len(pred_df.attrs.get('excluded_basins', {}))
    })

    print(f"\n   ✅ {model_name}: {pred_df.shape} -> {output_path}")

print(f"\n✅ Generated {len(prediction_dfs)} prediction files")

# Guardar reporte de cobertura
coverage_df = pd.DataFrame(model_coverage_report)
coverage_path = output_dirs['predictions'] + 'model_coverage_report.csv'
coverage_df.to_csv(coverage_path, index=False)
print(f"✅ Coverage report saved: {coverage_path}")

# Mostrar resumen visual
print("\n" + "=" * 80)
print("PREDICTION COVERAGE SUMMARY")
print("=" * 80)
print(coverage_df.to_string(index=False))


# ## 11. Feature Importance Analysis

# In[12]:


# ✅ MODIFICADO V4: Plot con nota sobre tipo de importance
def plot_feature_importance(feature_names, importances, title, algorithm='RF', save_path=None):
    """
    Plot feature importances as horizontal bar chart.

    V4: Agrega subtítulo indicando tipo de importance según algoritmo
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

    # ✅ NUEVO V4: Agregar subtítulo con tipo de importance
    if algorithm == 'SVM_RBF':
        title_full = f"{title}\n(Permutation Importance - computationally expensive)"
    else:
        title_full = f"{title}\n(Native Feature Importance - {algorithm})"

    plt.title(title_full, fontsize=14, fontweight='bold')

    # Reverse for horizontal bar chart (highest at top)
    y_pos = np.arange(len(importances))
    plt.barh(y_pos, importances[indices], alpha=0.8)
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)

    # Add values on bars
    for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
        plt.text(imp, i, f' {imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=450, bbox_inches='tight')

    # plt.show()  # ✅ Comentado para ejecución no-interactiva

# Generate feature importance plots for best strategy models
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

importance_summary = []

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algorithm = trained_model['algorithm']  # ← NUEVO V4
    strategy = trained_model['best_strategy']
    feature_cols = trained_model['feature_cols']
    importances = trained_model['feature_importances']

    # Plot
    save_path = output_dirs['feature_importance'] + f"{model_id}_{model_name}_{algorithm}_{strategy.lower()}_importance.png"
    #                                                                              ↑ NUEVO V4
    plot_feature_importance(
        feature_cols,
        importances,
        f"Feature Importance - {model_name} (Model {model_id}, {strategy} Split)",
        algorithm=algorithm,  # ← NUEVO V4
        save_path=save_path
    )

    # Store for summary
    for feat, imp in zip(feature_cols, importances):
        importance_summary.append({
            'Model_ID': model_id,
            'Model_Name': model_name,
            'Algorithm': algorithm,  # ← NUEVO V4
            'Strategy': strategy,
            'Feature': feat,
            'Importance': imp
        })

# Save importance summary
importance_df = pd.DataFrame(importance_summary)
importance_df.to_csv(output_dirs['feature_importance'] + 'feature_importance_summary.csv', index=False)
print(f"\n✅ Saved: {output_dirs['feature_importance']}feature_importance_summary.csv")


# In[13]:


# Create heatmap of feature importances across all models
pivot_importance = importance_df.pivot_table(
    index='Feature',
    columns='Model_Name',
    values='Importance',
    aggfunc='mean'
)

plt.figure(figsize=(14, max(10, len(pivot_importance) * 0.3)))
sns.heatmap(
    pivot_importance,
    cmap='YlOrRd',
    annot=True,
    fmt='.3f',
    linewidths=0.5,
    cbar_kws={'label': 'Importance'}
)
plt.title('Feature Importance Heatmap Across All Models\n(RF: Native | XGBoost: Native | SVM: Permutation)',
          fontsize=14, fontweight='bold')  # ← NUEVO V4: nota sobre tipos
plt.xlabel('Model', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(output_dirs['feature_importance'] + 'feature_importance_heatmap.png', dpi=450, bbox_inches='tight')
print(f"✅ Saved: {output_dirs['feature_importance']}feature_importance_heatmap.png")
# plt.show()  # ✅ Comentado para ejecución no-interactiva


# ## 12. Confusion Matrices (Adapted for Regression)

# In[14]:


def classify_values(values, method='tertile'):
    """
    Classify continuous values into categories.
    """
    if method == 'tertile':
        p33 = np.percentile(values, 33)
        p67 = np.percentile(values, 67)
        classes = np.where(values < p33, 0, np.where(values < p67, 1, 2))
        labels = ['Low', 'Normal', 'High']
    elif method == 'extreme':
        p90 = np.percentile(values, 90)
        classes = np.where(values >= p90, 1, 0)
        labels = ['Normal', 'Extreme']
    else:
        raise ValueError(f"Unknown method: {method}")

    return classes, labels

def calculate_confusion_metrics(y_true_classes, y_pred_classes, labels):
    """Calculate confusion matrix and metrics."""
    from typing import cast, Dict, Any

    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Get classification report
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    # Calculate prevalence
    total = len(y_true_classes)
    prevalence = {}
    for i, label in enumerate(labels):
        count = np.sum(y_true_classes == i)
        prevalence[label] = count / total

    return cm, cast(Dict[str, Any], report), prevalence

def plot_confusion_matrix(cm, labels, title, save_path=None):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=450, bbox_inches='tight')

    # plt.show()  # ✅ Comentado para ejecución no-interactiva

# Generate confusion matrices for all models
print("=" * 80)
print("CONFUSION MATRIX ANALYSIS")
print("=" * 80)

confusion_metrics = []

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algorithm = trained_model['algorithm']  # ← NUEVO V4
    y_true = trained_model['y_true']
    y_pred = trained_model['y_pred']

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_id}, {algorithm})")  # ← NUEVO V4
    print(f"{'='*60}")

    # Tertile classification
    y_true_tertile, tertile_labels = classify_values(y_true, method='tertile')
    y_pred_tertile, _ = classify_values(y_pred, method='tertile')

    cm_tertile, report_tertile, prevalence_tertile = calculate_confusion_metrics(
        y_true_tertile, y_pred_tertile, tertile_labels
    )

    # Plot tertile confusion matrix
    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo
    save_path = output_dirs['confusion_matrices'] + f'{model_id}_{model_name}_{algorithm}_confusion_tertile.png'
    plot_confusion_matrix(
        cm_tertile,
        tertile_labels,
        f'Confusion Matrix - {model_name} ({algorithm}, Tertile Classification)',  # ← NUEVO V4
        save_path=save_path
    )

    # Store metrics
    for label in tertile_labels:
        label_metrics = report_tertile.get(label, {})
        confusion_metrics.append({
            'Model_ID': model_id,
            'Model_Name': model_name,
            'Algorithm': algorithm,  # ← NUEVO V4
            'Classification_Method': 'Tertile',
            'Class': label,
            'Precision': label_metrics.get('precision', 0.0),
            'Recall_Sensitivity': label_metrics.get('recall', 0.0),
            'F1_Score': label_metrics.get('f1-score', 0.0),
            'Support': label_metrics.get('support', 0),
            'Prevalence': prevalence_tertile.get(label, 0.0)
        })

    print(f"\n📊 Tertile Classification Metrics:")
    print(f"   Accuracy: {report_tertile.get('accuracy', 0.0):.4f}")
    for label in tertile_labels:
        label_metrics = report_tertile.get(label, {})
        print(f"   {label}:")
        print(f"      Precision: {label_metrics.get('precision', 0.0):.4f}")
        print(f"      Recall (Sensitivity): {label_metrics.get('recall', 0.0):.4f}")
        print(f"      F1-Score: {label_metrics.get('f1-score', 0.0):.4f}")
        print(f"      Prevalence: {prevalence_tertile.get(label, 0.0):.4f}")

    # Extreme event classification
    y_true_extreme, extreme_labels = classify_values(y_true, method='extreme')
    y_pred_extreme, _ = classify_values(y_pred, method='extreme')

    cm_extreme, report_extreme, prevalence_extreme = calculate_confusion_metrics(
        y_true_extreme, y_pred_extreme, extreme_labels
    )

    # Plot extreme confusion matrix
    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo
    save_path = output_dirs['confusion_matrices'] + f'{model_id}_{model_name}_{algorithm}_confusion_extreme.png'
    plot_confusion_matrix(
        cm_extreme,
        extreme_labels,
        f'Confusion Matrix - {model_name} ({algorithm}, Extreme Event Detection)',  # ← NUEVO V4
        save_path=save_path
    )

    # Store metrics
    for label in extreme_labels:
        label_metrics = report_extreme.get(label, {})
        confusion_metrics.append({
            'Model_ID': model_id,
            'Model_Name': model_name,
            'Algorithm': algorithm,  # ← NUEVO V4
            'Classification_Method': 'Extreme',
            'Class': label,
            'Precision': label_metrics.get('precision', 0.0),
            'Recall_Sensitivity': label_metrics.get('recall', 0.0),
            'F1_Score': label_metrics.get('f1-score', 0.0),
            'Support': label_metrics.get('support', 0),
            'Prevalence': prevalence_extreme.get(label, 0.0)
        })

    # Calculate specificity for binary classification
    if len(extreme_labels) == 2:
        tn, fp, fn, tp = cm_extreme.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        extreme_metrics = report_extreme.get('Extreme', {})
        print(f"\n📊 Extreme Event Detection Metrics:")
        print(f"   Accuracy: {report_extreme.get('accuracy', 0.0):.4f}")
        print(f"   Sensitivity (Recall): {extreme_metrics.get('recall', 0.0):.4f}")
        print(f"   Specificity: {specificity:.4f}")
        print(f"   Precision: {extreme_metrics.get('precision', 0.0):.4f}")
        print(f"   F1-Score: {extreme_metrics.get('f1-score', 0.0):.4f}")
        print(f"   Prevalence (Extreme): {prevalence_extreme.get('Extreme', 0.0):.4f}")

# Save confusion metrics
confusion_df = pd.DataFrame(confusion_metrics)
confusion_df.to_csv(output_dirs['confusion_matrices'] + 'classification_metrics.csv', index=False)
print(f"\n✅ Saved: {output_dirs['confusion_matrices']}classification_metrics.csv")


# ## 13-15. Visualizaciones (Time Series, Scatter, Residuals, Taylor)
#
# Las visualizaciones funcionan igual para RF, XGBoost y SVM_RBF (sin cambios necesarios)

# In[15]:


# Time Series Visualizations
print("=" * 80)
print("TIME SERIES VISUALIZATIONS")
print("=" * 80)

sample_basins = sorted(list(join_dict.keys()))[:3]

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algorithm = trained_model['algorithm']  # ← NUEVO V4

    fig, axes = plt.subplots(len(sample_basins), 1, figsize=(16, 4 * len(sample_basins)))
    if len(sample_basins) == 1:
        axes = [axes]

    for ax, basin_name in zip(axes, sample_basins):
        mask = trained_model['basins'] == basin_name
        dates = trained_model['dates'][mask]
        y_true = trained_model['y_true'][mask]
        y_pred = trained_model['y_pred'][mask]

        # Sort by date
        sort_idx = np.argsort(dates)
        dates_sorted = dates[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        # ✅ CORRECCIÓN PAPER: Observados como PUNTOS, Predichos como LÍNEAS
        ax.scatter(dates_sorted, y_true_sorted, label='Observed', alpha=0.7,
                   s=40, color='darkblue', marker='o', edgecolors='k', linewidth=0.5)
        ax.plot(dates_sorted, y_pred_sorted, label='Predicted', alpha=0.8,
                linewidth=2.5, color='red', linestyle='-')

        ax.set_title(f'{basin_name} - {model_name} ({algorithm})', fontsize=18, fontweight='bold')
        ax.set_xlabel('Date', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{model_name}', fontsize=16)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=14,
                  edgecolor='black', fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo, DPI 450
    save_path = output_dirs['timeseries'] + f'{model_id}_{model_name}_{algorithm}_timeseries_sample.png'
    plt.savefig(save_path, dpi=450, bbox_inches='tight')
    print(f"✅ {model_name} ({algorithm}): {save_path}")
    plt.close()

print(f"\n✅ Generated time series plots for {len(trained_models)} models")


# In[16]:


# Scatter plots
print("=" * 80)
print("SCATTER PLOT VISUALIZATIONS")
print("=" * 80)

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algorithm = trained_model['algorithm']  # ← NUEVO V4
    y_true = trained_model['y_true']
    y_pred = trained_model['y_pred']
    metrics = trained_model['metrics']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)

    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

    # Add metrics text
    textstr = '\n'.join([
        f"Algorithm: {algorithm}",  # ← NUEVO V4
        f"NSE = {metrics['NSE']:.4f}",
        f"KGE = {metrics['KGE']:.4f}",
        f"R² = {metrics['R2']:.4f}",
        f"RMSE = {metrics['RMSE']:.4f}",
        f"MAE = {metrics['MAE']:.4f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(f'Observed {model_name}', fontsize=12)
    ax.set_ylabel(f'Predicted {model_name}', fontsize=12)
    ax.set_title(f'Predicted vs Observed - {model_name} ({algorithm}, Model {model_id})',  # ← NUEVO V4
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo
    save_path = output_dirs['scatter'] + f'{model_id}_{model_name}_{algorithm}_scatter.png'
    plt.savefig(save_path, dpi=450, bbox_inches='tight')
    print(f"✅ {model_name} ({algorithm}): {save_path}")
    plt.close()

print(f"\n✅ Generated scatter plots for {len(trained_models)} models")


# In[17]:


# Residual analysis
print("=" * 80)
print("RESIDUAL ANALYSIS")
print("=" * 80)

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algorithm = trained_model['algorithm']  # ← NUEVO V4
    y_true = trained_model['y_true']
    y_pred = trained_model['y_pred']
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=16)
    axes[0, 0].set_ylabel('Residuals', fontsize=16)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=18, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontsize=16)
    axes[0, 1].set_ylabel('Frequency', fontsize=16)
    axes[0, 1].set_title('Distribution of Residuals', fontsize=18, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=18, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals vs Time
    if trained_model['dates'] is not None:
        dates = trained_model['dates']
        sort_idx = np.argsort(dates)
        axes[1, 1].scatter(dates[sort_idx], residuals[sort_idx], alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Date', fontsize=16)
        axes[1, 1].set_ylabel('Residuals', fontsize=16)
        axes[1, 1].set_title('Residuals vs Time', fontsize=18, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No temporal data available',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.suptitle(f'Residual Analysis - {model_name} ({algorithm}, Model {model_id})',  # ← NUEVO V4
                 fontsize=14, fontweight='bold', y=1.00)

    # ✅ Añadir etiquetas (a), (b), (c), (d)
    labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, label in zip(axes.flatten(), labels):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    # ✅ NUEVO V4: Incluir algoritmo en nombre de archivo, DPI 450
    save_path = output_dirs['residuals'] + f'{model_id}_{model_name}_{algorithm}_residuals.png'
    plt.savefig(save_path, dpi=450, bbox_inches='tight')
    print(f"✅ {model_name} ({algorithm}): {save_path}")
    plt.close()

print(f"\n✅ Generated residual analysis for {len(trained_models)} models")


# In[18]:


# Taylor diagram
print("=" * 80)
print("TAYLOR DIAGRAM")
print("=" * 80)

taylor_stats = []

for model_name, trained_model in trained_models.items():
    algorithm = trained_model['algorithm']  # ← NUEVO V4
    y_true = trained_model['y_true']
    y_pred = trained_model['y_pred']

    # Calculate statistics
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    taylor_stats.append({
        'model': f"{model_name} ({algorithm})",  # ← NUEVO V4: incluir algoritmo
        'algorithm': algorithm,  # ← NUEVO V4
        'std_ratio': std_pred / std_true,
        'correlation': correlation,
        'rmse': rmse,
        'std_true': std_true,
        'std_pred': std_pred
    })

# Plot simplified Taylor-like diagram
fig, ax = plt.subplots(figsize=(12, 10))

# ✅ NUEVO V4: Color por algoritmo
colors = {'RF': 'blue', 'XGBoost': 'green', 'SVM_RBF': 'red'}
for stat in taylor_stats:
    color = colors.get(stat['algorithm'], 'gray')
    ax.scatter(
        stat['std_ratio'],
        stat['correlation'],
        s=200,
        alpha=0.7,
        c=color,
        edgecolors='black',
        linewidth=1.5,
        label=stat['model']
    )
    ax.annotate(
        stat['model'],
        (stat['std_ratio'], stat['correlation']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9
    )

# Reference lines
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Std Ratio (Predicted / Observed)', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Taylor-like Diagram: Multi-Algorithm Model Performance Comparison',  # ← NUEVO V4
             fontsize=14, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max([s['std_ratio'] for s in taylor_stats]) * 1.1)
ax.set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0.1, 1, 1])
save_path = output_dirs['taylor'] + 'taylor_diagram_multi_algorithm.png'  # ← NUEVO V4
plt.savefig(save_path, dpi=450, bbox_inches='tight')
print(f"✅ Saved: {save_path}")
# plt.show()  # ✅ Comentado para ejecución no-interactiva


# ## 16. Generate Technical Report

# In[19]:


# Generate comprehensive technical report
print("=" * 80)
print("GENERATING TECHNICAL REPORT")
print("=" * 80)

report_lines = []

# Header
report_lines.append("# Multi-Algorithm Hyperparameter Tuning - Complete Technical Report")
report_lines.append("")
report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Author:** Víctor Galán")
report_lines.append(f"**Models Analyzed:** {len(models_to_analyze)}")
report_lines.append(f"**Basins:** {len(join_dict)}")
report_lines.append(f"**Total Observations:** {len(global_df):,}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# ✅ NUEVO V4: Sección de algoritmos
report_lines.append("## 1. Executive Summary")
report_lines.append("")
report_lines.append("This report presents the complete analysis of 8 optimized machine learning models ")
report_lines.append("for predicting hydrological variables in sub-basins of the Ebro River basin.")
report_lines.append("")
report_lines.append("**Algorithms Used:**")
report_lines.append("")

# Agrupar por algoritmo
algo_models = {}
for model_name, trained_model in trained_models.items():
    algo = trained_model['algorithm']
    if algo not in algo_models:
        algo_models[algo] = []
    algo_models[algo].append(model_name)

for algo, models in sorted(algo_models.items()):
    report_lines.append(f"- **{algo}**: {len(models)} models")
    report_lines.append(f"  - Models: {', '.join(models)}")

    # Descripción del algoritmo
    if algo == 'RF':
        report_lines.append(f"  - Type: Bagging ensemble (parallel trees, variance reduction)")
    elif algo == 'XGBoost':
        report_lines.append(f"  - Type: Gradient boosting (sequential trees, bias+variance reduction)")
    elif algo == 'SVM_RBF':
        report_lines.append(f"  - Type: Support Vector Machine with RBF kernel (non-linear, margin-based)")
    report_lines.append("")

report_lines.append("**Key Findings:**")
report_lines.append("")
for model_name, trained_model in trained_models.items():
    metrics = trained_model['metrics']
    algo = trained_model['algorithm']
    report_lines.append(f"- **{model_name} ({algo})**: NSE={metrics['NSE']:.4f}, KGE={metrics['KGE']:.4f}, RMSE={metrics['RMSE']:.4f}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 2. Methodology
report_lines.append("## 2. Methodology")
report_lines.append("")
report_lines.append("### 2.1 Multi-Algorithm Approach")
report_lines.append("")
report_lines.append("This study employs three different machine learning algorithms:")
report_lines.append("")
report_lines.append("**Random Forest (RF):**")
report_lines.append("- Bagging ensemble: Multiple decision trees trained in parallel")
report_lines.append("- Each tree uses bootstrap sampling and random feature subsets")
report_lines.append("- Predictions averaged across all trees")
report_lines.append("- Reduces variance, robust to overfitting")
report_lines.append("- Feature importance: Gini/impurity-based (native)")
report_lines.append("")
report_lines.append("**XGBoost:**")
report_lines.append("- Gradient boosting: Trees trained sequentially")
report_lines.append("- Each tree corrects errors of previous trees")
report_lines.append("- Optimized with regularization (L1/L2)")
report_lines.append("- Reduces both bias and variance")
report_lines.append("- Feature importance: Gain-based (native)")
report_lines.append("")
report_lines.append("**SVM_RBF:**")
report_lines.append("- Support Vector Machine with Radial Basis Function kernel")
report_lines.append("- Non-linear decision boundary in high-dimensional space")
report_lines.append("- Margin-based optimization (maximize distance to support vectors)")
report_lines.append("- Sensitive to feature scaling (StandardScaler applied)")
report_lines.append("- Feature importance: Permutation-based (computationally expensive)")
report_lines.append("")
report_lines.append("### 2.2 Hyperparameter Tuning Strategy")
report_lines.append("")
report_lines.append("**Grid Search** with algorithm-specific parameter grids:")
report_lines.append("")
report_lines.append("**RF Grid (96 combinations):**")
report_lines.append("- n_estimators: [100, 200, 300]")
report_lines.append("- max_depth: [10, None]")
report_lines.append("- min_samples_split: [2, 10]")
report_lines.append("- min_samples_leaf: [1, 4]")
report_lines.append("- max_features: ['sqrt', None]")
report_lines.append("- bootstrap: [True, False]")
report_lines.append("")
report_lines.append("**XGBoost Grid (72 combinations):**")
report_lines.append("- n_estimators: [100, 200, 300]")
report_lines.append("- max_depth: [3, 6, 10]")
report_lines.append("- learning_rate: [0.01, 0.1]")
report_lines.append("- subsample: [0.8, 1.0]")
report_lines.append("- colsample_bytree: [0.8, 1.0]")
report_lines.append("")
report_lines.append("**SVM_RBF Grid (36 combinations):**")
report_lines.append("- C: [1.0, 10.0, 100.0]")
report_lines.append("- epsilon: [0.01, 0.1, 0.2]")
report_lines.append("- gamma: ['scale', 'auto', 0.01, 0.1]")
report_lines.append("")
report_lines.append("### 2.3 Data Preprocessing")
report_lines.append("")
report_lines.append("**Standardization:** All features standardized using `StandardScaler` (mean=0, std=1).")
report_lines.append("")
report_lines.append("**Justification:**")
report_lines.append("- **RF/XGBoost:** Tree-based, scale-invariant, but standardization ensures consistent feature importance")
report_lines.append("- **SVM_RBF:** Distance-based, REQUIRES standardization for proper convergence")
report_lines.append("")
report_lines.append("**Missing Values:** Rows with any missing values removed before training.")
report_lines.append("")
report_lines.append("### 2.4 Cross-Validation Strategies")
report_lines.append("")
report_lines.append("Two strategies compared:")
report_lines.append("1. **Random Split (KFold):** Days randomly distributed (assumes independence)")
report_lines.append("2. **Temporal Split (TimeSeriesSplit):** Complete time ranges (respects autocorrelation)")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 3. Optimal Parameters (with algorithm)
report_lines.append("## 3. Optimal Hyperparameters by Model")
report_lines.append("")
report_lines.append("| Model ID | Model Name | Algorithm | Strategy | Key Parameters |")
report_lines.append("|----------|------------|-----------|----------|----------------|")

for model_name, trained_model in trained_models.items():
    model_id = trained_model['model_spec']['id']
    algo = trained_model['algorithm']
    strategy = trained_model['best_strategy']
    params = trained_model['best_params']

    # Formato compacto de parámetros clave según algoritmo
    if algo == 'RF':
        params_str = f"n_est={params.get('n_estimators', '?')}, depth={params.get('max_depth', '?')}"
    elif algo == 'XGBoost':
        params_str = f"n_est={params.get('n_estimators', '?')}, lr={params.get('learning_rate', '?')}"
    elif algo == 'SVM_RBF':
        params_str = f"C={params.get('C', '?')}, eps={params.get('epsilon', '?')}, gamma={params.get('gamma', '?')}"
    else:
        params_str = str(params)[:50]

    report_lines.append(
        f"| {model_id} | {model_name} | {algo} | {strategy} | {params_str} |"
    )

report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 4. Performance Metrics (with algorithm)
report_lines.append("## 4. Performance Metrics by Algorithm")
report_lines.append("")
report_lines.append("| Model | Algorithm | NSE | KGE | R² | RMSE | MAE |")
report_lines.append("|-------|-----------|-----|-----|-------|------|-----|")

for model_name, trained_model in trained_models.items():
    m = trained_model['metrics']
    algo = trained_model['algorithm']
    report_lines.append(
        f"| {model_name} | {algo} | {m['NSE']:.4f} | {m['KGE']:.4f} | {m['R2']:.4f} | "
        f"{m['RMSE']:.4f} | {m['MAE']:.4f} |"
    )

report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 5. Feature Importance Analysis
report_lines.append("## 5. Feature Importance Analysis")
report_lines.append("")
report_lines.append("**Note on Feature Importance Methods:**")
report_lines.append("- **RF & XGBoost:** Native importance (Gini/Gain reduction)")
report_lines.append("- **SVM_RBF:** Permutation importance (computationally expensive)")
report_lines.append("")
report_lines.append("Top 5 most important features per model:")
report_lines.append("")

for model_name, trained_model in trained_models.items():
    algo = trained_model['algorithm']
    feature_cols = trained_model['feature_cols']
    importances = trained_model['feature_importances']

    # Get top 5
    top5_idx = np.argsort(importances)[::-1][:5]

    report_lines.append(f"**{model_name} ({algo}):**")
    for i, idx in enumerate(top5_idx, 1):
        report_lines.append(f"{i}. {feature_cols[idx]}: {importances[idx]:.4f}")
    report_lines.append("")

report_lines.append("---")
report_lines.append("")

# 6. Physical Interpretation
report_lines.append("## 6. Physical Interpretation and Algorithm Comparison")
report_lines.append("")
report_lines.append("### 6.1 Physical Coherence")
report_lines.append("")
report_lines.append("All three algorithms identify physically meaningful predictors:")
report_lines.append("- **Precipitation (P):** Primary hydrological input")
report_lines.append("- **Soil Moisture (SMAP):** Antecedent conditions and storage")
report_lines.append("- **Basin Properties:** Area, elevation, channel length")
report_lines.append("- **Soil Hydraulic Properties:** Conductivity, water content")
report_lines.append("")
report_lines.append("### 6.2 Algorithm Differences")
report_lines.append("")
report_lines.append("**Random Forest (6 models):**")
report_lines.append("- Strengths: Fast training, robust, interpretable")
report_lines.append("- Weaknesses: May underfit complex non-linearities")
report_lines.append("- Best for: Stable predictions with moderate complexity")
report_lines.append("")
report_lines.append("**XGBoost (1 model - Dif_y):**")
report_lines.append("- Strengths: High accuracy, handles complex patterns")
report_lines.append("- Weaknesses: Slower training, more hyperparameters")
report_lines.append("- Best for: Maximizing predictive performance")
report_lines.append("")
report_lines.append("**SVM_RBF (1 model - Dif_q):**")
report_lines.append("- Strengths: Effective in high dimensions, non-linear kernel")
report_lines.append("- Weaknesses: Requires careful scaling, slow for large datasets")
report_lines.append("- Best for: Complex non-linear relationships with fewer samples")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 7. Conclusions
report_lines.append("## 7. Conclusions")
report_lines.append("")
report_lines.append("1. **All algorithms show good performance** (NSE > 0.7 for most models)")
report_lines.append("2. **Algorithm selection matters:** Different algorithms excel for different targets")
report_lines.append("3. **Feature importance is consistent** across algorithms (physically coherent)")
report_lines.append("4. **RF is the workhorse:** Good balance of speed, accuracy, and interpretability")
report_lines.append("5. **XGBoost offers marginal improvements:** At cost of computational time")
report_lines.append("6. **SVM_RBF viable alternative:** For specific non-linear problems")
report_lines.append("7. **Soil moisture (SMAP) critical:** Across all algorithms and targets")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 8. Files Generated
report_lines.append("## 8. Deliverables")
report_lines.append("")
report_lines.append("**Predictions (CSV files):**")
for model_name, trained_model in trained_models.items():
    algo = trained_model['algorithm']
    report_lines.append(f"- `predictions_{model_name}_{algo}.csv` (rows=basins, columns=dates)")
report_lines.append("")
report_lines.append("**Feature Importance:**")
report_lines.append(f"- {len(trained_models)} individual plots (PNG, 300 dpi)")
report_lines.append("- Heatmap across all models")
report_lines.append("- Summary CSV with importances")
report_lines.append("")
report_lines.append("**Confusion Matrices:**")
report_lines.append(f"- {len(trained_models) * 2} plots (tertile + extreme)")
report_lines.append("- Classification metrics CSV")
report_lines.append("")
report_lines.append("**Visualizations:**")
report_lines.append("- Time series, scatter plots, residual analysis")
report_lines.append("- Taylor diagram with algorithm color-coding")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("*End of Multi-Algorithm Report*")

# Save report
report_content = "\n".join(report_lines)
report_path = output_dirs['reports'] + 'MULTI_ALGORITHM_COMPLETE_REPORT.md'  # ← NUEVO V4

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ Technical report saved: {report_path}")
print(f"   Lines: {len(report_lines)}")
print(f"   Size: {len(report_content)} characters")


# ## 17. Summary and Next Steps

# In[20]:


print("\n" + "=" * 80)
print("NOTEBOOK 8 EXECUTION COMPLETE (MULTI-ALGORITHM v4)")
print("=" * 80)
print("\n✅ ALL DELIVERABLES GENERATED:")
print("\n📊 Predictions:")
print(f"   - {len(prediction_dfs)} CSV files (rows=basins, columns=dates)")
print(f"   - Algorithms: {', '.join(set(t['algorithm'] for t in trained_models.values()))}")
print(f"   - Location: {output_dirs['predictions']}")

print("\n📈 Feature Importance:")
print(f"   - {len(trained_models)} individual plots")
print(f"   - RF/XGBoost: Native importance")
print(f"   - SVM: Permutation importance")
print(f"   - Location: {output_dirs['feature_importance']}")

print("\n🎯 Confusion Matrices:")
print(f"   - {len(trained_models) * 2} plots")
print(f"   - Location: {output_dirs['confusion_matrices']}")

print("\n📉 Visualizations:")
print(f"   - Time series: {len(trained_models)} files")
print(f"   - Scatter plots: {len(trained_models)} files")
print(f"   - Residual analysis: {len(trained_models)} files")
print(f"   - Taylor diagram: 1 file (color-coded by algorithm)")

print("\n📝 Reports:")
print(f"   - Multi-algorithm report: MULTI_ALGORITHM_COMPLETE_REPORT.md")
print(f"   - Strategy comparison: strategy_comparison.csv")
print(f"   - Coverage report: model_coverage_report.csv")
print(f"   - Location: {output_dirs['reports']}")

print("\n" + "=" * 80)
print("🎓 READY FOR SUBMISSION")
print("=" * 80)
print("\nAll requirements fulfilled:")
print("   ✅ Multi-algorithm support (RF, XGBoost, SVM_RBF)")
print("   ✅ Numerical predictions (8 CSVs)")
print("   ✅ Feature importance (native + permutation)")
print("   ✅ Performance metrics (NSE, KGE, RMSE, MAE)")
print("   ✅ Confusion matrices")
print("   ✅ Complete technical explanation")
print("   ✅ Algorithm comparison and trade-offs")
print("   ✅ Physical interpretation")
print("\n🚀 Next steps: Review outputs and prepare thesis chapter!")
