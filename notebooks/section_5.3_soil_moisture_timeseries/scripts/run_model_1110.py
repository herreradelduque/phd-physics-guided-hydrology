#!/usr/bin/env python3
"""
Script para ejecutar el modelo de prueba ymin (1110) con grid search.
Este script ejecuta SOLO el primer modelo para evaluar tiempo de computación.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime
import time

# Machine Learning
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

print("=" * 80)
print("EJECUTANDO MODELO DE PRUEBA: ymin (1110)")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================================================================
# FUNCIONES DE MÉTRICAS
# ===========================================================================
def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan

def calculate_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def calculate_pbias(y_true, y_pred):
    if np.sum(y_true) == 0:
        return np.nan
    return (np.sum(y_true - y_pred) / np.sum(y_true)) * 100

def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_all_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'NSE': calculate_nse(y_true, y_pred),
        'KGE': calculate_kge(y_true, y_pred),
        'PBIAS_%': calculate_pbias(y_true, y_pred),
        'MAPE_%': calculate_mape(y_true, y_pred)
    }

# ===========================================================================
# CARGA DE DATOS
# ===========================================================================
print("\n📁 Cargando datos...")
base_path = "data/clean/merged_data/"

join_dict = {}
for filename in os.listdir(base_path):
    if filename.endswith("-without-outliers.json"):
        basin_name = filename.replace("-without-outliers.json", "")
        df = pd.read_json(os.path.join(base_path, filename), orient="records")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        join_dict[basin_name] = df

print(f"✅ Cargadas {len(join_dict)} cuencas")

# Crear variables sintéticas
print("🔧 Creando variables sintéticas...")
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

# Dataset global
all_basins_data = []
for basin_name, df in join_dict.items():
    df_copy = df.copy()
    df_copy['Basin'] = basin_name
    all_basins_data.append(df_copy)

global_df = pd.concat(all_basins_data, ignore_index=True)
if 'Date' in global_df.columns:
    global_df = global_df.sort_values('Date').reset_index(drop=True)

print(f"✅ Dataset global: {len(global_df):,} observaciones")

# ===========================================================================
# MAPEO DE VARIABLES (CORREGIDO)
# ===========================================================================
var_mapping = {
    'P': 'P',
    'Av_qr': 'sm_rootzone_av',
    'Av_qs': 'sm_surface_av',
    'q0r': 'q0_r',
    'q0s': 'q0_s',
    'area_catch': 'area_cuenca',
    'chan_length': 'longitud_total',
    'CN_we': 'CN_ponderado',
    'CN_DT': 'CN_DT',
    'z_min': 'z_min',
    'z_max': 'z_max',
    'DT_ks': 'DT_ks',
    'mean_ks': 'mean_ks',
    'mean_thetas': 'mean_thetas',
    'DT_thetas': 'DT_thetas',
    'mean_thetar': 'mean_thetar',
    'DT_thetar': 'DT_thetar',
    'mean_thau': 'mean_thau',
    'DT_thau': 'DT_thau',
    'ymin': 'ymin',
    'yav': 'yav',
    'ymax': 'ymax'
}

# ===========================================================================
# MODELO 1110 - ymin
# ===========================================================================
print("\n" + "=" * 80)
print("MODELO 1110: ymin (Eq 6 - Variables primarias con humedad promedio)")
print("=" * 80)

# Features para modelo 1110
features_paper = ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                  'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                  'area_catch', 'chan_length', 'z_min', 'z_max']

features_real = [var_mapping.get(v, v) for v in features_paper]
target_real = 'ymin'

print(f"Features: {len(features_real)}")
print(f"Target: {target_real}")

# Preparar datos
X = global_df[features_real]
y = global_df[target_real]

# Drop NaN
valid_idx = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_idx].values
y_clean = y[valid_idx].values

print(f"Datos limpios: {len(X_clean):,} muestras")

# ===========================================================================
# GRID DE HIPERPARÁMETROS
# ===========================================================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

total_combinations = 3 * 4 * 3 * 3 * 3 * 2
print(f"\n🔍 Grid Search: {total_combinations} combinaciones × 2 estrategias")

# ===========================================================================
# ESTRATEGIA 1: RANDOM SPLIT (Días aleatorios)
# ===========================================================================
print("\n" + "-" * 80)
print("ESTRATEGIA 1: Random Split (Días aleatorios)")
print("-" * 80)

start_time = time.time()

# Tipificar
scaler_random = StandardScaler()
X_scaled_random = scaler_random.fit_transform(X_clean)

# Grid Search
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
cv_random = KFold(n_splits=5, shuffle=True, random_state=42)

grid_random = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv_random,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

print("⏳ Ejecutando Grid Search (esto tomará ~15-30 minutos)...")
grid_random.fit(X_scaled_random, y_clean)

elapsed_random = time.time() - start_time

# Resultados
best_model_random = grid_random.best_estimator_
y_pred_random = best_model_random.predict(X_scaled_random)
metrics_random = calculate_all_metrics(y_clean, y_pred_random)

print(f"\n✅ Completado en {elapsed_random/60:.2f} minutos")
print(f"\nMejores hiperparámetros:")
for param, value in grid_random.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nMétricas (dataset completo):")
for metric, value in metrics_random.items():
    print(f"  - {metric}: {value:.6f}")

# ===========================================================================
# ESTRATEGIA 2: TEMPORAL SPLIT (Rangos temporales)
# ===========================================================================
print("\n" + "-" * 80)
print("ESTRATEGIA 2: Temporal Split (Rangos temporales completos)")
print("-" * 80)

start_temporal = time.time()

# Tipificar
scaler_temporal = StandardScaler()
X_scaled_temporal = scaler_temporal.fit_transform(X_clean)

# Grid Search
rf_temporal = RandomForestRegressor(random_state=42, n_jobs=-1)
cv_temporal = TimeSeriesSplit(n_splits=5)

grid_temporal = GridSearchCV(
    estimator=rf_temporal,
    param_grid=param_grid,
    cv=cv_temporal,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

print("⏳ Ejecutando Grid Search...")
grid_temporal.fit(X_scaled_temporal, y_clean)

elapsed_temporal = time.time() - start_temporal

# Resultados
best_model_temporal = grid_temporal.best_estimator_
y_pred_temporal = best_model_temporal.predict(X_scaled_temporal)
metrics_temporal = calculate_all_metrics(y_clean, y_pred_temporal)

print(f"\n✅ Completado en {elapsed_temporal/60:.2f} minutos")
print(f"\nMejores hiperparámetros:")
for param, value in grid_temporal.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nMétricas (dataset completo):")
for metric, value in metrics_temporal.items():
    print(f"  - {metric}: {value:.6f}")

# ===========================================================================
# RESUMEN Y COMPARACIÓN
# ===========================================================================
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(f"\n⏱️  TIEMPOS DE COMPUTACIÓN:")
print(f"  Random split:   {elapsed_random/60:6.2f} min")
print(f"  Temporal split: {elapsed_temporal/60:6.2f} min")
print(f"  Total:          {(elapsed_random + elapsed_temporal)/60:6.2f} min")

estimated_total = ((elapsed_random + elapsed_temporal) / 2) * 8
print(f"\n📈 TIEMPO ESTIMADO para 8 modelos: {estimated_total/60:.1f} min ({estimated_total/3600:.2f} horas)")

print(f"\n📊 COMPARACIÓN DE ESTRATEGIAS:")
print(f"\nRandom Split:")
print(f"  NSE:  {metrics_random['NSE']:.4f}")
print(f"  KGE:  {metrics_random['KGE']:.4f}")
print(f"  RMSE: {metrics_random['RMSE']:.4f}")

print(f"\nTemporal Split:")
print(f"  NSE:  {metrics_temporal['NSE']:.4f}")
print(f"  KGE:  {metrics_temporal['KGE']:.4f}")
print(f"  RMSE: {metrics_temporal['RMSE']:.4f}")

# Guardar resultados
output_dir = "notebooks/nb_tesis/7-results/"
os.makedirs(output_dir, exist_ok=True)

results = {
    'model_id': 1110,
    'model_name': 'ymin',
    'execution_date': datetime.now().isoformat(),
    'random_split': {
        'elapsed_time_min': elapsed_random / 60,
        'best_params': grid_random.best_params_,
        'metrics': metrics_random
    },
    'temporal_split': {
        'elapsed_time_min': elapsed_temporal / 60,
        'best_params': grid_temporal.best_params_,
        'metrics': metrics_temporal
    },
    'estimated_time_all_models_hours': estimated_total / 3600
}

with open(os.path.join(output_dir, 'model_1110_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Resultados guardados en: {output_dir}model_1110_results.json")
print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
