#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notebook 3 — Modelos Predictivos
Ejecuta todos los modelos predictivos con clusters predefinidos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print('Warning: XGBoost no disponible')

try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False
    print('Warning: CatBoost no disponible')

warnings.filterwarnings('ignore')

# Configuración
PROC_PATH = '../../data/processed/'
OUT_PATH = '../output/3-predictive-models/'
os.makedirs(OUT_PATH, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

print('Setup completo.')
print(f'XGBoost disponible: {XGBOOST_OK}')
print(f'CatBoost disponible: {CATBOOST_OK}')

# Clusters predefinidos
CLUSTERS_CN = {
    0: ['FlamisellCabdella', 'VallfarreraAlins'],
    1: ['IzalzuAnduña', 'BailinSabiñanigo', 'UrriobiEspinal',
        'UbaguaRiezu', 'ZatoyaOchagavia', 'ZidacosGarinoiain'],
    2: ['DezaEmbidDeAriza', 'GaronaBossost', 'IsuelaTrasobares',
        'NelaVillarcayo', 'OroncilloOron', 'RudronValdelateja',
        'SanguesaOnsella', 'TironSanMiguelPedroso', 'TruebaMedinaDePomar',
        'CidacosYanguas', 'SubialdeLarrinoa']
}

CLUSTERS_GA = {
    0: ['OroncilloOron', 'FlamisellCabdella', 'GaronaBossost', 'TruebaMedinaDePomar'],
    1: ['IzalzuAnduña', 'DezaEmbidDeAriza', 'NelaVillarcayo', 'RudronValdelateja',
        'SanguesaOnsella', 'SubialdeLarrinoa', 'TironSanMiguelPedroso',
        'UrriobiEspinal', 'UbaguaRiezu', 'ZatoyaOchagavia', 'ZidacosGarinoiain',
        'BailinSabiñanigo', 'IsuelaTrasobares', 'VallfarreraAlins', 'CidacosYanguas']
}

CUENCA_MAP = {
    'Anduña': 'IzalzuAnduña',
    'Bailin': 'BailinSabiñanigo',
    'Deza': 'DezaEmbidDeAriza',
    'Flamisell': 'FlamisellCabdella',
    'Garona': 'GaronaBossost',
    'Isuela': 'IsuelaTrasobares',
    'Nela': 'NelaVillarcayo',
    'Oroncillo': 'OroncilloOron',
    'Rudron': 'RudronValdelateja',
    'Sangüesa': 'SanguesaOnsella',
    'Subialde': 'SubialdeLarrinoa',
    'Tiron': 'TironSanMiguelPedroso',
    'Trueba': 'TruebaMedinaDePomar',
    'Ubagua': 'UbaguaRiezu',
    'Urrobi': 'UrriobiEspinal',
    'Vallfarrera': 'VallfarreraAlins',
    'Yanguas': 'CidacosYanguas',
    'Zatoya': 'ZatoyaOchagavia',
    'Zidacos': 'ZidacosGarinoiain'
}

def create_cluster_map(clusters_dict):
    mapping = {}
    for cluster_id, cuencas in clusters_dict.items():
        for cuenca in cuencas:
            mapping[cuenca] = cluster_id
    return mapping

CN_MAP = create_cluster_map(CLUSTERS_CN)
GA_MAP = create_cluster_map(CLUSTERS_GA)

print(f'\nClusters CN definidos: {len(CLUSTERS_CN)} clusters')
for k, v in CLUSTERS_CN.items():
    print(f"  Cluster {k}: {len(v)} cuencas")

print(f'\nClusters GA definidos: {len(CLUSTERS_GA)} clusters')
for k, v in CLUSTERS_GA.items():
    print(f"  Cluster {k}: {len(v)} cuencas")

# Métricas
def nse(y_true, y_pred):
    num = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    den = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return float(1 - num / den) if den > 0 else np.nan

def kge(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-12)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

def pbias_pct(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    return float(100 * np.sum(y_true - y_pred) / denom) if denom > 0 else np.nan

def mape_pct(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-9
    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.sum() > 0 else np.nan

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'NSE': nse(y_true, y_pred),
        'KGE': kge(y_true, y_pred),
        'PBIAS_pct': pbias_pct(y_true, y_pred),
        'MAPE_pct': mape_pct(y_true, y_pred),
    }

print('\nMetricas definidas: R2, RMSE, MAE, NSE, KGE, PBIAS_pct, MAPE_pct')

# Modelos
def get_models():
    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        'SVR (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(64, 32), activation='relu',
                max_iter=500, random_state=RANDOM_STATE, early_stopping=True
            ))
        ]),
    }

    if XGBOOST_OK:
        models['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            ))
        ])

    if CATBOOST_OK:
        models['CatBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=5,
                random_state=RANDOM_STATE, verbose=0
            ))
        ])

    return models

MODEL_NAMES = list(get_models().keys())
print(f'\nAlgoritmos configurados ({len(MODEL_NAMES)}): {MODEL_NAMES}')

# Configuraciones de modelos
MODEL_SPECS = [
    {'id': 'M1', 'features': ['DuracionP'], 'descripcion': 'Tiempos=f(Duración de la tormenta)'},
    {'id': 'M2', 'features': ['VolumenP'], 'descripcion': 'Tiempos = f(Volumen de Lluvia)'},
    {'id': 'M3', 'features': ['MaximoR'], 'descripcion': 'Tiempos = f(Escorrentía Máxima)'},
    {'id': 'M4', 'features': ['MaximoR_VolumenP'], 'descripcion': 'Tiempos = f(Escorrentía Máxima / Volumen de Lluvia)'},
    {'id': 'M5', 'features': ['VolumenR_VolumenP'], 'descripcion': 'Tiempos = f(Volumen Escorrentía / Volumen de Lluvia)'},
    {'id': 'M6', 'features': ['MaximoP'], 'descripcion': 'Tiempos = f(Intensidad Máxima de Precipitación)'},
    {'id': 'M7', 'features': ['MaximoQ'], 'descripcion': 'Tiempos = f(Caudal Pico)'},
    {'id': 'M8', 'features': ['VolumenP', 'MaximoP'], 'descripcion': 'Tiempos = f(Volumen de Lluvia, Intensidad Máxima de Precipitación)'},
]

print(f'\nConfiguraciones: {len(MODEL_SPECS)}')
for m in MODEL_SPECS:
    print(f"  {m['id']}: {m['descripcion']}  features={m['features']}")

# Cargar datos
print('\n=== CARGANDO DATOS ===')
df_raw = pd.read_csv(os.path.join(PROC_PATH, 'events_dataset.csv'))

print(f'Dataset cargado: {df_raw.shape}')
print(f'Cuencas en dataset: {sorted(df_raw["cuenca"].unique())}')

# Mapear y asignar clusters
df_raw['cuenca_completa'] = df_raw['cuenca'].map(CUENCA_MAP)
df_raw['cluster_cn'] = df_raw['cuenca_completa'].map(CN_MAP)
df_raw['cluster_ga'] = df_raw['cuenca_completa'].map(GA_MAP)

print(f'Clusters CN asignados: {sorted(df_raw["cluster_cn"].dropna().unique())}')
print(f'Clusters GA asignados: {sorted(df_raw["cluster_ga"].dropna().unique())}')

# Renombrar columnas
df_all = df_raw.rename(columns={
    'tiempos': 'Tiempos',
    'duracion_p': 'DuracionP',
    'duracion_q': 'DuracionQ',
    'maximo_p': 'MaximoP',
    'maximo_q': 'MaximoQ',
    'maximo_r': 'MaximoR',
    'volumen_p': 'VolumenP',
    'volumen_q': 'VolumenQ',
    'volumen_r': 'VolumenR',
    'ratio_r_p': 'RatioR_P',
    'ratio_vr_vp': 'VolumenR_VolumenP'
}).copy()

# Crear feature derivada
df_all['MaximoR_VolumenP'] = df_all['MaximoR'] / (df_all['VolumenP'] + 1e-12)

print(f'\nDataset preparado: {df_all.shape}')
print(f'Total eventos: {len(df_all):,}')

# Función de entrenamiento
def train_and_evaluate(df_subset, feature_cols, target_col='Tiempos',
                       model_pipeline=None, n_folds=5):
    df_valid = df_subset[[target_col] + feature_cols].dropna()

    if len(df_valid) < n_folds * 2:
        return {
            'n_samples': len(df_valid),
            'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan,
            'NSE': np.nan, 'KGE': np.nan, 'PBIAS_pct': np.nan, 'MAPE_pct': np.nan,
            'error': 'insufficient_data'
        }

    X = df_valid[feature_cols].values
    y = df_valid[target_col].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    metrics_per_fold = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model_pipeline.fit(X_tr, y_tr)
        y_pred = model_pipeline.predict(X_val)
        metrics_per_fold.append(compute_metrics(y_val, y_pred))

    avg = {k: float(np.nanmean([m[k] for m in metrics_per_fold])) for k in metrics_per_fold[0]}
    avg['n_samples'] = len(df_valid)
    return avg

print('\nFunción de entrenamiento definida.')

# Construir ámbitos
print('\n=== CONSTRUYENDO ÁMBITOS ===')
scopes = {}
scopes['Global'] = df_all.copy()

for c in sorted(df_all['cluster_cn'].dropna().unique()):
    label = f'CN_Cluster{int(c)}'
    scopes[label] = df_all[df_all['cluster_cn'] == c].copy()

for c in sorted(df_all['cluster_ga'].dropna().unique()):
    label = f'GA_Cluster{int(c)}'
    scopes[label] = df_all[df_all['cluster_ga'] == c].copy()

print('Ambitos de entrenamiento:')
for label, subset in scopes.items():
    print(f'  {label:20s}: {len(subset):,} eventos')

# Ejecutar modelos
print('\n=== EJECUTANDO MODELOS ===')
all_results = []

total_count = 0
for spec in MODEL_SPECS:
    for scope_label in scopes.keys():
        total_count += len(MODEL_NAMES)

print(f'Total combinaciones a ejecutar: {total_count}')
print(f'  {len(MODEL_SPECS)} configs × {len(scopes)} ámbitos × {len(MODEL_NAMES)} algoritmos\n')

pbar = tqdm(total=total_count, desc='Modelos')

for spec in MODEL_SPECS:
    for scope_label, df_scope in scopes.items():
        for model_name, model_pipeline in get_models().items():
            result = train_and_evaluate(
                df_subset=df_scope,
                feature_cols=spec['features'],
                target_col='Tiempos',
                model_pipeline=model_pipeline,
                n_folds=N_FOLDS
            )

            record = {
                'Config': spec['id'],
                'Descripción': spec['descripcion'],
                'Features': ', '.join(spec['features']),
                'Ámbito': scope_label,
                'Algoritmo': model_name,
                'n_samples': result.get('n_samples', 0),
                'R²': result.get('R2', np.nan),
                'RMSE': result.get('RMSE', np.nan),
                'MAE': result.get('MAE', np.nan),
                'NSE': result.get('NSE', np.nan),
                'KGE': result.get('KGE', np.nan),
                'PBIAS_%': result.get('PBIAS_pct', np.nan),
                'MAPE_%': result.get('MAPE_pct', np.nan),
            }
            all_results.append(record)
            pbar.update(1)

pbar.close()

results_df = pd.DataFrame(all_results)
print(f'\n✓ Resultados completos: {results_df.shape}')

# Guardar resultados
print('\n=== GUARDANDO RESULTADOS ===')
results_df['Num.'] = range(1, len(results_df) + 1)

results_output = results_df[[
    'Num.', 'Config', 'Descripción', 'Features', 'Ámbito', 'Algoritmo',
    'R²', 'RMSE', 'MAE', 'NSE', 'KGE', 'PBIAS_%', 'MAPE_%', 'n_samples'
]].copy()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUT_PATH, f'results_complete_{timestamp}.csv')
results_output.to_csv(csv_path, index=False, sep=';', encoding='utf-8')
print(f'✓ Tabla completa guardada en: {csv_path}')

template_path = os.path.join(OUT_PATH, 'results_complete.csv')
results_output.to_csv(template_path, index=False, sep=';', encoding='utf-8')
print(f'✓ Resultados guardados en formato template: {template_path}')

print(f'\n✓✓✓ Total de modelos ejecutados: {len(results_output)}')
print('\n=== RESUMEN ===')
print('\nMétricas promedio por algoritmo:')
print(results_df.groupby('Algoritmo')[['R²', 'RMSE', 'NSE', 'KGE', 'PBIAS_%']].mean().round(3))

print('\n=== EJECUCIÓN COMPLETADA ===')
