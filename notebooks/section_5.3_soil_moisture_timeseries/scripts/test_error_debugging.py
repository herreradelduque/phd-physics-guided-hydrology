#!/usr/bin/env python3
"""
Script de debugging para identificar la causa exacta del KeyError.
Ejecutar este script EN LUGAR del script 8 para diagnosticar el problema.

Simula el comportamiento del script 8 paso a paso con output de debug.

Uso:
    python test_error_debugging.py

Autor: Víctor Galán
Fecha: 2026-03-19
"""

import pandas as pd
import numpy as np
import os
import sys

print("=" * 80)
print("DEBUGGING SCRIPT: Identificación de causa del KeyError")
print("=" * 80)
print()

# Configuración de paths (ajustar según servidor de Sergio)
BASE_PATH = "../../data/clean/merged_data/"

print("1. VERIFICACIÓN DE PATH Y CARGA DE DATOS")
print("-" * 80)
print(f"BASE_PATH: {BASE_PATH}")
print(f"Absoluto: {os.path.abspath(BASE_PATH)}")
print(f"Existe: {os.path.exists(BASE_PATH)}")
print()

if not os.path.exists(BASE_PATH):
    print(f"❌ ERROR: Path {BASE_PATH} no existe")
    print("Ajustar BASE_PATH en este script según tu configuración")
    sys.exit(1)

# Cargar datos (igual que script 8)
join_dict = {}
for filename in os.listdir(BASE_PATH):
    if filename.endswith("-without-outliers.json"):
        basin_name = filename.replace("-without-outliers.json", "")
        df = pd.read_json(os.path.join(BASE_PATH, filename), orient="records")

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        join_dict[basin_name] = df

print(f"✅ Cargadas {len(join_dict)} cuencas")
print()

# Verificar columnas ANTES de crear variables sintéticas
print("2. COLUMNAS EN LOS DATOS ORIGINALES (ANTES de variables sintéticas)")
print("-" * 80)

test_basin = list(join_dict.keys())[0]
print(f"Cuenca de prueba: {test_basin}")
print(f"Total columnas: {len(join_dict[test_basin].columns)}")
print()

required_base_cols = ['Hrmin', 'Hrmax', 'wv', 'wvmax']
for col in required_base_cols:
    exists = col in join_dict[test_basin].columns
    status = "✅" if exists else "❌"
    print(f"{status} {col}: {exists}")

print()
print("Primeras 20 columnas:")
print(sorted(join_dict[test_basin].columns.tolist())[:20])
print()

# Verificar si hay variantes de nombres
print("3. BÚSQUEDA DE VARIANTES DE NOMBRES DE COLUMNAS")
print("-" * 80)

all_cols = join_dict[test_basin].columns.tolist()
hr_variants = [col for col in all_cols if 'hr' in col.lower() and 'min' in col.lower()]
wv_variants = [col for col in all_cols if 'w' in col.lower() and 'max' in col.lower()]

print(f"Variantes de 'Hrmin': {hr_variants}")
print(f"Variantes de 'wvmax': {wv_variants}")
print()

# Crear variables sintéticas (MÉTODO ORIGINAL del script 8)
print("4. CREACIÓN DE VARIABLES SINTÉTICAS (MÉTODO ORIGINAL)")
print("-" * 80)

print("Método usado en el script 8:")
print("  for basin_name, df in join_dict.items():")
print("      df['Dif_HR'] = ...")
print("      join_dict[basin_name] = df  # ⚠️ Reasignación")
print()

for basin_name, df in join_dict.items():
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['Dif_t'] = df['tmax'] - df['tmin']
    if 'Hrmax' in df.columns and 'Hrmin' in df.columns:
        df['Dif_HR'] = df['Hrmax'] - df['Hrmin']
        print(f"  ✅ {basin_name}: Creada Dif_HR (Hrmax={df['Hrmax'].notna().sum()}, Hrmin={df['Hrmin'].notna().sum()})")
    else:
        print(f"  ❌ {basin_name}: NO se pudo crear Dif_HR (falta Hrmax o Hrmin)")

    if 'qmax' in df.columns and 'qmin' in df.columns:
        df['Dif_q'] = df['qmax'] - df['qmin']\n    if 'ymax' in df.columns and 'ymin' in df.columns:
        df['Dif_y'] = df['ymax'] - df['ymin']
    if 'sm_rootzone_max' in df.columns and 'sm_rootzone_min' in df.columns:
        df['Dif_qr'] = df['sm_rootzone_max'] - df['sm_rootzone_min']
    if 'sm_surface_max' in df.columns and 'sm_surface_min' in df.columns:
        df['Dif_qs'] = df['sm_surface_max'] - df['sm_surface_min']

    join_dict[basin_name] = df

print()
print("Resultado: Variables sintéticas creadas")
print()

# Verificar columnas DESPUÉS de crear variables sintéticas
print("5. VERIFICACIÓN CRÍTICA: ¿Se mantienen las columnas en join_dict?")
print("-" * 80)

print(f"Cuenca de prueba: {test_basin}")
for col in required_base_cols:
    exists = col in join_dict[test_basin].columns
    status = "✅" if exists else "❌ PROBLEMA DETECTADO"
    print(f"{status} {col}: {exists}")

# Verificar variables sintéticas
synthetic_cols = ['Dif_t', 'Dif_HR', 'Dif_q', 'Dif_y', 'Dif_qr', 'Dif_qs']
print()
print("Variables sintéticas:")
for col in synthetic_cols:
    exists = col in join_dict[test_basin].columns
    status = "✅" if exists else "⚠️ "
    print(f"{status} {col}: {exists}")

print()

# Crear global_df (igual que script 8)
print("6. CREACIÓN DE global_df")
print("-" * 80)

all_basins_data = []
for basin_name, df in join_dict.items():
    df_copy = df.copy()
    df_copy['Basin'] = basin_name
    all_basins_data.append(df_copy)

global_df = pd.concat(all_basins_data, ignore_index=True)

print(f"✅ global_df creado: {global_df.shape}")
print()

# Comparar columnas
print("7. COMPARACIÓN: global_df vs join_dict")
print("-" * 80)

print(f"Columnas en global_df: {len(global_df.columns)}")
print(f"Columnas en join_dict[{test_basin}]: {len(join_dict[test_basin].columns)}")
print()

for col in required_base_cols:
    in_global = col in global_df.columns
    in_dict = col in join_dict[test_basin].columns
    match = "✅ MATCH" if in_global == in_dict else "❌ DIFERENCIA"
    print(f"{match} {col}: global_df={in_global}, join_dict={in_dict}")

print()

# Simular la llamada problemática
print("8. SIMULACIÓN DEL ERROR (línea 586 del script 8)")
print("-" * 80)

# Features del modelo qmin (1131) que causa el error
feature_cols = [
    'P', 'Dif_qr', 'Dif_qs', 'tav', 'tmax', 'tmin',
    'Hrav', 'Hrmax', 'Hrmin',  # ⚠️ Hrmin requerido
    'wv', 'wvmax',              # ⚠️ wvmax requerido
    'SolarRad', 'mean_swcfc', 'DT_swcfc', 'mean_swcpwp', 'DT_swcpwp',
    'area_cuenca', 'longitud_total', 'z_min', 'z_max'
]

print(f"Intentando acceder a {len(feature_cols)} columnas")
print(f"Features requeridas: {feature_cols}")
print()

# Test con global_df
print("TEST 1: Acceso con global_df (como en retrain_best_model)")
try:
    X_global = global_df[feature_cols]
    print(f"  ✅ ÉXITO: global_df[feature_cols] funciona")
    print(f"     Shape: {X_global.shape}")
except KeyError as e:
    print(f"  ❌ ERROR: {e}")

print()

# Test con join_dict
print("TEST 2: Acceso con join_dict (como en generate_predictions_by_basin)")
for basin_name, df in list(join_dict.items())[:3]:  # Solo primeras 3 cuencas
    print(f"  Cuenca: {basin_name}")
    try:
        X = df[feature_cols]
        print(f"    ✅ ÉXITO: df[feature_cols] funciona")
        print(f"       Shape: {X.shape}")
    except KeyError as e:
        print(f"    ❌ ERROR ENCONTRADO: {e}")
        print(f"       Este es el error que ocurre en el script 8!")

        # Identificar columnas faltantes
        missing = [col for col in feature_cols if col not in df.columns]
        print(f"       Columnas faltantes: {missing}")

        # Buscar variantes
        for miss_col in missing:
            variants = [col for col in df.columns if miss_col.lower() in col.lower()]
            if variants:
                print(f"       Posibles variantes de '{miss_col}': {variants}")

        break  # Parar en el primer error

print()
print("=" * 80)
print("FIN DEL DEBUGGING")
print("=" * 80)
print()
print("CONCLUSIÓN:")
print("Si los TEST 1 y TEST 2 tienen resultados diferentes, confirma que")
print("el problema está en cómo se modifican los DataFrames en join_dict.")
print()
print("SIGUIENTE PASO:")
print("Revisar el archivo ANALISIS_ERROR_SOLO_CODIGO.md para las soluciones.")
