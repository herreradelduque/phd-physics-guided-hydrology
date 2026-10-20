#!/usr/bin/env python3
"""
Script de verificación de configuración.
Ejecutar ANTES de los scripts 7 y 8 para verificar que todo está correcto.

Uso:
    python verificar_configuracion.py

Autor: Víctor Galán
Fecha: 2026-03-19
"""

import os
import sys

print("=" * 80)
print("VERIFICACIÓN DE CONFIGURACIÓN")
print("=" * 80)
print()

# ============================================================================
# CONFIGURACIÓN - CAMBIAR ESTOS PATHS POR LOS TUYOS
# ============================================================================

# Path a los datos (debe ser el mismo en scripts 7 y 8)
DATA_PATH = "../../data/clean/merged_data/"

# Path donde guardarás los resultados del script 7
SCRIPT7_OUTPUT = "./"

# Path donde guardarás los resultados del script 8
SCRIPT8_OUTPUT = "./8-results/"

print("PATHS CONFIGURADOS:")
print(f"  Datos:            {DATA_PATH}")
print(f"  Resultados script 7: {SCRIPT7_OUTPUT}")
print(f"  Resultados script 8: {SCRIPT8_OUTPUT}")
print()

# ============================================================================
# VERIFICACIÓN 1: Path a los datos
# ============================================================================

print("=" * 80)
print("1. VERIFICACIÓN DEL PATH A LOS DATOS")
print("=" * 80)

data_path_abs = os.path.abspath(DATA_PATH)
print(f"Path absoluto: {data_path_abs}")

if not os.path.exists(DATA_PATH):
    print("❌ ERROR: El directorio NO existe")
    print()
    print("SOLUCIÓN:")
    print("  1. Verificar que la ruta es correcta")
    print("  2. Descomprimir merged_data.zip si es necesario")
    print("  3. Cambiar DATA_PATH en este script (línea 23)")
    sys.exit(1)

print("✅ El directorio existe")
print()

# ============================================================================
# VERIFICACIÓN 2: Archivos JSON
# ============================================================================

print("=" * 80)
print("2. VERIFICACIÓN DE ARCHIVOS JSON")
print("=" * 80)

json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('-without-outliers.json')]

if not json_files:
    print("❌ ERROR: No se encontraron archivos *-without-outliers.json")
    print()
    print("SOLUCIÓN:")
    print("  1. Verificar que estás apuntando al directorio correcto")
    print("  2. Re-ejecutar notebook 0-ebro-data-cleaning.ipynb")
    sys.exit(1)

print(f"✅ Encontrados {len(json_files)} archivos JSON")
print(f"   Cuencas: {', '.join([f.replace('-without-outliers.json', '') for f in json_files[:5]])}...")
print()

# ============================================================================
# VERIFICACIÓN 3: Columnas en los datos
# ============================================================================

print("=" * 80)
print("3. VERIFICACIÓN DE COLUMNAS EN LOS DATOS")
print("=" * 80)

try:
    import pandas as pd
except ImportError:
    print("❌ ERROR: pandas no está instalado")
    print("   Instalar con: pip install pandas")
    sys.exit(1)

test_file = os.path.join(DATA_PATH, json_files[0])
print(f"Archivo de prueba: {json_files[0]}")

try:
    df = pd.read_json(test_file)
except Exception as e:
    print(f"❌ ERROR al leer el archivo: {e}")
    sys.exit(1)

print(f"   Total de columnas: {len(df.columns)}")
print(f"   Total de filas: {len(df)}")
print()

# Columnas críticas que DEBEN existir
critical_cols = {
    'Climáticas básicas': ['P', 'tav', 'tmax', 'tmin'],
    'Humedad relativa': ['Hrav', 'Hrmax', 'Hrmin'],
    'Viento': ['wv', 'wvmax'],
    'Radiación': ['SolarRad', 'et0'],
    'SMAP': ['sm_rootzone_av', 'sm_surface_av', 'q0_r', 'q0_s'],
    'Propiedades cuenca': ['area_cuenca', 'longitud_total', 'z_min', 'z_max'],
    'Variables objetivo': ['ymin', 'yav', 'ymax', 'qmin', 'qav', 'qmax']
}

all_ok = True
for category, cols in critical_cols.items():
    print(f"{category}:")
    for col in cols:
        exists = col in df.columns
        status = "✅" if exists else "❌"
        print(f"  {status} {col}")
        if not exists:
            all_ok = False
    print()

if not all_ok:
    print("=" * 80)
    print("❌ PROBLEMA DETECTADO: FALTAN COLUMNAS EN LOS DATOS")
    print("=" * 80)
    print()
    print("ESTE ES EL PROBLEMA QUE CAUSA EL ERROR:")
    print("  KeyError: \"['Hrmin', 'wvmax'] not in index\"")
    print()
    print("SOLUCIÓN:")
    print("  1. Re-ejecutar el notebook 0-ebro-data-cleaning.ipynb")
    print("  2. Verificar que estás usando la versión correcta de los datos")
    print()
    print("COLUMNAS DISPONIBLES EN EL ARCHIVO:")
    print(sorted(df.columns.tolist()))
    sys.exit(1)

print("=" * 80)
print("✅ TODAS LAS COLUMNAS NECESARIAS ESTÁN PRESENTES")
print("=" * 80)
print()

# ============================================================================
# VERIFICACIÓN 4: Buscar typos en nombres de columnas
# ============================================================================

print("=" * 80)
print("4. VERIFICACIÓN DE TYPOS EN NOMBRES DE COLUMNAS")
print("=" * 80)

# Buscar 'wymax' (typo) en lugar de 'wvmax'
if 'wymax' in df.columns:
    print("⚠️  ADVERTENCIA: Detectado 'wymax' en los datos")
    print("   Debería ser 'wvmax' (velocidad del viento máxima)")
    print("   Esto causará un KeyError en los scripts")
    print()
    all_ok = False
else:
    print("✅ No se detectaron typos conocidos")
    print()

# ============================================================================
# VERIFICACIÓN 5: Directorios de salida
# ============================================================================

print("=" * 80)
print("5. VERIFICACIÓN DE DIRECTORIOS DE SALIDA")
print("=" * 80)

script7_abs = os.path.abspath(SCRIPT7_OUTPUT)
script8_abs = os.path.abspath(SCRIPT8_OUTPUT)

print(f"Script 7 guardará en: {script7_abs}")
if os.path.exists(SCRIPT7_OUTPUT):
    print("  ✅ El directorio existe")
else:
    print("  ⚠️  El directorio NO existe (se creará automáticamente)")

print()
print(f"Script 8 guardará en: {script8_abs}")
if os.path.exists(SCRIPT8_OUTPUT):
    print("  ✅ El directorio existe")
else:
    print("  ⚠️  El directorio NO existe (se creará automáticamente)")

print()

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("=" * 80)
print("RESUMEN DE LA VERIFICACIÓN")
print("=" * 80)
print()

if all_ok:
    print("✅ ✅ ✅ VERIFICACIÓN EXITOSA ✅ ✅ ✅")
    print()
    print("Tu configuración está lista para ejecutar los scripts.")
    print()
    print("SIGUIENTE PASO:")
    print("  1. Actualizar los paths en 7-rf-hyperparameter-tuning_2.py:")
    print(f"     Línea 145: base_path = '{DATA_PATH}'")
    print()
    print("  2. Ejecutar el script 7:")
    print("     python 7-rf-hyperparameter-tuning_2.py")
    print()
    print("  3. Actualizar los paths en 8-rf-predictions-and-analysis_2.py:")
    print(f"     Línea 79:  BASE_PATH = '{DATA_PATH}'")
    print(f"     Línea 82:  RESULTS_PATH_NB7 = '{SCRIPT7_OUTPUT}'")
    print(f"     Línea 84:  OUTPUT_BASE = '{SCRIPT8_OUTPUT}'")
    print()
    print("  4. Ejecutar el script 8:")
    print("     python 8-rf-predictions-and-analysis_2.py")
else:
    print("❌ VERIFICACIÓN FALLIDA")
    print()
    print("Por favor, corrige los problemas detectados antes de ejecutar los scripts.")

print()
print("=" * 80)
