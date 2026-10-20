#!/usr/bin/env python3
"""
Script auxiliar para aplicar las modificaciones restantes al script 8_v3.py
Este script completa las modificaciones del PLAN_ADAPTACION_SCRIPT8.md

Uso:
    python aplicar_modificaciones_restantes_v3.py

Autor: Víctor Galán
Fecha: 2026-03-19
"""

import re

# Leer el script actual
with open('8-rf-predictions-and-analysis_3.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Aplicando modificaciones restantes al script v3...")
print()

# ============================================================================
# MODIFICACIÓN 1: Actualizar el loop de predicciones (después de línea ~768)
# ============================================================================

old_loop = r'''# Generate predictions for all models
print\("=" \* 80\)
print\("GENERATING PREDICTIONS BY BASIN"\)
print\("=" \* 80\)

prediction_dfs = \{\}

for model_name, trained_model in tqdm\(trained_models\.items\(\), desc="Generating predictions"\):
    pred_df = generate_predictions_by_basin\(trained_model, join_dict\)
    prediction_dfs\[model_name\] = pred_df

    # Save to CSV
    output_path = output_dirs\['predictions'\] \+ f'predictions_\{model_name\}\.csv'
    pred_df\.to_csv\(output_path\)

    print\(f"✅ \{model_name\}: \{pred_df\.shape\} -> \{output_path\}"\)

print\(f"\\n✅ Generated \{len\(prediction_dfs\)\} prediction files"\)'''

new_loop = '''# Generate predictions for all models
print("=" * 80)
print("GENERATING PREDICTIONS BY BASIN")
print("=" * 80)

prediction_dfs = {}
model_coverage_report = []  # ✅ NUEVO V3: Para reporte final

for model_name, trained_model in tqdm(trained_models.items(), desc="Generating predictions"):
    print(f"\\n{'='*60}")
    print(f"Model: {model_name} ({trained_model['model_spec']['id']})")
    print(f"{'='*60}")

    # ✅ NUEVO V3: Mostrar features requeridas
    print(f"   Required features: {len(trained_model['feature_cols'])}")
    if len(trained_model['feature_cols']) <= 10:
        print(f"   Features: {', '.join(trained_model['feature_cols'])}")
    else:
        print(f"   Features: {', '.join(trained_model['feature_cols'][:5])}... (+{len(trained_model['feature_cols'])-5} more)")

    pred_df = generate_predictions_by_basin(trained_model, join_dict)
    prediction_dfs[model_name] = pred_df

    # Save to CSV
    output_path = output_dirs['predictions'] + f'predictions_{model_name}.csv'
    pred_df.to_csv(output_path)

    # ✅ NUEVO V3: Guardar también reporte de cuencas excluidas
    if hasattr(pred_df, 'attrs') and pred_df.attrs.get('excluded_basins'):
        excluded_path = output_dirs['predictions'] + f'excluded_basins_{model_name}.json'
        with open(excluded_path, 'w') as f:
            json.dump(pred_df.attrs['excluded_basins'], f, indent=2)

    # ✅ NUEVO V3: Tracking para reporte
    model_coverage_report.append({
        'model_id': trained_model['model_spec']['id'],
        'model_name': model_name,
        'n_predictions': pred_df.shape[0],
        'n_total_basins': len(join_dict),
        'coverage_%': pred_df.shape[0] / len(join_dict) * 100 if join_dict else 0,
        'n_excluded': len(pred_df.attrs.get('excluded_basins', {}))
    })

    print(f"\\n   ✅ {model_name}: {pred_df.shape} -> {output_path}")

print(f"\\n✅ Generated {len(prediction_dfs)} prediction files")

# ✅ NUEVO V3: Guardar reporte de cobertura
coverage_df = pd.DataFrame(model_coverage_report)
coverage_path = output_dirs['predictions'] + 'model_coverage_report.csv'
coverage_df.to_csv(coverage_path, index=False)
print(f"✅ Coverage report saved: {coverage_path}")

# ✅ NUEVO V3: Mostrar resumen visual
print("\\n" + "=" * 80)
print("PREDICTION COVERAGE SUMMARY")
print("=" * 80)
print(coverage_df.to_string(index=False))'''

if re.search(old_loop, content, re.MULTILINE):
    content = re.sub(old_loop, new_loop, content, flags=re.MULTILINE)
    print("✅ Modificación 1: Loop de predicciones actualizado")
else:
    print("⚠️  Modificación 1: Patrón no encontrado (puede estar ya modificado)")

print()

# ============================================================================
# MODIFICACIÓN 2: Análisis de completitud después de cargar datos
# ============================================================================

# Buscar después de la línea que dice "print(f"   Date range: {global_df['Date'].min()} to {global_df['Date'].max()}")"
completeness_analysis = '''

# ============================================================================
# NUEVO EN V3: Análisis de Completitud de Columnas por Cuenca
# ============================================================================

print("=" * 80)
print("DATA COMPLETENESS ANALYSIS")
print("=" * 80)

# Obtener todas las columnas únicas en todas las cuencas
all_possible_cols = set()
for df in join_dict.values():
    all_possible_cols.update(df.columns)

print(f"Total de columnas únicas en todas las cuencas: {len(all_possible_cols)}")
print()

# Analizar cuenca por cuenca
basin_column_report = {}
for basin_name, df in join_dict.items():
    basin_column_report[basin_name] = {
        'n_cols': len(df.columns),
        'cols': set(df.columns),
        'n_rows': len(df)
    }

# Identificar columnas que NO todas las cuencas tienen
incomplete_cols = []
for col in all_possible_cols:
    basins_with_col = [b for b, info in basin_column_report.items()
                       if col in info['cols']]
    if len(basins_with_col) < len(join_dict):
        incomplete_cols.append({
            'column': col,
            'present_in': len(basins_with_col),
            'total_basins': len(join_dict),
            'coverage': len(basins_with_col) / len(join_dict) * 100
        })

if incomplete_cols:
    print(f"⚠️  Detectadas {len(incomplete_cols)} columnas con cobertura incompleta:")
    incomplete_cols_sorted = sorted(incomplete_cols, key=lambda x: x['coverage'])
    for item in incomplete_cols_sorted[:10]:  # Mostrar las 10 con menor cobertura
        print(f"   - {item['column']}: {item['present_in']}/{item['total_basins']} cuencas ({item['coverage']:.1f}%)")
    if len(incomplete_cols) > 10:
        print(f"   ... and {len(incomplete_cols) - 10} more")
else:
    print("✅ Todas las cuencas tienen las mismas columnas")

print()

# ============================================================================


'''

# Insertar después de la línea del date range
insert_after = 'print(f"📅 Date range: {global_df[\'Date\'].min()} to {global_df[\'Date\'].max()}")'
if insert_after in content:
    content = content.replace(insert_after, insert_after + completeness_analysis)
    print("✅ Modificación 2: Análisis de completitud añadido")
else:
    print("⚠️  Modificación 2: Punto de inserción no encontrado")

print()

# Guardar el archivo modificado
with open('8-rf-predictions-and-analysis_3.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("=" * 80)
print("MODIFICACIONES APLICADAS")
print("=" * 80)
print("✅ Script 8-rf-predictions-and-analysis_3.py actualizado")
print()
print("Modificaciones aplicadas:")
print("  1. ✅ Loop de predicciones con reporting detallado")
print("  2. ✅ Análisis de completitud de columnas por cuenca")
print()
print("El script está listo para ser ejecutado.")
