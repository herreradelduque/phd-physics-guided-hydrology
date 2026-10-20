import pandas as pd
import numpy as np
import json
import os

# Load one basin to verify columns
base_path = "data/clean/merged_data/"
files = [f for f in os.listdir(base_path) if f.endswith('-without-outliers.json')]
sample_file = files[0]

df = pd.read_json(os.path.join(base_path, sample_file), orient='records')

# Test the corrected mapping
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
    'DT_thau': 'DT_thau'
}

# Test ALL 8 models
models_to_test = [
    ('1110 ymin', ['P', 'Av_qr', 'Av_qs', 'DT_ks', 'mean_ks', 'mean_thetas', 'DT_thetas',
                   'mean_thetar', 'DT_thetar', 'mean_thau', 'DT_thau',
                   'area_catch', 'chan_length', 'z_min', 'z_max']),
    ('1123 yav', ['P', 'Av_qr', 'Av_qs', 'CN_we', 'CN_DT',
                  'area_catch', 'chan_length', 'z_min', 'z_max']),
]

print("=" * 70)
print("VERIFICACIÓN DE VARIABLES PARA TODOS LOS MODELOS")
print("=" * 70)

all_ok = True
for model_name, features_paper in models_to_test:
    features_real = [var_mapping.get(v, v) for v in features_paper]
    missing = [real for real in features_real if real not in df.columns]
    
    if missing:
        all_ok = False
        print(f"\n❌ {model_name}:")
        print(f"   Faltantes: {missing}")
        for m in missing:
            print(f"   Buscando '{m}'...")
            similar = [col for col in df.columns if m.lower()[:5] in col.lower()]
            if similar:
                print(f"      Posibles: {similar[:5]}")
    else:
        print(f"✅ {model_name}: OK ({len(features_real)} variables)")

print("\n" + "=" * 70)
if all_ok:
    print("✅ TODAS LAS VARIABLES ESTÁN DISPONIBLES!")
else:
    print("❌ HAY VARIABLES FALTANTES - VER ARRIBA")
print("=" * 70)
