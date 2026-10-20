# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 08:18:53 2025

@author: sergi
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Mostrar todas las columnas
pd.set_option("display.max_columns", None)

# (Opcional) Mostrar todas las filas
#pd.set_option("display.max_rows", None)

# (Opcional) Ajustar el ancho de cada columna para que no se corte
pd.set_option("display.max_colwidth", None)


import warnings
warnings.filterwarnings("ignore", message="Could not infer format")


# base_path_climate="C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/climate"
base_path_climate="/Users/sergiozubelzu/Library/CloudStorage/OneDrive-UniversidadPolitécnicadeMadrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/climate"

climate_dict_restored = {}

for filename in os.listdir(base_path_climate):
    if filename.endswith(".json"):
        # El nombre del archivo sin extensión
        basin_name = os.path.splitext(filename)[0]
        
        # Leer el JSON como DataFrame
        df = pd.read_json(os.path.join(base_path_climate, filename), orient="records")
        
        climate_dict_restored[basin_name] = df




#TRANSFORMO UNIDADES DE CAUDAL EN BAILÍN Y ME CARGO TRES VALORES DE Q DE SANGÜESA QUE SON DEL ORDEN DE 10E9
climate_dict_restored["Bailin"]["qmax"]=climate_dict_restored["Bailin"]["qmax"]/1000
climate_dict_restored["Bailin"]["qav"]=climate_dict_restored["Bailin"]["qav"]/1000
climate_dict_restored["Bailin"]["qmin"]=climate_dict_restored["Bailin"]["qmin"]/1000

for a in climate_dict_restored.keys():
    for d in range(0,len(climate_dict_restored[a]["qmax"])):
        if climate_dict_restored[a]["qmax"][d]>1000:
            
            climate_dict_restored[a]["qmax"][d]=0
            
for a in climate_dict_restored.keys():            
    for d in range(0,len(climate_dict_restored[a]["qav"])):
        if climate_dict_restored[a]["qav"][d]>1000:
           
            climate_dict_restored[a]["qav"][d]=0
            
for a in climate_dict_restored.keys():    
    for d in range(0,len(climate_dict_restored[a]["qmin"])):
        if climate_dict_restored[a]["qmin"][d]>1000:
            
            climate_dict_restored[a]["qmin"][d]=0
           




# base_path_physical="C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/physical"
base_path_physical="/Users/sergiozubelzu/Library/CloudStorage/OneDrive-UniversidadPolitécnicadeMadrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/physical"
physical_dict_restored = {}

for filename in os.listdir(base_path_physical):
    if filename.endswith(".json"):
        # El nombre del archivo sin extensión
        basin_name = os.path.splitext(filename)[0]
        
        # Leer el JSON como DataFrame
        df = pd.read_json(os.path.join(base_path_physical, filename), orient="records")
        
        physical_dict_restored[basin_name] = df



# base_path_humidity="C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/humidity"
# base_path_humidity="/Users/sergiozubelzu/Library/CloudStorage/OneDrive-UniversidadPolitécnicadeMadrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/basins/humidity"



# humidity_dict_restored = {}

# for filename in os.listdir(base_path_humidity):
#     if filename.endswith(".json"):
#         # El nombre del archivo sin extensión
#         basin_name = os.path.splitext(filename)[0]
        
#         # Leer el JSON como DataFrame
#         df = pd.read_json(os.path.join(base_path_humidity, filename), orient="records")
        
#         humidity_dict_restored[basin_name] = df

# tipos=["AM","PM"]
# humidity_dict_grouped={}

# print(humidity_dict_restored.items())

# for cuenca, df in humidity_dict_restored.items():
#     humidity_dict_grouped[cuenca] = {
#         'AM': df[df['tipo'] == 'AM'].copy(),
#         'PM': df[df['tipo'] == 'PM'].copy()
#     }


# base_path_merged="C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/merged_data"
# base_path_merged="/Users/sergiozubelzu/Library/CloudStorage/OneDrive-UniversidadPolitécnicadeMadrid/TesisVictor/CuencasEbro_20251209/ebro-flow-prediction-main/ebro-flow-prediction-main/data/clean/merged_data"
# join_dict_restored = {}

# for filename in os.listdir(base_path_merged):
#     if filename.endswith(".json"):
#         # El nombre del archivo sin extensión
#         basin_name = os.path.splitext(filename)[0]
        
#         # Leer el JSON como DataFrame
#         df = pd.read_json(os.path.join(base_path_merged, filename), orient="records")
        
#         join_dict_restored[basin_name] = df




    

#----------------------------------------------------------------------------------
# Boxplots de Q,Y y P incluyendo mínimo, medio y máximo de los primeros por cuencas
#Hay que activar un único par variable=[],a=, en el caso de q e y figsize=(12,6), para P figsize=(12,6), 
#----------------------------------------------------------------------------------
# variables = ["qmin", "qav", "qmax"]
# a="q"
# variables = ["ymin", "yav", "ymax"]
# a="y"
# variables = ["P"]
# a="P"
# for k, v in climate_dict_restored.items():
#     print("\n", k)
# # Suponiendo que tienes un DataFrame para una cuenca, por ejemplo:
#     df = climate_dict_restored[k]
    
    
#     plt.figure(figsize=(6, 6))
#     for i, var in enumerate(variables, 1):
#         if a=="P":
#             plt.subplot(1, 1, 1)
#         else:
#             plt.subplot(1, 3, i)
#         sns.boxplot(y=df[var], color="skyblue", fliersize=3, showfliers=False)  # fliersize controla los outliers
#         plt.title(k)
#     plt.tight_layout()
#     plt.savefig("graphs/"+a+k+"NO.jpg", format='jpg', dpi=600)
    # plt.show()

# for cuenca, grupos in humidity_dict_grouped.items():

#     print("\n", cuenca)

#     plt.figure(figsize=(10, 5))

#     for i, tipo in enumerate(tipos, 1):
#         df = grupos[tipo]["soil_moisture"]

#         plt.subplot(1, 2, i)
#         sns.boxplot(y=df, fliersize=3, showfliers=True)
#         plt.title(f"{cuenca} - {tipo}")
#         plt.ylabel("soil_moisture")

#     plt.tight_layout()
#     plt.savefig(f"graphs/SM_{cuenca}.jpg", format='jpg', dpi=600)
#     plt.show()

# for c in humidity_dict_grouped.keys():
#     print(humidity_dict_grouped[c]["AM"]["soil_moisture"].mean(),humidity_dict_grouped[c]["PM"]["soil_moisture"].mean())
#----------------------------------------------------------------------------------
#Recoloco el diccionario de clima y propiedades físicas y agrupo por variables y dentro de cada variable un df por cada cuenca
#Calculo alguna nueva variable, diferencia entre máximo y mínimo para q e y
#----------------------------------------------------------------------------------


# columns = ['Area_nm', 'Date', 'Hrav', 'Hrmax', 'Hrmin', 'Jdate', 'P', 
#            'SolarRad', 'Year', 'et0', 'qav', 'qmax', 'qmin',
#            'tav', 'tmax', 'tmin', 'wv', 'wvmax',
#            'yav', 'ymax', 'ymin']


# # Diccionario final
# climate_dict_derived = {col: {} for col in columns}

# for area_name, df in climate_dict_restored.items():
    
#     df = df.reset_index(drop=True).copy()

#     if "Date" in df.columns:
#         df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

#     for col in columns:
        
#         if col not in df.columns:
#             values = pd.Series([np.nan] * len(df))
#         else:
#             values = df[col].reset_index(drop=True)

#         # Guardamos la serie tal cual, SIN usar Date como índice
#         climate_dict_derived[col][area_name] = values

# # Convertimos cada variable en DataFrame conservando el orden
# for col in columns:
#     climate_dict_derived[col] = pd.DataFrame(climate_dict_derived[col])




# variables=[ 'Hrav', 'Hrmax', 'Hrmin', 'P', 'SolarRad',
#            'et0', 'qav', 'qmax', 'qmin', 'tav', 'tmax', 'tmin',
#            'wv', 'wvmax', 'yav', 'ymax', 'ymin',"Dif_y","Dif_q", "Dif_HR", "Dif_t"]

# if "ymax" in climate_dict_derived and "ymin" in climate_dict_derived:
#     hr_r_df = climate_dict_derived["ymax"] + climate_dict_derived["ymin"]
#     climate_dict_derived["Dif_y"] = hr_r_df
# if "qmax" in climate_dict_derived and "qmin" in climate_dict_derived:
#     hr_r_df = climate_dict_derived["qmax"] + climate_dict_derived["qmin"]
#     climate_dict_derived["Dif_q"] = hr_r_df

# if "Hrmax" in climate_dict_derived and "Hrmin" in climate_dict_derived:
#     HR_r_df = climate_dict_derived["Hrmax"] + climate_dict_derived["Hrmin"]
#     climate_dict_derived["Dif_HR"] = HR_r_df

# if "tmax" in climate_dict_derived and "tmin" in climate_dict_derived:
#     t_r_df = climate_dict_derived["tmax"] + climate_dict_derived["tmin"]
#     climate_dict_derived["Dif_t"] = t_r_df

#--------Empiezo con las propiedades 


props=['Area_nm', 'CN_DT', 'CN_ponderado', 'DT_ks', 'DT_swcfc', 'DT_swcpwp',
       'DT_thau', 'DT_thetar', 'DT_thetas', 'area_cuenca', 'ks_st', 'lat_max',
       'lat_min', 'lon_max', 'lon_min', 'longitud_total', 'mean_clay',
       'mean_ks', 'mean_sand', 'mean_silt', 'mean_swcfc', 'mean_swcpwp',
       'mean_thau', 'mean_thetar', 'mean_thetas', 'soil_type', 'swcfc_st',
       'swcpwp_st', 'thau_st', 'thetar_st', 'thetas_st', 'z_max', 'z_min']
physical_dict_derived = {p: {} for p in props}


for area_name, df in physical_dict_restored.items():
    
    df = df.reset_index(drop=True).copy()

    for p in props:
        
        if p not in df.columns:
            values = pd.Series([np.nan] * len(df))
        else:
            values = df[p].reset_index(drop=True)

        # Guardamos la serie tal cual, SIN usar Date como índice
        physical_dict_derived[p][area_name] = values

# Convertimos cada variable en DataFrame conservando el orden
for p in props:
    physical_dict_derived[p] = pd.DataFrame(physical_dict_derived[p])




#----------------------------------------------------------------------------------
# Boxplots de propiedades físicas para todas las cuencas
#----------------------------------------------------------------------------------

variables_physical=['CN_DT', 'CN_ponderado', 'DT_ks', 'DT_swcfc', 'DT_swcpwp',
       'DT_thau', 'DT_thetar', 'DT_thetas', 'area_cuenca', 'ks_st', 'longitud_total', 'mean_clay',
       'mean_ks', 'mean_sand', 'mean_silt', 'mean_swcfc', 'mean_swcpwp',
       'mean_thau', 'mean_thetar', 'mean_thetas', 'swcfc_st',
       'swcpwp_st', 'thau_st', 'thetar_st', 'thetas_st', 'z_max', 'z_min']

#-----Saco estadisticos descriptivos
estadisticas_fisicas = {}

for var in variables_physical:

    if var not in physical_dict_derived:
        print(f"Variable '{var}' no encontrada. Se ignora.")
        continue

    df = physical_dict_derived[var]

    # Convertir a una única serie (apilar todas las cuencas)
    serie = df.select_dtypes(include="number").stack().dropna()

    if serie.empty:
        print(f"Variable '{var}' no tiene datos numéricos.")
        continue

    estadisticas_fisicas[var] = {
        "mean": serie.mean(),
        "std": serie.std(),
        "range": serie.max() - serie.min(),
        "skew": serie.skew(),
        "kurt": serie.kurt()
    }

df_estadisticas_fisicas = pd.DataFrame(estadisticas_fisicas).T
pd.options.display.float_format = '{:.2f}'.format
print(df_estadisticas_fisicas)

#----Hago clusters

variables_GreenAmpt1=["mean_swcfc", "mean_swcpwp", "mean_thau", "mean_thetar", "mean_thetas","DT_ks", "DT_swcfc", "DT_swcpwp", "DT_thau", "DT_thetar", "DT_thetas", "area_cuenca", "longitud_total", "z_max", "z_min"]
variables_GreenAmpt2=["ks_st", "swcfc_st", "swcpwp_st", "thau_st", "thetar_st", "thetas_st","area_cuenca", "longitud_total", "z_max", "z_min"]
variables_GreenAmpt3=["mean_clay", "mean_ks", "mean_sand", "mean_silt" ,"area_cuenca", "longitud_total", "z_max", "z_min"]
variables_NumeroCurva=["CN_DT", "CN_ponderado","area_cuenca", "longitud_total", "z_max", "z_min"]
variables_physical=variables_GreenAmpt3
# variables_physical=variables_NumeroCurva
# Todas las cuencas
cuencas = physical_dict_derived[variables_physical[0]].columns

df_physical = pd.DataFrame(index=cuencas)

for var in variables_physical:
    if var not in physical_dict_derived:
        continue

    df = physical_dict_derived[var]

    # Si es numérica → media por cuenca
    if df.select_dtypes(include="number").shape[1] > 0:
        df_physical[var] = df.mean(axis=0)



X = df_physical.select_dtypes(include="number")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K = range(1, 10)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertias, 'o-')
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.show()

k = 2

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_clusters = df_physical.copy()
df_clusters["cluster"] = clusters

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(
    X_pca,
    index=df_physical.index,
    columns=["PC1", "PC2"]
)
df_pca["cluster"] = clusters

plt.figure(figsize=(7,6))

# Centroides en el espacio PCA
centroids_pca = pca.transform(kmeans.cluster_centers_)

df_centroids_pca = pd.DataFrame(
    centroids_pca,
    columns=["PC1", "PC2"]
)
df_centroids_pca["cluster"] = range(k)


plt.figure(figsize=(7,6))

# Cuencas
for c in sorted(df_pca["cluster"].unique()):
    subset = df_pca[df_pca["cluster"] == c]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        s=80, alpha=0.7,
        label=f"Cluster {c}"
    )

# Centroides
plt.scatter(
    df_centroids_pca["PC1"],
    df_centroids_pca["PC2"],
    s=300,
    c="black",
    marker="x",
    label="Centroides"
)

# Etiquetas de cuencas
for cuenca in df_pca.index:
    plt.text(
        df_pca.loc[cuenca, "PC1"],
        df_pca.loc[cuenca, "PC2"],
        cuenca,
        fontsize=8,
        alpha=0.7
    )

# Etiquetas de centroides
for i, row in df_centroids_pca.iterrows():
    plt.text(
        row["PC1"],
        row["PC2"],
        f"C{i}",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        color="white"
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


cuencas_por_cluster = {}

for cluster_id in sorted(df_pca["cluster"].unique()):
    cuencas_por_cluster[cluster_id] = (
        df_pca[df_pca["cluster"] == cluster_id]
        .index
        .tolist()
    )

# Mostrar resultado
for c, cuencas in cuencas_por_cluster.items():
    print(f"Cluster {c}:")
    print(cuencas)
    print()
# pca ya ajustado
loadings = pd.DataFrame(
    pca.components_.T,
    index=variables_physical,
    columns=[f"PCA{i+1}" for i in range(pca.n_components_)]
)


# print(loadings[["PCA1", "PCA2"]])


contribucion = loadings.abs().div(loadings.abs().sum(axis=0), axis=1) * 100
contribucion[["PCA1", "PCA2"]].sort_values("PCA1", ascending=False)
print(contribucion)





# print(X)
#----------------------------------------------------------------------------------
# Boxplots de Q,Y y P en un único gráfico todas las cuencas
#----------------------------------------------------------------------------------
# for v in variables:
#     if v in climate_dict_derived:
#         plt.figure(figsize=(14, 6))
#         climate_dict_derived[v].boxplot(grid=False, rot=45,showfliers=False)
#         plt.suptitle("")  # quitar título automático de pandas
#         plt.xlabel("Cuenca")
#         plt.ylabel(v)
#         plt.tight_layout()
#         plt.savefig("graphs/"+v+"NO.jpg", format='jpg', dpi=600)
#         plt.show()

# def mediamovil(vector,orden):
#     mm=vector
#     for i in range(orden,len(vector)):
#         mm[i]=vector.iloc[i-orden:i].sum()/orden
#     return mm


#----------------------------------------------------------------------------------
# Gráficos de valores junto con medias móviles
#----------------------------------------------------------------------------------
# ordenes=[1,2,3,4,8,16,32]
# colores=["yellow","blue", "grey", "red","green","red","orange", "purple"]
# climate_mediasmoviles = {orden: {var: {} for var in variables} for orden in ordenes}
# # variables=["ymax"]
# OrdenesImprimir=[8,32]

# # print(climate_dict_derived["P"])

# for var in variables:

#     # comprobar existencia de la variable en climate_dict_derived
#     if var not in climate_dict_derived:
#         print(f"Variable '{var}' no encontrada en climate_dict_derived. Se ignora.")
#         continue
    
#     df_var = climate_dict_derived[var]  # DataFrame ancho: index = Date, columns = cuencas
    
#     # Guard: si df_var no es DataFrame o está vacío
#     if not isinstance(df_var, pd.DataFrame) or df_var.empty:
#         print(f"Variable '{var}' vacía o no DataFrame. Se ignora.")
#         continue
    
    
#     df_numeric = df_var.apply(pd.to_numeric, errors='coerce')

#     for orden in ordenes:
    
#             if orden <= 0:
#                 raise ValueError("Los valores de 'ordenes' deben ser enteros positivos.")
#             df_numeric = df_var.apply(pd.to_numeric, errors='coerce')
#             mm = df_numeric.rolling(orden, min_periods=orden).mean()

#             # mm ya tiene NaN en las primeras `orden` posiciones → EXACTAMENTE lo que quieres.

#             # Guardamos por cuencas como Series
#             for cuenca in mm.columns:
#                 serie_mm = mm[cuenca]
#                 serie_mm.name = cuenca
#                 climate_mediasmoviles[orden][var][cuenca] = serie_mm
# print("Esto es",climate_mediasmoviles[2]["HRmin"]["Flamisell"])
# for var in variables:
#     for cuenca in climate_dict_restored.keys():
#         figMM1 = plt.figure(figsize=(14,6))
#         ax1 = figMM1.add_subplot(111)
#         for orden in OrdenesImprimir:
#             ax1.plot(climate_mediasmoviles[orden][var][cuenca], '-', lw=2, color=colores[ordenes.index(orden)], label='Medias Móviles orden '+str(orden))
#         ax1.plot(climate_dict_derived[var][cuenca], 'o', ms=6, mfc="cyan", mec="blue", alpha=0.2, label='Datos Diarios')
#         # ax1.plot(df_final["caudal_diario"].tolist(), CurvaLinealAjustadaN2[0], '-.', lw=2, color="blue", label='Ajuste cuadrático')
#         # ax1.legend(bbox_to_anchor=(-0.05,1.15), loc='upper left',ncol=1)
#         ax1.legend(loc='best')
#         ax1.set_ylabel(str(var))
#         ax1.set_title(cuenca)
#         # ax1.set_xlabel("Caudal río Pisuerga ($m^{3}/s$)")
#         ax1.grid(True)
#         plt.tight_layout()
#         plt.show()
#         figMM1.savefig("graphs/MM"+var+cuenca+".jpg", format='jpg', dpi=600)
    

# print(climate_mediasmoviles[orden][var][cuenca])


#----------------------------------------------------------------------------------
# Gráficos de distribución de lags
#----------------------------------------------------------------------------------
# lags=ordenes
# colores=["yellow","green", "grey", "red","blue","orange","black", "purple"]
# climate_lags = {lag: {var: {} for var in variables} for lag in lags}
# # variables=["ymax"]
# LagsImprimir=OrdenesImprimir


# for lag in lags:
#     for var in variables:
#         if var not in climate_dict_derived:
#             print(f"Variable '{var}' no encontrada en climate_dict_derived. Se ignora.")
#             continue

#         for cuenca in climate_dict_derived[var].keys():

#             serie = climate_dict_derived[var][cuenca]

#             # Crear dataframe con lag
#             df_lag = pd.DataFrame(index=serie.index)
#             df_lag[var + f"_lag{lag}"] = serie.shift(lag)

#             # Guardar
#             climate_lags[lag][var][cuenca] = df_lag
# for var in variables:
#     for cuenca in climate_dict_derived[var].keys():

        # figMM2 = plt.figure(figsize=(14,6))
        # ax1 = figMM2.add_subplot(111)

        # for lag in LagsImprimir:
        #     serie_lag = climate_lags[lag][var][cuenca][var + f"_lag{lag}"].astype(float)
        #     ax1.plot(serie_lag, 'o', lw=2, ms=6, alpha=0.4, 
        #              color=colores[lags.index(lag)],
        #              label=f'Lag {lag}')

        # ax1.legend(loc='best')
        # ax1.set_ylabel(var)
        # ax1.set_title(cuenca)
        # ax1.grid(True)
        # plt.tight_layout()

        # figMM2.savefig(f"graphs/LAG_{var}_{cuenca}.jpg", format='jpg', dpi=600)
        # plt.show()
        # plt.close(figMM2)

#Multiscaling analysis

# Inicialización
# climate_momentos = {var: {} for var in variables}

# for var in variables:
#     if var not in climate_dict_derived:
#         print(f"Variable '{var}' no encontrada en climate_dict_derived. Se ignora.")
#         continue
    
#     for cuenca in climate_dict_derived[var].keys():
#         # Inicializamos diccionarios internos por cuenca
#         climate_momentos[var][cuenca] = {}
        
#         # Obtener la serie (CORREGIDO: sin climate_lags[lag])
#         serie = climate_dict_derived[var][cuenca].astype(float)
#         # print(var,cuenca,serie.mean())
#         # Si la serie está completamente vacía → todos los lags seguirán siendo NaN
#         if serie.isna().all():  # Solo un .all() para Series
#             for lag in lags:
#                 climate_momentos[var][cuenca][lag] = np.nan
#             continue
        
#         # Eliminamos NaN
#         values = serie.dropna().values
#         # print(var,cuenca,values.mean())
#         for lag in lags:
#             try:
#                 # momentos_val = stats.moment(values, moment=lag, center=0)
#                 momentos_val = np.mean(values**lag)
#             except Exception as e:
#                 print(f"Error calculando momento {lag} para {var}-{cuenca}: {e}")
#                 momentos_val = np.nan
            
#             # Guardamos resultados
#             climate_momentos[var][cuenca][lag] = momentos_val

# # print(climate_momentos["yav"]["Zidacos"][1])
# # print(climate_momentos["yav"]["Zidacos"][32])


# # Visualización (CORREGIDO)
# cmap = plt.cm.tab20b
# colores = [cmap(i) for i in range(20)]

# for var in variables:
#     if var not in climate_momentos:
#         print(f"Variable {var} no encontrada en climate_momentos. Se ignora.")
#         continue
    
#     plt.figure(figsize=(12, 7))
#     i = 0
    
#     for cuenca in climate_dict_derived[var].keys():
#         # Extraer datos del diccionario
#         momentos_dict = climate_momentos[var][cuenca]
#         lags_ordenados = sorted(momentos_dict.keys())
#         momentos_vals = [momentos_dict[lag] for lag in lags_ordenados]
        
#         # Plotear (con manejo de color cíclico si hay más de 20 cuencas)
#         color_idx = i % 20
#         plt.plot(lags_ordenados, momentos_vals, '-o', lw=2, 
#                 color=colores[color_idx], label=f'Cuenca {cuenca}')
#         i += 1
    
#     plt.title(f"Momentos - {var}")
#     plt.xlabel("Orden del momento")
#     plt.ylabel("Valor del momento")
#     plt.yscale('log')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"graphs/Momentos_{var}.jpg", format='jpg', dpi=600, bbox_inches='tight')
#     plt.show()


#----------------------------------------------------------------------------------
# Gráficos de autocorrelación
#----------------------------------------------------------------------------------


# Inicializamos el diccionario con la estructura requerida
# climate_acf  = {var: {} for var in variables}
# climate_pacf = {var: {} for var in variables}

# for var in variables:

#     if var not in climate_dict_derived:
#         print(f"Variable '{var}' no encontrada en climate_dict_derived. Se ignora.")
#         continue

#     for cuenca in climate_dict_derived[var].keys():

#         # Inicializamos diccionarios internos por cuenca
#         climate_acf[var][cuenca]  = {}
#         climate_pacf[var][cuenca] = {}

#         serie = climate_dict_derived[var][cuenca].astype(float)

#         # Si la serie está completamente vacía → todos los lags seguirán siendo NaN
#         if serie.isna().all():
#             for lag in lags:
#                 climate_acf[var][cuenca][lag]  = np.nan
#                 climate_pacf[var][cuenca][lag] = np.nan
#             continue

#         # Eliminamos NaN antes de calcular ACF/PACF
#         values = serie.dropna().values

#         for lag in lags:

#             try:
#                 acf_val = acf(values, nlags=lag, fft=False)[lag]
#             except:
#                 acf_val = np.nan

#             try:
#                 pacf_val = pacf(values, nlags=lag)[lag]
#             except:
#                 pacf_val = np.nan

#             # Guardamos resultados correctamente en los diccionarios
#             climate_acf[var][cuenca][lag]  = acf_val
#             climate_pacf[var][cuenca][lag] = pacf_val



# --- HEATMAPS ACF ---
# for var in variables:

#     if var not in climate_acf:
#         print(f"Variable {var} no encontrada en climate_acf. Se ignora.")
#         continue

#     # Crear matriz (DataFrame): filas = cuencas, columnas = lags
#     df_acf = pd.DataFrame({
#         cuenca: climate_acf[var][cuenca]
#         for cuenca in climate_acf[var]
#     }).T  # transpuesto para que cuencas sean filas

#     plt.figure(figsize=(12, 7))
#     sns.heatmap(df_acf, annot=True, cmap="coolwarm", center=0)
#     plt.title(f"ACF - {var}")
#     plt.xlabel("Lag")
#     plt.ylabel("Cuenca")
#     plt.tight_layout()
#     plt.savefig(f"graphs/ACF_{var}.jpg", format='jpg', dpi=600)
#     plt.show()
    



# --- HEATMAPS PACF ---
# for var in variables:

#     if var not in climate_pacf:
#         print(f"Variable {var} no encontrada en climate_pacf. Se ignora.")
#         continue

#     df_pacf = pd.DataFrame({
#         cuenca: climate_pacf[var][cuenca]
#         for cuenca in climate_pacf[var]
#     }).T

#     plt.figure(figsize=(12, 7))
#     sns.heatmap(df_pacf, annot=True, cmap="coolwarm", center=0)
#     plt.title(f"PACF - {var}")
#     plt.xlabel("Lag")
#     plt.ylabel("Cuenca")
#     plt.tight_layout()
#     plt.savefig(f"graphs/PACF_{var}.png", dpi=300)
#     plt.show()




