"""
Script QGIS para procesado de información territorial
Anexo 1 - Tesis Doctoral
Autor: Víctor Galán Vaquerizo
Director: Dr. Sergio Zubelzu
Universidad Politécnica de Madrid

Este script procesa información territorial de cuencas para obtener:
- Parámetros de Curve Number (CN)
- Propiedades físicas del suelo (arena, limo, arcilla)
- Parámetros hidráulicos (van Genuchten)
- Características geomorfológicas de la cuenca
"""

from qgis.core import QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsPointXY
import math

from qgis.core import QgsRaster, QgsRasterBlock, QgsRasterDataProvider, QgsRasterFileWriter, QgsRasterPipe, QgsRasterProjector, QgsCoordinateTransformContext, QgsRasterBandStats
from qgis.PyQt.QtCore import QVariant
import numpy as np
import os

# ===============================
# Función CNs
# ===============================
def CNs(SIOSE):
    cods = [101,102,103,104,105,106,111,112,121,122,131,132,133,140,141,142,143,144,145,
    200,210,211,212,220,221,222,223,224,231,232,241,280,290,
    300,301,302,310,311,312,313,316,320,330,331,333,334,335,336,350,351,352,353,354,
    410,411,412,413,420,421,422,
    500,510,511,514,521,522,523,
    600,
    700,701,702,703,704,
    810,811,812,813,821,822,823,830,831,832,833,834,841,842,843,844,851,852,853,854,855,856,857,858,859,860,881,882,883,884,891,892,893,894,895,896,897,
    900,911,912,913,920,921,922]
    CurveNumbers = [98,98,98,98,98,98,98,98,86,86,84,86,86,98,98,98,98,98,98,
    83,83,83,83,83,76,76,76,76,76,76,76,76,86,86,
    86,77,82,82,82,82,82,77,89,89,89,89,89,89,91,91,91,91,91,
    98,98,98,98,98,98,98,
    98,98,98,98,98,98,98,
    77,
    82,82,82,82,82,
    84,84,84,84,91,91,91,91,91,91,91,91,94,94,94,94,94,94,94,94,94,
    94,94,94,94,94,94,94]
    if SIOSE != -9999 and int(SIOSE) in cods:
        return CurveNumbers[cods.index(int(SIOSE))]
    else:
        return None


# ===============================
# Función SoilTexture
# ===============================
def SoilTexture(sand, silt, clay):
    if (sand>85 and silt+1.5*clay>=15):
        return "Sand"
    elif (sand>=85 and sand<=90 and silt+1.5*clay>=15) or (sand>=70 and sand<=85 and silt+2*clay<=30):
        return "LoamySand"
    elif (clay<=20 and silt+2*clay>30 and sand>=52) or (clay<7 and silt<50 and sand>=43 and sand<=52):
        return "SandyLoam"
    elif clay>=7 and clay<=27 and silt>=28 and silt<=50 and sand<52:
        return "Loam"
    elif (silt>=50 and clay>=12 and clay<=27) or (silt>=50 and silt<=80 and clay<=12):
        return "SiltyLoam"
    elif silt>=80 and clay<=12:
        return "Silt"
    elif clay>=20 and clay<=35 and silt<28 and sand>=45:
        return "SandyClayLoam"
    elif clay>=27 and clay<=40 and sand>=20 and sand<=45:
        return "ClayLoam"
    elif clay>=27 and clay<=40 and sand<20:
        return "SiltyClayLoam"
    elif clay>=35 and sand>=45:
        return "SandyClay"
    elif clay>=40 and silt>=40:
        return "SiltyClay"
    elif clay>=40 and sand<45 and silt<40:
        return "Clay"
    else:
        return "Error!"


# ===============================
# Función soil_param
# ===============================
def soil_param(soil):
    if soil=="Sand":
        thetas, thetar, alpha, ks, m = 0.43, 0.045, 0.145/10, 29.70*10, 0.627
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="Clay":
        thetas, thetar, alpha, ks, m = 0.38, 0.068, 0.008/10, 0.2*10, 0.083
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="Silt":
        thetas, thetar, alpha, ks, m = 0.46, 0.034, 0.016/10, 0.25*10, 0.27
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="SandyClay":
        thetas, thetar, alpha, ks, m = 0.38, 0.1, 0.027/10, 0.12*10, 0.187
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="SandyClayLoam":
        thetas, thetar, alpha, ks, m = 0.39, 0.1, 0.059/10, 1.31*10, 0.324
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="SandyLoam":
        thetas, thetar, alpha, ks, m = 0.41, 0.065, 0.075/10, 4.42*10, 0.4709
        n=1/(1-m)
    elif soil=="LoamySand":
        thetas, thetar, alpha, ks, m = 0.41, 0.057, 0.124/10, 14.50*10, 0.5614
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="Loam":
        thetas, thetar, alpha, ks, m = 0.43, 0.078, 0.036/10, 1.04*10, 0.359
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="ClayLoam":
        thetas, thetar, alpha, ks, m = 0.41, 0.095, 0.019/10, 0.26*10, 0.237
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="SiltyLoam":
        thetas, thetar, alpha, ks, m = 0.45, 0.067, 0.02/10, 0.45*10, 0.291
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    elif soil=="SiltyClayLoam":
        thetas, thetar, alpha, ks, m = 0.43, 0.089, 0.01/10, 0.07*10, 0.187
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)
    else:
        thetas, thetar, alpha, ks, m = 0.36, 0.07, 0.005/10, 0.02*10, 0.083
        thau=Neuman(ks,thetas, thetar, alpha, m)
        n=1/(1-m)

    swcfc = thetar+(thetas-thetar)*pow(1+pow(alpha*309,n),-m)
    swcpwp = thetar+(thetas-thetar)*pow(1+pow(alpha*15349,n),-m)
    #thau = 6.93/100
    return ks, thetas, thetar, thau, swcfc, swcpwp

def Neuman(ks,thetas, thetar, alpha, m):
    divisiones=10
    step=(thetas-thetar)/divisiones
    thau=0
    n=1/(1-m)
    for i in range (0,divisiones):
        thetainf=thetar+i*step
        thetasup=thetar+(i+1)*step
        thetam=(thetasup+thetainf)/2
        thetan=(thetam-thetar)/(thetas-thetar)
        thau+=ks*thetan*pow(1-pow(1-pow(thetan,1/m),m),2)*((pow(pow(thetasup,-1/m)-1,1/n)/alpha)-(pow(pow(thetainf,-1/m)-1,1/n)/alpha))
    return -thau

# ===============================
# Leer capas de QGIS
# ===============================
from qgis import processing
layer_poligonos_name = "Usos_OcaOña"  # 🔸
layer_linea_name = "Long_seg_OcaOña"          # 🔸
layer_raster_name = "MDT_clip_OcaOña"        # 🔸
raster_sand_name = "RS_SAND_0_5_OcaOna_MIN_BUF"               # 🔸
raster_silt_name = "RS_SILT_0_5_OcaOna_MIN_BUF"                # 🔸
raster_clay_name = "RS_CLAY_0_5_OcaOna_MIN_BUF"             # 🔸
layer_mask_vector_name = "Basin_v_OcaOña"  # <-- cámbialo por el nombre exacto de tu capa


project = QgsProject.instance()
layer_poligonos = project.mapLayersByName(layer_poligonos_name)[0]
layer_linea = project.mapLayersByName(layer_linea_name)[0]
layer_raster = project.mapLayersByName(layer_raster_name)[0]
raster_sand = project.mapLayersByName(raster_sand_name)[0]
raster_silt = project.mapLayersByName(raster_silt_name)[0]
raster_clay = project.mapLayersByName(raster_clay_name)[0]



from qgis import processing

# Nombre de la capa vectorial de máscara (ya existente en el proyecto)

layer_mask_vector = project.mapLayersByName(layer_mask_vector_name)[0]

if not layer_mask_vector.isValid():
    raise Exception(f"❌ No se pudo cargar la capa vectorial de máscara: {layer_mask_vector_name}")

# Directorio del proyecto (para guardar los resultados)
project_dir = os.path.dirname(QgsProject.instance().fileName())

# Función para recortar cada raster usando la capa vectorial
def clip_raster(input_layer, mask_layer, output_name):
    out_path = os.path.join(project_dir, f"{output_name}.tif")
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': input_layer.source(),
        'MASK': mask_layer,
        'SOURCE_CRS': input_layer.crs(),
        'TARGET_CRS': input_layer.crs(),
        'NODATA': -9999,
        'ALPHA_BAND': False,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'SET_RESOLUTION': False,
        'OUTPUT': out_path
    })
    clipped_layer = QgsRasterLayer(out_path, output_name)
    if clipped_layer.isValid():
        QgsProject.instance().addMapLayer(clipped_layer)
        print(f"✅ Capa recortada creada: {output_name}")
    else:
        raise Exception(f"❌ Error al recortar la capa {output_name}.")
    return clipped_layer

# Recortar las tres capas de textura (arena, limo, arcilla)
raster_sand_clip = clip_raster(raster_sand, layer_mask_vector, "raster_sand_clip")
raster_silt_clip = clip_raster(raster_silt, layer_mask_vector, "raster_silt_clip")
raster_clay_clip = clip_raster(raster_clay, layer_mask_vector, "raster_clay_clip")

# Reemplazar las variables originales por las capas recortadas
raster_sand = raster_sand_clip
raster_silt = raster_silt_clip
raster_clay = raster_clay_clip


# ===============================
# CN ponderado
# ===============================
import statistics

def vector_std(layer, field_name):
    """Calcula la desviación típica de un atributo numérico en una capa vectorial."""
    values = []
    for f in layer.getFeatures():
        val = f[field_name]
        if val is not None:
            values.append(val)
    if len(values) == 0:
        return None
    return statistics.pstdev(values)


suma_area = 0
suma_CN_area = 0
CNN=[]
for f in layer_poligonos.getFeatures():
    id_cobertu = f["ID_COBERTU"]
    #if "ID_COBERTU" in f.fields().names() and f["ID_COBERTU"] not in [None, '', -9999]:
    #    id_cobertu = f["ID_COBERTU"]
    #elif "CODIIGE2" in f.fields().names() and f["CODIIGE2"] not in [None, '', -9999]:
    #    id_cobertu = f["CODIIGE2"]
    #elif "CODIIGE" in f.fields().names() and f["CODIIGE"] not in [None, '', -9999]:
    #    id_cobertu = f["CODIIGE"]
    #else:
    #    id_cobertu = None
    area = f.geometry().area()
    cn = CNs(id_cobertu)
    if cn:
        suma_CN_area += cn * area
        suma_area += area
        CNN.append(cn)
CN_ponderado = suma_CN_area / suma_area if suma_area > 0 else None
CN_DT=np.std(CNN)
# ===============================
# Longitud de línea
# ===============================
longitud_total = sum([feat.geometry().length() for feat in layer_linea.getFeatures()])

# ===============================
# Valores min y max del raster
# ===============================
prov = layer_raster.dataProvider()
stats = prov.bandStatistics(1)
valor_min, valor_max = stats.minimumValue, stats.maximumValue

# ===============================
# Medias de tres rasters
# ===============================
def raster_mean(layer):
    prov = layer.dataProvider()
    stats = prov.bandStatistics(1)
    return stats.mean

def raster_DT(layer):
    prov = layer.dataProvider()
    stats = prov.bandStatistics(1)
    return stats.stdDev

mean_sand = raster_mean(raster_sand) / 10
mean_silt = raster_mean(raster_silt) / 10
mean_clay = raster_mean(raster_clay) / 10

# ===============================
# Determinar tipo de suelo y parámetros
# ===============================
soil_type = SoilTexture(mean_sand, mean_silt, mean_clay)
params = soil_param(soil_type)

# ===============================
# Extensión geográfica (EPSG:25830 → EPSG:4326)
# ===============================
crs_src = layer_poligonos.crs()
crs_dest = QgsCoordinateReferenceSystem("EPSG:4326")
transform = QgsCoordinateTransform(crs_src, crs_dest, QgsProject.instance())

extent = layer_poligonos.extent()
min_point = transform.transform(QgsPointXY(extent.xMinimum(), extent.yMinimum()))
max_point = transform.transform(QgsPointXY(extent.xMaximum(), extent.yMaximum()))

lat_min, lon_min = min_point.y(), min_point.x()
lat_max, lon_max = max_point.y(), max_point.x()




# ===============================
# 🔹 Calcular parámetros píxel a píxel
# ===============================

from osgeo import gdal

# Aseguramos que las tres capas tengan el mismo tamaño, extensión y CRS
extent = raster_sand.extent()
cols = raster_sand.width()
rows = raster_sand.height()
crs = raster_sand.crs()

prov_sand = raster_sand.dataProvider()
prov_silt = raster_silt.dataProvider()
prov_clay = raster_clay.dataProvider()

# Leemos los bloques completos
sand_block = prov_sand.block(1, extent, cols, rows)
silt_block = prov_silt.block(1, extent, cols, rows)
clay_block = prov_clay.block(1, extent, cols, rows)

# Creamos arrays numpy vacíos del mismo tamaño que el ráster de referencia
ks_arr = np.full((rows, cols), -9999, dtype=float)
thetas_arr = np.full((rows, cols), -9999, dtype=float)
thetar_arr = np.full((rows, cols), -9999, dtype=float)
thau_arr = np.full((rows, cols), -9999, dtype=float)
swcfc_arr = np.full((rows, cols), -9999, dtype=float)
swcpwp_arr = np.full((rows, cols), -9999, dtype=float)

# Recorremos todos los píxeles
for i in range(rows):
    for j in range(cols):
        sand = sand_block.value(i, j)
        silt = silt_block.value(i, j)
        clay = clay_block.value(i, j)

        # Ignorar NoData o valores inválidos
        if sand == -9999 or silt == -9999 or clay == -9999:
            continue

        sand /= 10
        silt /= 10
        clay /= 10

        soil_type = SoilTexture(sand, silt, clay)
        ks, thetas, thetar, thau, swcfc, swcpwp = soil_param(soil_type)

        ks_arr[i, j] = ks
        thetas_arr[i, j] = thetas
        thetar_arr[i, j] = thetar
        thau_arr[i, j] = thau
        swcfc_arr[i, j] = swcfc
        swcpwp_arr[i, j] = swcpwp


# ===============================
# 🔹 Función para guardar ráster idéntico al original
# ===============================
def save_raster(array, base_layer, output_path):
    """Guarda una matriz numpy como raster GeoTIFF con la misma geometría que otra capa."""
    provider = base_layer.dataProvider()
    extent = base_layer.extent()
    cols = base_layer.width()
    rows = base_layer.height()

    # Resolución del ráster
    xres = extent.width() / cols
    yres = extent.height() / rows

    # GeoTransform
    geotransform = (extent.xMinimum(), xres, 0, extent.yMaximum(), 0, -yres)

    # Crear ráster con GDAL
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    output_raster.SetProjection(base_layer.crs().toWkt())

    output_band = output_raster.GetRasterBand(1)
    output_band.WriteArray(array)
    output_band.SetNoDataValue(-9999)
    output_band.FlushCache()
    output_raster = None


# ===============================
# 🔹 Guardar las seis capas
# ===============================
project_path = QgsProject.instance().fileName()
project_dir = os.path.dirname(project_path)

if not project_dir:
    raise Exception("⚠️ Guarda el proyecto QGIS antes de ejecutar el script (se necesita su ruta).")

save_raster(ks_arr, raster_sand, os.path.join(project_dir, "param_ks.tif"))
save_raster(thetas_arr, raster_sand, os.path.join(project_dir, "param_thetas.tif"))
save_raster(thetar_arr, raster_sand, os.path.join(project_dir, "param_thetar.tif"))
save_raster(thau_arr, raster_sand, os.path.join(project_dir, "param_thau.tif"))
save_raster(swcfc_arr, raster_sand, os.path.join(project_dir, "param_swcfc.tif"))
save_raster(swcpwp_arr, raster_sand, os.path.join(project_dir, "param_swcpwp.tif"))


# ===============================
# 🔹Cargar en QGIS
# ===============================
for name in ["param_ks", "param_thetas", "param_thetar", "param_thau", "param_swcfc", "param_swcpwp"]:
    path = os.path.join(project_dir, f"{name}.tif")
    rl = QgsRasterLayer(path, name)
    if rl.isValid():
        QgsProject.instance().addMapLayer(rl)

print("✅ Se han generado y cargado los 6 rásteres de parámetros de suelo (mismo tamaño que raster_sand).")


# ===============================
# 🔹Calculamos media y DT
# ===============================
param_ks = project.mapLayersByName("param_ks")[0]
param_thetas = project.mapLayersByName("param_thetas")[0]
param_thetar = project.mapLayersByName("param_thetar")[0]
param_thau = project.mapLayersByName("param_thau")[0]
param_swcfc = project.mapLayersByName("param_swcfc")[0]
param_swcpwp = project.mapLayersByName("param_swcpwp")[0]

mean_ks = raster_mean(param_ks)
DT_ks = raster_DT(param_ks)
mean_thetas = raster_mean(param_thetas)
DT_thetas = raster_DT(param_thetas)
mean_thetar = raster_mean(param_thetar)
DT_thetar = raster_DT(param_thetar)
mean_thau = raster_mean(param_thau)
DT_thau = raster_DT(param_thau)
mean_swcfc = raster_mean(param_swcfc)
DT_swcfc = raster_DT(param_swcfc)
mean_swcpwp = raster_mean(param_swcpwp)
DT_swcpwp = raster_DT(param_swcpwp)

# ===============================
# Resultados finales
# ===============================
print("========== RESULTADOS ==========")
if CN_ponderado:
    print(f"CN ponderado: {CN_ponderado:.2f}")
    print(f"Desviación Tipica CN: {CN_DT:.2f}")
    print(f"Longitud total línea: {longitud_total:.2f}")
    print(f"Area cuenca: {suma_area:.2f}")
    print(f"Ráster - valor mínimo: {valor_min:.2f}")
    print(f"Ráster - valor máximo: {valor_max:.2f}")
    print(f"Media arena: {mean_sand:.2f}, limo: {mean_silt:.2f}, arcilla: {mean_clay:.2f}")
    print(f"Tipo de suelo: {soil_type}")
    print(f"Parámetros (ks, thetas, thetar, thau, swcfc, swcpwp): {params}")
    print(f"Media Ks: {mean_ks:.2f}")
    print(f"DT Ks: {DT_ks:.2f}")
    print(f"Media thetas: {mean_thetas:.2f}")
    print(f"DT thetas: {DT_thetas:.2f}")
    print(f"Media thetar: {mean_thetar:.2f}")
    print(f"DT thetar: {DT_thetar:.2f}")
    print(f"Media thau: {mean_thau:.2f}")
    print(f"DT thau: {DT_thau:.2f}")
    print(f"Media swcfc: {mean_swcfc:.2f}")
    print(f"DT swcfc: {DT_swcfc:.2f}")
    print(f"Media swcpwp: {mean_swcpwp:.2f}")
    print(f"DT swcpwp: {DT_swcpwp:.2f}")
    print(f"Latitud mínima: {lat_min:.6f}, Longitud mínima: {lon_min:.6f}")
    print(f"Latitud máxima: {lat_max:.6f}, Longitud máxima: {lon_max:.6f}")
    print("================================")

import csv
import os

# ===============================
# 🔹 Ruta y nombre del archivo CSV
# ===============================
out_csv_dir = QFileDialog.getExistingDirectory(None, "Selecciona carpeta para guardar el CSV")
if not out_csv_dir:
    raise Exception("No se ha seleccionado carpeta de salida")

csv_name, _ = QFileDialog.getSaveFileName(None, "Nombre del archivo CSV", os.path.join(out_csv_dir, "resultados.csv"), "CSV Files (*.csv)")

if not csv_name:
    raise Exception("No se ha especificado un nombre para el CSV")

# ===============================
# 🔹 Datos a guardar
# ===============================
data = {
    "CN_ponderado": CN_ponderado,
    "CN_DT": CN_DT,
    "area_cuenca": suma_area,
    "longitud_total": longitud_total,
    "z_min": valor_min,
    "z_max": valor_max,
    "mean_sand": mean_sand,
    "mean_silt": mean_silt,
    "mean_clay": mean_clay,
    "soil_type": soil_type,
    "ks_st": params[0],
    "thetas_st": params[1],
    "thetar_st": params[2],
    "thau_st": params[3],
    "swcfc_st": params[4],
    "swcpwp_st": params[5],
    "mean_ks": mean_ks,
    "DT_ks": DT_ks,
    "mean_thetas": mean_thetas,
    "DT_thetas": DT_thetas,
    "mean_thetar": mean_thetar,
    "DT_thetar": DT_thetar,
    "mean_thau": mean_thau,
    "DT_thau": DT_thau,
    "mean_swcfc": mean_swcfc,
    "DT_swcfc": DT_swcfc,
    "mean_swcpwp": mean_swcpwp,
    "DT_swcpwp": DT_swcpwp,
    "lon_min": lon_min,
    "lon_max": lon_max,
    "lat_min": lat_min,
    "lat_max": lat_max
}

# ===============================
# 🔹 Guardar en CSV
# ===============================
with open(csv_name, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(data.keys())
    writer.writerow(data.values())

iface.messageBar().pushMessage("Éxito", f"Archivo CSV guardado en: {csv_name}", level=Qgis.Success)
