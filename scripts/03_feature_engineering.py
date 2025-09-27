
#!/usr/bin/env python3
# Extrai features tabulares e cria presenças/pseudo-ausências (placeholder)

from pathlib import Path
import yaml, numpy as np, pandas as pd
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))
PRO = ROOT/"data"/"processed"
OUT = ROOT/"data"/"processed"

sst = rxr.open_rasterio(PRO/"sst_gradients.tif", masked=True).squeeze()
chl = rxr.open_rasterio(PRO/"chlorophyll.tif", masked=True).squeeze()

bbox = CFG["aoi"]["bbox"]

def random_points(n=300):
    xs = np.random.uniform(bbox[0], bbox[2], n)
    ys = np.random.uniform(bbox[1], bbox[3], n)
    return [Point(x,y) for x,y in zip(xs,ys)]

# TODO: substitua por trilhas reais
presences = gpd.GeoDataFrame({"label":[1]*100}, geometry=random_points(100), crs="EPSG:4326")
absences  = gpd.GeoDataFrame({"label":[0]*300}, geometry=random_points(300), crs="EPSG:4326")

def sample(da, gdf, name):
    vals = []
    for pt in gdf.geometry:
        try:
            val = float(da.sel(x=pt.x, y=pt.y, method="nearest").values)
        except Exception:
            val = np.nan
        vals.append(val)
    gdf[name] = vals
    return gdf

presences = sample(sst, presences, "sst")
presences = sample(chl, presences, "chl")
absences  = sample(sst, absences, "sst")
absences  = sample(chl, absences, "chl")

df = pd.concat([presences.drop(columns="geometry"), absences.drop(columns="geometry")], ignore_index=True).dropna()
df.to_csv(OUT/"dataset.csv", index=False)
print("Dataset salvo em data/processed/dataset.csv")
