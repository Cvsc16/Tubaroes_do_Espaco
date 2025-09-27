
#!/usr/bin/env python3
# Gera raster de probabilidade e salva GeoTIFF

from pathlib import Path
import numpy as np
import rioxarray as rxr
import joblib
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))
PRO = ROOT/"data"/"processed"
TILES = ROOT/"data"/"tiles"
TILES.mkdir(parents=True, exist_ok=True)

model = joblib.load(PRO/"model_xgb.pkl")

sst = rxr.open_rasterio(PRO/"sst_gradients.tif").squeeze()
chl = rxr.open_rasterio(PRO/"chlorophyll.tif").squeeze()

arr_sst = sst.values
arr_chl = chl.values
mask = np.isnan(arr_sst) | np.isnan(arr_chl)

H, W = arr_sst.shape
stack = np.stack([arr_sst, arr_chl], axis=-1)
stack[mask] = 0.0

proba = model.predict_proba(stack.reshape(-1, 2))[:,1].reshape(H, W)
proba = np.where(mask, np.nan, proba)

transform = sst.rio.transform()
meta = {
    "driver": "GTiff",
    "height": H,
    "width": W,
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:4326",
    "transform": transform
}
out_path = TILES/"hotspots_probability.tif"
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(proba.astype("float32"), 1)

print(f"Mapa salvo em {out_path}")
